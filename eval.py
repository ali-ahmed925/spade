"""SPADE evaluation script.

Evaluates image-level AUROC, pixel/patch-level AUROC, and generates
localization heatmaps on the MVTec test split.

Usage:
    python eval.py --checkpoint checkpoints/spade_best.pt
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

from data.mvtec_dataset import MVTecDataset
from models.spade import SPADE
from utils.heatmap import patches_to_heatmap, overlay_heatmap, save_heatmap
from utils.logging import get_logger
from utils.metrics import compute_image_auroc, compute_pixel_auroc


def load_config() -> dict:
    cfg = {}
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    for name in ("model", "data", "train"):
        with open(os.path.join(config_dir, f"{name}.yaml")) as f:
            cfg.update(yaml.safe_load(f))
    return cfg


def save_localization_visualization(
    image_path: str,
    image_np: np.ndarray,
    gt_mask: np.ndarray,
    heatmap: np.ndarray,
    image_score: float,
    label: int,
    output_path: str,
) -> None:
    """Save side-by-side localization visualization.

    Shows: original image, GT mask, predicted heatmap, overlay.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        f"{Path(image_path).name}\n"
        f"Label: {'Anomaly' if label == 1 else 'Normal'} | "
        f"Score: {image_score:.4f}",
        fontsize=12,
    )

    # Original image
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Ground truth mask
    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    # Predicted heatmap (use 'hot' colormap for better visualization)
    im = axes[2].imshow(heatmap, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("Predicted Heatmap")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    # Overlay (use 'hot' colormap)
    overlay = overlay_heatmap(image_np, heatmap, colormap_name="hot")
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay (Red = Anomaly)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def evaluate(
    model: SPADE,
    loader: DataLoader,
    device: torch.device,
    image_size: int,
    patch_size: int,
    save_dir: str | None = None,
    save_visualizations: bool = False,
) -> dict[str, float]:
    """Run evaluation on the test set.

    Args:
        save_dir: If provided, saves heatmaps here.
        save_visualizations: If True, saves side-by-side visualizations (image + GT + heatmap + overlay).

    Returns:
        dict with image_auroc and pixel_auroc.
    """
    model.eval()

    all_labels: list[int] = []
    all_image_scores: list[float] = []
    all_masks: list[np.ndarray] = []
    all_heatmaps: list[np.ndarray] = []
    all_image_info: list[dict] = []  # Store image paths and scores for detailed output

    for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
        images = batch["image"].to(device)
        labels = batch["label"]
        masks = batch["mask"]
        paths = batch["path"]

        outputs = model(images)  # No update_stats in eval
        patch_scores = outputs["patch_scores"]  # Changed from patch_logits

        image_scores = model.get_image_score(patch_scores)

        for i in range(images.size(0)):
            image_score = float(image_scores[i].cpu())
            label = int(labels[i])
            image_path = paths[i] if isinstance(paths, (list, tuple)) else paths
            
            all_labels.append(label)
            all_image_scores.append(image_score)
            
            # Store image info for detailed output
            path_obj = Path(image_path)
            defect_type = path_obj.parent.name
            image_name = path_obj.stem
            all_image_info.append({
                "path": str(image_path),
                "defect_type": defect_type,
                "image_name": image_name,
                "score": image_score,
                "label": label,
            })

            # CRITICAL: Use RAW scores for pixel AUROC computation (no per-image normalization)
            # Per-image normalization artificially inflates pixel AUROC by stretching each
            # image's score distribution independently, removing global calibration.
            patch_scores_raw = patch_scores[i].detach().cpu()
            hmap_raw = patches_to_heatmap(
                patch_scores_raw,
                image_size=image_size,
                patch_size=patch_size,
                normalize=False,  # NO normalization for metric computation
            )
            all_heatmaps.append(hmap_raw)
            
            # For visualization: create normalized version (only if saving)
            if save_dir is not None:
                # Normalize patch_scores to [0, 1] for visualization only
                patch_scores_np = patch_scores_raw.numpy()
                p5, p95 = np.percentile(patch_scores_np, [5, 95])
                patch_scores_clipped = np.clip(patch_scores_np, p5, p95)
                if p95 - p5 > 1e-8:
                    patch_scores_normalized = (patch_scores_clipped - p5) / (p95 - p5)
                else:
                    patch_scores_normalized = np.zeros_like(patch_scores_clipped)
                
                hmap_viz = patches_to_heatmap(
                    torch.from_numpy(patch_scores_normalized),
                    image_size=image_size,
                    patch_size=patch_size,
                    normalize=True,  # Normalize for visualization
                    percentile_clip=(0, 100),  # Already normalized
                )
            else:
                hmap_viz = None

            if isinstance(masks[i], torch.Tensor):
                gt_mask = masks[i].numpy()
            else:
                gt_mask = masks[i]

            all_masks.append(gt_mask)

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                
                # Extract defect type and image name from path
                image_path = paths[i] if isinstance(paths, (list, tuple)) else paths
                path_obj = Path(image_path)
                
                # Defect type is the parent folder name (e.g., "good", "broken_large", "contamination")
                defect_type = path_obj.parent.name
                image_name = path_obj.stem  # filename without extension
                
                # Create descriptive filenames: {defect_type}_{image_name}_{suffix}.png
                heatmap_filename = f"{defect_type}_{image_name}_heatmap.png"
                localization_filename = f"{defect_type}_{image_name}_localization.png"
                
                # Save normalized heatmap for visualization (hmap_viz was created above)
                if hmap_viz is not None:
                    save_heatmap(hmap_viz, os.path.join(save_dir, heatmap_filename), colormap="hot")
                    
                    # Save full visualization if requested
                    if save_visualizations:
                        # Load original image (not transformed)
                        image_np = cv2.imread(image_path)
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                        image_np = cv2.resize(image_np, (image_size, image_size))
                        
                        viz_path = os.path.join(save_dir, localization_filename)
                        save_localization_visualization(
                            image_path=image_path,
                            image_np=image_np,
                            gt_mask=gt_mask,
                            heatmap=hmap_viz,  # Use normalized version for visualization
                            image_score=float(image_scores[i].cpu()),
                            label=int(labels[i]),
                            output_path=viz_path,
                        )

    labels_arr = np.array(all_labels)
    scores_arr = np.array(all_image_scores)
    masks_arr = np.stack(all_masks)
    heatmaps_arr = np.stack(all_heatmaps)

    results = {}
    results["image_auroc"] = compute_image_auroc(labels_arr, scores_arr)

    if masks_arr.max() > 0:
        results["pixel_auroc"] = compute_pixel_auroc(masks_arr, heatmaps_arr)
    else:
        results["pixel_auroc"] = float("nan")
    
    # ── Print detailed image scores ──
    print("\n" + "=" * 80)
    print("IMAGE-LEVEL ANOMALY SCORES")
    print("=" * 80)
    print(f"{'Defect Type':<20} {'Image Name':<30} {'Score':<12} {'Label':<10}")
    print("-" * 80)
    
    # Sort by score (descending) to see most anomalous first
    sorted_info = sorted(all_image_info, key=lambda x: x["score"], reverse=True)
    
    for info in sorted_info:
        label_str = "Anomaly" if info["label"] == 1 else "Normal"
        print(f"{info['defect_type']:<20} {info['image_name']:<30} {info['score']:<12.6f} {label_str:<10}")
    
    print("-" * 80)
    print(f"Total images: {len(all_image_info)}")
    print(f"Normal images: {sum(1 for info in all_image_info if info['label'] == 0)}")
    print(f"Anomaly images: {sum(1 for info in all_image_info if info['label'] == 1)}")
    print(f"Score range: [{min(all_image_scores):.6f}, {max(all_image_scores):.6f}]")
    print("=" * 80 + "\n")
    
    # Save scores to CSV file if save_dir is provided
    if save_dir is not None:
        import csv
        csv_path = os.path.join(save_dir, "image_scores.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["defect_type", "image_name", "image_path", "anomaly_score", "label", "is_anomaly"])
            for info in sorted_info:
                writer.writerow([
                    info["defect_type"],
                    info["image_name"],
                    info["path"],
                    f"{info['score']:.6f}",
                    info["label"],
                    "Anomaly" if info["label"] == 1 else "Normal",
                ])
        print(f"Saved detailed scores to: {csv_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="SPADE Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--save_heatmaps", type=str, default=None, help="Dir to save heatmaps (raw heatmap PNGs)")
    parser.add_argument("--save_visualizations", action="store_true", help="Save full localization visualizations (image + GT + heatmap + overlay)")
    parser.add_argument("--log_wandb", action="store_true", help="Log results to wandb")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="Wandb run ID to log to (if resuming)")
    args = parser.parse_args()

    cfg = load_config()
    logger = get_logger("eval")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Initialize wandb (optional) ──
    if args.log_wandb:
        if args.wandb_run_id:
            wandb.init(id=args.wandb_run_id, resume="must")
        else:
            wandb.init(
                project=cfg.get("wandb", {}).get("project", "spade-anomaly-detection"),
                name=f"eval_{cfg['dataset']['category']}",
                job_type="eval",
            )
        logger.info("Logging evaluation to Weights & Biases")

    # ── Dataset ──
    dataset = MVTecDataset(
        root=cfg["dataset"]["root"],
        category=cfg["dataset"]["category"],
        split="test",
        image_size=cfg["vit"]["image_size"],
        patch_size=cfg["vit"]["patch_size"],
        synthetic_method=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["dataset"]["num_workers"],
        pin_memory=True,
    )
    logger.info(f"Test samples: {len(dataset)}")

    # ── Model ──
    model = SPADE(
        blip2_model_name=cfg["blip2"]["model_name"],
        llm_embed_dim=cfg["projection"]["output_dim"],
        # HPA parameters
        hpa_n_max=cfg["hpa"]["n_max"],
        hpa_n_min=cfg["hpa"]["n_min"],
        hpa_t_steps=cfg["hpa"]["t_steps"],
        hpa_w=cfg["hpa"]["w"],
        hpa_p1=cfg["hpa"]["p1"],
        hpa_p2=cfg["hpa"]["p2"],
        # Scoring parameters
        score_alpha=cfg["scoring"]["alpha"],
        score_beta=cfg["scoring"]["beta"],
        score_lambda=cfg["scoring"]["lambda"],
        mahalanobis_gamma=cfg["scoring"]["mahalanobis_gamma"],
        mahalanobis_reg=cfg["scoring"]["mahalanobis_reg"],
        # Normal statistics parameters
        normal_stats_buffer_size=cfg["normal_stats"]["buffer_size"],
        normal_stats_update_frequency=cfg["normal_stats"]["update_frequency"],
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        # Load only trainable parameters (Q-Former + custom heads)
        # Vision encoder will remain frozen from BLIP-2 initialization
        model.load_state_dict(state["model_state_dict"], strict=False)
        checkpoint_epoch = state.get("epoch", "unknown")
        logger.info(f"Loaded checkpoint: {args.checkpoint} (epoch: {checkpoint_epoch})")
        if "config" in state:
            logger.info(f"Checkpoint config: {state['config']}")
    else:
        # Legacy format: assume full state_dict
        model.load_state_dict(state, strict=False)
        checkpoint_epoch = "unknown"
        logger.info(f"Loaded checkpoint: {args.checkpoint} (legacy format)")

    # ── Evaluate ──
    results = evaluate(
        model, loader, device,
        image_size=cfg["vit"]["image_size"],
        patch_size=cfg["vit"]["patch_size"],
        save_dir=args.save_heatmaps,
        save_visualizations=args.save_visualizations,
    )
    logger.info(f"Image AUROC: {results['image_auroc']:.4f}")
    logger.info(f"Pixel AUROC: {results['pixel_auroc']:.4f}")

    # ── Log to wandb ──
    if args.log_wandb:
        wandb.log({
            "eval/image_auroc": results["image_auroc"],
            "eval/pixel_auroc": results["pixel_auroc"],
            "eval/checkpoint": args.checkpoint,
            "eval/checkpoint_epoch": checkpoint_epoch,
        })
        wandb.run.summary.update({
            "best_image_auroc": results["image_auroc"],
            "best_pixel_auroc": results["pixel_auroc"],
        })
        wandb.finish()
        logger.info("Logged evaluation metrics to wandb")


if __name__ == "__main__":
    main()
