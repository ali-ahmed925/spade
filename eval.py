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
from models.builder import checkpoint_path, describe, load_spade
from models.spade import SPADE
from utils.heatmap import patches_to_heatmap, overlay_heatmap, save_heatmap
from utils.logging import get_logger
from utils.metrics import compute_image_auroc, compute_pixel_auroc, compute_pro
from models.tiling import (
    crops_from_image,
    fine_statistics,
    image_score_from_map,
    stitch_tile_scores,
    tile_boxes,
)
from utils.heatmap import smooth_map


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
        f"Label: {'Anomaly' if label == 1 else ('Normal' if label == 0 else 'Unknown')} | "
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

@torch.no_grad()
def _dense_tiling_map(
    model: SPADE,
    image_path: str,
    tiling: dict,
    image_size: int,
    device: torch.device,
    smooth_sigma: float = 0.0,
) -> np.ndarray:
    """Score one image by dense overlapping tiles of the NATIVE-resolution file.

    This is the Phase 1 ceiling test: the upper bound on what any adaptive
    refinement scheme could achieve, since it refines everywhere.

    The model, weights and score composition are identical to the coarse path —
    only the input scale and the (scale-matched) Mahalanobis statistics change,
    so the comparison isolates resolution.
    """
    from PIL import Image as PILImage

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"cannot read {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    native = image_rgb.shape[0]

    boxes = tile_boxes(native, grid=tiling["grid"], overlap=tiling["overlap"])
    crops = crops_from_image(image_rgb, boxes, out_size=image_size)
    batch = torch.stack(
        [tiling["transform"](PILImage.fromarray(c)) for c in crops]
    ).to(device)

    # fine_statistics validates the incoming statistics on entry; checking the
    # scorer inside the context would be useless, since the swap sets
    # is_initialized=True regardless of whether the stats were ever fitted.
    # fine_statistics validates the incoming statistics on entry — checking the
    # scorer inside the block would be useless, since the swap has by then set
    # is_initialized=True regardless of whether the stats were ever fitted.
    with fine_statistics(model, tiling["stats"]):
        outputs = model(batch)
        tile_scores = outputs["patch_scores"]

    fused = stitch_tile_scores(tile_scores, boxes, native_size=native, canvas_size=image_size)
    return smooth_map(fused, smooth_sigma)


@torch.no_grad()
def evaluate(
    model: SPADE,
    loader: DataLoader,
    device: torch.device,
    image_size: int,
    patch_size: int,
    save_dir: str | None = None,
    save_visualizations: bool = False,
    smooth_sigma: float = 0.0,
    tiling: dict | None = None,
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
            # Gaussian smoothing IS applied here (it is a fixed, image-independent
            # filter, so it does not leak calibration) because every comparable
            # method smooths, and because an unsmoothed baseline would confound
            # any resolution experiment.
            patch_scores_raw = patch_scores[i].detach().cpu()
            if tiling is not None:
                hmap_raw = _dense_tiling_map(
                    model=model,
                    image_path=paths[i] if isinstance(paths, (list, tuple)) else paths,
                    tiling=tiling,
                    image_size=image_size,
                    device=device,
                    smooth_sigma=smooth_sigma,
                )
                # Image score must come from the same fused evidence
                all_image_scores[-1] = image_score_from_map(hmap_raw)
                all_image_info[-1]["score"] = all_image_scores[-1]
            else:
                hmap_raw = patches_to_heatmap(
                    patch_scores_raw,
                    image_size=image_size,
                    patch_size=patch_size,
                    normalize=False,  # NO normalization for metric computation
                    smooth_sigma=smooth_sigma,
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
        # PRO weights every defect region equally, so a method that only finds
        # large defects cannot hide behind a good pixel AUROC.
        # It is a secondary metric: a failure here must not discard an entire
        # evaluation run, so it is reported as nan with the cause surfaced.
        try:
            results["pro"] = compute_pro(masks_arr, heatmaps_arr)
        except Exception as exc:  # noqa: BLE001 - report, do not lose the run
            print(f"[warn] PRO computation failed ({type(exc).__name__}: {exc}); reporting nan")
            results["pro"] = float("nan")
    else:
        results["pixel_auroc"] = float("nan")
        results["pro"] = float("nan")
    
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


def evaluate_single_image(
    model: SPADE,
    image_path: str,
    device: torch.device,
    image_size: int,
    patch_size: int,
    save_dir: str | None = None,
) -> float:
    """Run inference on a single image, print score, and optionally save heatmap."""
    from data.transforms import get_eval_transforms

    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_np = cv2.resize(image_np, (image_size, image_size))

    from PIL import Image as PILImage
    transform = get_eval_transforms(image_size)
    image_tensor = transform(PILImage.fromarray(image_np)).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)

    patch_scores = outputs["patch_scores"]  # (1, N)
    image_score = float(model.get_image_score(patch_scores).cpu())

    print(f"\nImage : {image_path}")
    print(f"Score : {image_score:.6f}")

    # Normalize for visualization only
    patch_scores_np = patch_scores[0].detach().cpu().numpy()
    p5, p95 = np.percentile(patch_scores_np, [5, 95])
    patch_scores_clipped = np.clip(patch_scores_np, p5, p95)
    if p95 - p5 > 1e-8:
        patch_scores_norm = (patch_scores_clipped - p5) / (p95 - p5)
    else:
        patch_scores_norm = np.zeros_like(patch_scores_clipped)

    hmap_viz = patches_to_heatmap(
        torch.from_numpy(patch_scores_norm),
        image_size=image_size,
        patch_size=patch_size,
        normalize=True,
        percentile_clip=(0, 100),
    )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        stem = Path(image_path).stem
        gt_mask = np.zeros((image_size, image_size), dtype=np.float32)
        save_localization_visualization(
            image_path=image_path,
            image_np=image_np,
            gt_mask=gt_mask,
            heatmap=hmap_viz,
            image_score=image_score,
            label=-1,
            output_path=os.path.join(save_dir, f"{stem}_localization.png"),
        )
        save_heatmap(hmap_viz, os.path.join(save_dir, f"{stem}_heatmap.png"), colormap="hot")
        print(f"Saved  : {save_dir}/{stem}_heatmap.png")
        print(f"Saved  : {save_dir}/{stem}_localization.png")

    return image_score


def main() -> None:
    parser = argparse.ArgumentParser(description="SPADE Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, default=None, help="Single image path (skips full dataset eval)")
    parser.add_argument("--save_heatmaps", type=str, default=None, help="Dir to save heatmaps (raw heatmap PNGs)")
    parser.add_argument("--save_visualizations", action="store_true", help="Save full localization visualizations (image + GT + heatmap + overlay)")
    parser.add_argument("--log_wandb", action="store_true", help="Log results to wandb")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="Wandb run ID to log to (if resuming)")
    parser.add_argument(
        "--split", type=str, default="all", choices=["all", "val", "heldout"],
        help="Which half of the test set to score. 'val' is the half used for "
             "checkpoint selection during training (optimistic); 'heldout' is the "
             "half never seen by model selection (the honest number); 'all' is both.",
    )
    parser.add_argument(
        "--smooth-sigma", type=float, default=None,
        help="Gaussian sigma applied to the anomaly map before metrics. Defaults "
             "to config scoring.smooth_sigma. Comparable methods use ~4 at 224px.",
    )
    parser.add_argument(
        "--dense-tiling", action="store_true",
        help="Phase 1 ceiling test: score NxN overlapping crops of the native-"
             "resolution image and fuse. Requires fine-scale statistics.",
    )
    parser.add_argument("--grid", type=int, default=3, help="Tiles per axis for --dense-tiling")
    parser.add_argument("--overlap", type=float, default=0.5, help="Tile overlap for --dense-tiling")
    parser.add_argument(
        "--fine-stats", type=str, default=None,
        help="Path to fine-scale statistics (default: checkpoints/<cat>/fine_stats_g<grid>_o<overlap>.pt)",
    )
    parser.add_argument(
        "--positional", action="store_true",
        help="P1: score with per-position Mahalanobis means instead of one global "
             "mean. Targets structural/positional defects (transistor misplaced, "
             "cable_swap) that a position-agnostic Gaussian cannot see. Requires "
             "scripts/fit_positional_statistics.py.",
    )
    parser.add_argument(
        "--positional-stats", type=str, default=None,
        help="Path to positional statistics (default: checkpoints/<cat>/positional_stats.pt)",
    )
    parser.add_argument(
        "--position-blend", type=float, default=1.0,
        help="1.0 = per-position means; 0.0 = global mean (the control arm, which "
             "reproduces current behaviour); values between interpolate.",
    )
    args = parser.parse_args()

    cfg = load_config()
    logger = get_logger("eval")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Organize outputs by category ──
    category = cfg["dataset"]["category"]
    if args.save_heatmaps is not None:
        # Create category-specific directory: eval_output/{category}/heatmaps/
        base_dir = args.save_heatmaps
        save_dir = os.path.join(base_dir, category, "heatmaps")
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Heatmaps will be saved to: {save_dir}")
    else:
        save_dir = None

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
    # train.py splits the test set 50/50 with seed 42 and uses the first half for
    # validation, i.e. for early stopping and "best checkpoint" decisions. Scoring
    # that same half reports a number the checkpoint was selected on. --split
    # heldout reproduces the complement, which is the honest figure to publish.
    subset_indices = None
    if args.split != "all":
        probe = MVTecDataset(
            root=cfg["dataset"]["root"],
            category=cfg["dataset"]["category"],
            split="test",
            image_size=cfg["vit"]["image_size"],
            patch_size=cfg["vit"]["patch_size"],
            synthetic_method=None,
        )
        n_test = len(probe)
        val_size = n_test // 2
        seed = int(cfg.get("validation", {}).get("seed", 42))
        perm = np.random.RandomState(seed).permutation(n_test).tolist()
        subset_indices = perm[:val_size] if args.split == "val" else perm[val_size:]
        logger.info(
            f"split={args.split}: {len(subset_indices)} of {n_test} test images "
            f"(seed {seed}, matching train.py)"
        )

    dataset = MVTecDataset(
        root=cfg["dataset"]["root"],
        category=cfg["dataset"]["category"],
        split="test",
        image_size=cfg["vit"]["image_size"],
        patch_size=cfg["vit"]["patch_size"],
        synthetic_method=None,
        subset_indices=subset_indices,
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
    model, ckpt_meta = load_spade(cfg, args.checkpoint, device=device, logger=logger)
    logger.info(f"model: {describe(model)}")
    if "config" in ckpt_meta:
        logger.info(f"checkpoint config: {ckpt_meta['config']}")

    # ── P1: swap in position-conditioned Mahalanobis ──
    if args.positional:
        from models.positional_mahalanobis import build_positional_scorer

        pstats_path = args.positional_stats or checkpoint_path(cfg, category, "positional_stats.pt")
        if not os.path.exists(pstats_path):
            raise FileNotFoundError(
                f"positional statistics not found: {pstats_path}\n"
                f"Run: python scripts/fit_positional_statistics.py --category {category}"
            )
        pstats = torch.load(pstats_path, map_location="cpu", weights_only=False)
        model.mahalanobis_scorer = build_positional_scorer(
            pstats,
            blend=args.position_blend,
            gamma=cfg["scoring"]["mahalanobis_gamma"],
            device=device,
        )
        pmeta = pstats.get("meta", {})
        logger.info(
            f"POSITIONAL Mahalanobis: blend={args.position_blend}, "
            f"{pstats['n_positions']} positions, fitted on {pstats['n_images']} images "
            f"({pstats['samples_per_dim']:.1f} samples/dim, shrinkage={pmeta.get('shrinkage')})"
        )
        if pmeta.get("category") not in (None, category):
            logger.warning(
                f"statistics were fitted for '{pmeta.get('category')}' but you are "
                f"evaluating '{category}'"
            )

    # ── Smoothing (applies to BOTH arms of any comparison) ──
    smooth_sigma = (
        args.smooth_sigma
        if args.smooth_sigma is not None
        else float(cfg.get("scoring", {}).get("smooth_sigma", 0.0))
    )
    logger.info(f"anomaly-map smoothing sigma = {smooth_sigma}")

    # ── Dense tiling (Phase 1 ceiling test) ──
    tiling = None
    if args.dense_tiling:
        from data.transforms import get_eval_transforms

        stats_path = args.fine_stats or checkpoint_path(
            cfg, category, f"fine_stats_g{args.grid}_o{args.overlap}.pt"
        )
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"fine-scale statistics not found: {stats_path}\n"
                f"Run: python scripts/fit_fine_statistics.py --category {category} "
                f"--grid {args.grid} --overlap {args.overlap}\n"
                "Whole-image statistics do NOT transfer to zoomed crops — scoring "
                "tiles without them produces meaningless distances."
            )
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        meta = stats.get("meta", {})
        if meta.get("grid") != args.grid or meta.get("overlap") != args.overlap:
            logger.warning(
                f"fine statistics were fitted at grid={meta.get('grid')} "
                f"overlap={meta.get('overlap')} but you are evaluating at "
                f"grid={args.grid} overlap={args.overlap} — scale mismatch"
            )
        tiling = {
            "grid": args.grid,
            "overlap": args.overlap,
            "stats": stats,
            "transform": get_eval_transforms(cfg["vit"]["image_size"]),
        }
        logger.info(
            f"DENSE TILING: {args.grid}x{args.grid} tiles, overlap {args.overlap}, "
            f"stats from {stats_path} ({stats['spatial']['n_patches']} patches)"
        )

    results = evaluate(
        model, loader, device,
        image_size=cfg["vit"]["image_size"],
        patch_size=cfg["vit"]["patch_size"],
        save_dir=save_dir,
        save_visualizations=args.save_visualizations,
        smooth_sigma=smooth_sigma,
        tiling=tiling,
    )
    logger.info(f"Image AUROC: {results['image_auroc']:.4f}")
    logger.info(f"Pixel AUROC: {results['pixel_auroc']:.4f}")
    logger.info(f"PRO       : {results.get('pro', float('nan')):.4f}")
    logger.info(
        f"[config] split={args.split} smooth_sigma={smooth_sigma} "
        f"dense_tiling={bool(tiling)}"
        + (f" grid={args.grid} overlap={args.overlap}" if tiling else "")
        + (f" positional=True blend={args.position_blend}" if args.positional else " positional=False")
    )

    # ── Log to wandb ──
    if args.log_wandb:
        wandb.log({
            "eval/image_auroc": results["image_auroc"],
            "eval/pixel_auroc": results["pixel_auroc"],
            "eval/pro": results.get("pro", float("nan")),
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
