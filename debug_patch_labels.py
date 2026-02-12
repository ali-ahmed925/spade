"""Debug and visualize patch-level label generation on MVTec AD.

Validates patch-level label generation by visualizing:
- Original image, ground truth mask, patch grid, patch heatmap, overlay, reconstructed mask
- Per-sample diagnostics and sanity checks
"""

import argparse
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.mvtec_dataset import MVTecDataset
from data.synthetic import mask_to_patch_labels


def reconstruct_mask_from_patches(
    patch_labels: torch.Tensor,
    image_size: int,
    patch_size: int,
) -> np.ndarray:
    """Reconstruct a pixel-level mask from patch labels.

    Args:
        patch_labels: (N_patches,) binary patch labels.
        image_size: Original image size.
        patch_size: ViT patch size.

    Returns:
        (image_size, image_size) binary mask.
    """
    grid = int(np.sqrt(patch_labels.numel()))
    mask = np.zeros((image_size, image_size), dtype=np.uint8)

    patch_labels_np = patch_labels.numpy().reshape(grid, grid)
    for i in range(grid):
        for j in range(grid):
            y0, y1 = i * patch_size, (i + 1) * patch_size
            x0, x1 = j * patch_size, (j + 1) * patch_size
            mask[y0:y1, x0:x1] = int(patch_labels_np[i, j] * 255)

    return mask


def draw_patch_grid(image: np.ndarray, patch_size: int, color: tuple = (0, 255, 0)) -> np.ndarray:
    """Draw patch grid overlay on image.

    Args:
        image: (H, W, 3) RGB image.
        patch_size: Patch size.
        color: Grid line color (BGR for OpenCV).

    Returns:
        Image with grid overlay.
    """
    h, w = image.shape[:2]
    grid_h, grid_w = h // patch_size, w // patch_size

    overlay = image.copy()
    # Vertical lines
    for j in range(1, grid_w):
        x = j * patch_size
        cv2.line(overlay, (x, 0), (x, h), color, 1)
    # Horizontal lines
    for i in range(1, grid_h):
        y = i * patch_size
        cv2.line(overlay, (0, y), (w, y), color, 1)

    return overlay


def overlay_patch_labels(
    image: np.ndarray,
    patch_labels: torch.Tensor,
    patch_size: int,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay patch labels on image (red = anomaly patch).

    Args:
        image: (H, W, 3) RGB image.
        patch_labels: (N_patches,) binary patch labels.
        patch_size: Patch size.
        alpha: Blending factor.

    Returns:
        Blended image.
    """
    h, w = image.shape[:2]
    grid = int(np.sqrt(patch_labels.numel()))
    patch_labels_np = patch_labels.numpy().reshape(grid, grid)

    overlay = image.copy()
    for i in range(grid):
        for j in range(grid):
            if patch_labels_np[i, j] > 0.5:
                y0, y1 = i * patch_size, (i + 1) * patch_size
                x0, x1 = j * patch_size, (j + 1) * patch_size
                overlay[y0:y1, x0:x1] = cv2.addWeighted(
                    image[y0:y1, x0:x1], 1 - alpha,
                    np.full((patch_size, patch_size, 3), [255, 0, 0], dtype=np.uint8), alpha, 0
                )

    return overlay


def visualize_sample(
    image: np.ndarray,
    mask: np.ndarray,
    patch_labels: torch.Tensor,
    patch_size: int,
    output_path: str,
    diagnostics: dict,
) -> None:
    """Create side-by-side visualization for a single sample.

    Args:
        image: (H, W, 3) RGB image.
        mask: (H, W) binary ground truth mask.
        patch_labels: (N_patches,) patch labels.
        patch_size: Patch size.
        output_path: Path to save visualization.
        diagnostics: Dict with diagnostic info to print.
    """
    grid = int(np.sqrt(patch_labels.numel()))
    patch_heatmap = patch_labels.numpy().reshape(grid, grid)
    reconstructed_mask = reconstruct_mask_from_patches(
        patch_labels, image.shape[0], patch_size
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"Sample: {Path(diagnostics['path']).name}\n"
        f"Anomaly pixels: {diagnostics['anomaly_pixels']} | "
        f"Anomaly patches: {diagnostics['anomaly_patches']} | "
        f"Ratio: {diagnostics['anomaly_patch_ratio']:.3f}",
        fontsize=12,
    )

    # Row 1
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(mask, cmap="gray")
    axes[0, 1].set_title("Ground Truth Mask (Pixel-level)")
    axes[0, 1].axis("off")

    grid_overlay = draw_patch_grid(image.copy(), patch_size)
    axes[0, 2].imshow(grid_overlay)
    axes[0, 2].set_title(f"Patch Grid ({grid}×{grid})")
    axes[0, 2].axis("off")

    # Row 2
    im = axes[1, 0].imshow(patch_heatmap, cmap="hot", vmin=0, vmax=1)
    axes[1, 0].set_title("Patch Label Heatmap")
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0])

    patch_overlay = overlay_patch_labels(image.copy(), patch_labels, patch_size)
    axes[1, 1].imshow(patch_overlay)
    axes[1, 1].set_title("Patch Labels Overlay (Red = Anomaly)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(reconstructed_mask, cmap="gray")
    axes[1, 2].set_title("Reconstructed Mask (from Patches)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug patch-level label generation")
    parser.add_argument(
        "--root",
        type=str,
        default="./datasets/mvtec_anomaly_detection",
        help="MVTec dataset root directory",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="bottle",
        help="MVTec category (default: bottle)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Image size (default: 224)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=14,
        help="Patch size (default: 14)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./debug_output",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="Maximum number of samples to visualize (default: 10)",
    )
    args = parser.parse_args()

    # ── Setup ──
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading MVTec AD test split: {args.category}")
    print(f"Output directory: {args.output_dir}")

    # ── Load dataset ──
    dataset = MVTecDataset(
        root=args.root,
        category=args.category,
        split="test",
        image_size=args.image_size,
        patch_size=args.patch_size,
        synthetic_method=None,
    )

    # Filter to anomalous samples only
    anomalous_indices = [i for i in range(len(dataset)) if dataset.labels[i] == 1]
    print(f"Found {len(anomalous_indices)} anomalous samples")

    if len(anomalous_indices) == 0:
        print("ERROR: No anomalous samples found in test split!")
        return

    # ── Process samples ──
    num_processed = 0
    num_warnings = 0

    for idx in anomalous_indices[: args.max_samples]:
        sample = dataset[idx]
        image_path = sample["path"]
        patch_labels = sample["patch_labels"]
        mask = sample["mask"]

        # Load original image (not transformed)
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_np = cv2.resize(image_np, (args.image_size, args.image_size))

        # ── Diagnostics ──
        anomaly_pixels = int(mask.sum())
        anomaly_patches = int(patch_labels.sum())
        total_patches = patch_labels.numel()
        anomaly_patch_ratio = float(anomaly_patches / max(total_patches, 1))

        diagnostics = {
            "path": image_path,
            "anomaly_pixels": anomaly_pixels,
            "anomaly_patches": anomaly_patches,
            "total_patches": total_patches,
            "anomaly_patch_ratio": anomaly_patch_ratio,
        }

        # ── Sanity checks ──
        warnings = []
        if anomaly_pixels > 0 and anomaly_patches == 0:
            warnings.append(
                f"⚠️  SANITY CHECK FAILED: Mask has {anomaly_pixels} anomaly pixels "
                f"but 0 anomaly patches!"
            )
            num_warnings += 1
        if anomaly_pixels == 0 and anomaly_patches > 0:
            warnings.append(
                f"⚠️  SANITY CHECK FAILED: Mask is empty but {anomaly_patches} patches "
                f"are labeled as anomalous!"
            )
            num_warnings += 1

        # ── Print diagnostics ──
        print(f"\n{'='*60}")
        print(f"Sample {num_processed + 1}: {Path(image_path).name}")
        print(f"  Anomaly pixels:      {anomaly_pixels:,}")
        print(f"  Anomaly patches:     {anomaly_patches:,} / {total_patches:,}")
        print(f"  Anomaly patch ratio:  {anomaly_patch_ratio:.3f}")
        if warnings:
            for w in warnings:
                print(f"  {w}")

        # ── Visualize ──
        output_path = os.path.join(
            args.output_dir, f"sample_{num_processed:03d}_{Path(image_path).stem}.png"
        )
        visualize_sample(image_np, mask, patch_labels, args.patch_size, output_path, diagnostics)
        print(f"  Saved visualization: {output_path}")

        num_processed += 1

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Processed samples: {num_processed}")
    print(f"  Warnings:          {num_warnings}")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

