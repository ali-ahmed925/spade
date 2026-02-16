"""Heatmap visualization for patch-level anomaly scores."""

import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def patches_to_heatmap(
    patch_scores: torch.Tensor,
    image_size: int = 224,
    patch_size: int = 14,
    percentile_clip: tuple[float, float] = (5.0, 95.0),
) -> np.ndarray:
    """Reshape flat patch scores into a 2-D heatmap and upsample.

    Args:
        patch_scores: (N_patches,) anomaly scores.
        image_size: original image spatial size.
        patch_size: ViT patch size.
        percentile_clip: (low, high) percentiles for robust normalization.

    Returns:
        (image_size, image_size) numpy heatmap in [0, 1].
    """
    grid = int(math.sqrt(patch_scores.numel()))
    hmap = patch_scores.detach().cpu().float().reshape(grid, grid).numpy()
    # Bilinear upsample to image resolution
    hmap = cv2.resize(hmap, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    
    # Robust normalization using percentiles to handle outliers
    hmin = np.percentile(hmap, percentile_clip[0])
    hmax = np.percentile(hmap, percentile_clip[1])
    
    # Clip extreme values
    hmap = np.clip(hmap, hmin, hmax)
    
    # Normalize to [0, 1]
    if hmax - hmin > 1e-8:
        hmap = (hmap - hmin) / (hmax - hmin)
    else:
        hmap = np.zeros_like(hmap)
    
    return hmap


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap_name: str = "hot",
) -> np.ndarray:
    """Overlay a heatmap on an RGB image.

    Args:
        image: (H, W, 3) uint8 RGB image.
        heatmap: (H, W) float heatmap in [0, 1].
        alpha: blending factor.
        colormap_name: matplotlib colormap name ('hot', 'jet', 'viridis', etc.).

    Returns:
        (H, W, 3) uint8 blended image.
    """
    # Use 'hot' colormap (black -> red -> yellow) for better visualization
    colormap = plt.cm.get_cmap(colormap_name)(heatmap)[..., :3]  # (H, W, 3) float [0,1]
    colormap = (colormap * 255).astype(np.uint8)
    blended = cv2.addWeighted(image, 1 - alpha, colormap, alpha, 0)
    return blended


def save_heatmap(heatmap: np.ndarray, path: str, colormap: str = "hot", vmin: float = 0.0, vmax: float = 1.0) -> None:
    """Save a heatmap array as a colour-mapped PNG.
    
    Args:
        heatmap: (H, W) float heatmap in [0, 1].
        path: Output file path.
        colormap: Matplotlib colormap name.
        vmin: Minimum value for colormap scaling.
        vmax: Maximum value for colormap scaling.
    """
    plt.imsave(path, heatmap, cmap=colormap, vmin=vmin, vmax=vmax)

