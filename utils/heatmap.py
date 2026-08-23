"""Heatmap visualization for patch-level anomaly scores."""

import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def smooth_map(hmap: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth an anomaly map.

    Every comparable method smooths its anomaly map before scoring (PaDiM,
    PatchCore, and the WACV SingleNet reference all use a Gaussian with sigma=4
    at 224-256 px). This pipeline did not, which both costs pixel AUROC and —
    more importantly — confounds any experiment that changes map resolution:
    a resolution gain and a smoothing gain are indistinguishable unless both
    arms are smoothed identically.

    Smoothing is a fixed, image-independent linear filter, so unlike per-image
    percentile normalization it does NOT leak calibration across images and is
    safe to use on the metric path (see EVALUATION_FIX.md).

    Args:
        hmap: (H, W) score map.
        sigma: Gaussian sigma in pixels. <= 0 disables smoothing.

    Returns:
        (H, W) smoothed map, same dtype/scale as the input.
    """
    if sigma is None or sigma <= 0:
        return hmap
    # ksize=0 lets OpenCV derive the kernel size from sigma
    return cv2.GaussianBlur(hmap, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))


def patches_to_heatmap(
    patch_scores: torch.Tensor,
    image_size: int = 224,
    patch_size: int = 14,
    percentile_clip: tuple[float, float] = (5.0, 95.0),
    normalize: bool = True,
    smooth_sigma: float = 0.0,
) -> np.ndarray:
    """Reshape flat patch scores into a 2-D heatmap and upsample.

    Args:
        patch_scores: (N_patches,) anomaly scores.
        image_size: original image spatial size.
        patch_size: ViT patch size.
        percentile_clip: (low, high) percentiles for robust normalization.
        normalize: If True, normalize to [0, 1] using percentiles. If False, return raw scores.
                  Set to False for metric computation (pixel AUROC), True for visualization.
        smooth_sigma: Gaussian sigma applied AFTER upsampling and BEFORE any
                  normalization, so it is part of the metric path. 0 disables.

    Returns:
        (image_size, image_size) numpy heatmap.
        - If normalize=True: values in [0, 1]
        - If normalize=False: raw anomaly scores (may be negative or > 1)
    """
    grid = int(math.sqrt(patch_scores.numel()))
    hmap = patch_scores.detach().cpu().float().reshape(grid, grid).numpy()
    # Bilinear upsample to image resolution
    hmap = cv2.resize(hmap, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    # Smooth on the raw scale — before normalization, so metrics see it too.
    hmap = smooth_map(hmap, smooth_sigma)
    
    if normalize:
        # Robust normalization using percentiles to handle outliers
        # ONLY for visualization - NOT for metric computation
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

