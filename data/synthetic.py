"""Synthetic anomaly generation for self-supervised training.

Generates fake defects on normal images and produces patch-level masks.
Two methods: CutPaste and Perlin noise.
"""

import math
import random

import cv2
import numpy as np
import torch


# ──────────────────────────────────────────────
# CutPaste
# ──────────────────────────────────────────────

def cutpaste(
    image: np.ndarray,
    area_ratio: tuple[float, float] = (0.02, 0.15),
    aspect_ratio: tuple[float, float] = (0.3, 3.3),
    rng: random.Random | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply CutPaste augmentation.

    Cuts a random patch from the image and pastes it at another location.

    Args:
        image: (H, W, 3) uint8 image.
        area_ratio: min/max fraction of image area for the cut patch.
        aspect_ratio: min/max aspect ratio of the cut patch.

    Returns:
        augmented_image: (H, W, 3) uint8.
        mask: (H, W) binary mask where 1 = anomalous pixel.
    """
    rng = rng or random
    h, w = image.shape[:2]
    area = h * w

    # Random patch dimensions
    for _ in range(100):
        target_area = rng.uniform(*area_ratio) * area
        ar = math.exp(rng.uniform(math.log(aspect_ratio[0]), math.log(aspect_ratio[1])))
        pw = int(round(math.sqrt(target_area * ar)))
        ph = int(round(math.sqrt(target_area / ar)))
        if pw < w and ph < h:
            break
    else:
        pw, ph = w // 4, h // 4

    # Source crop
    sx = rng.randint(0, w - pw)
    sy = rng.randint(0, h - ph)
    patch = image[sy : sy + ph, sx : sx + pw].copy()

    # Random rotation (0, 90, 180, 270)
    k = rng.randint(0, 3)
    patch = np.rot90(patch, k)
    ph_r, pw_r = patch.shape[:2]

    # Clamp to image bounds
    pw_r = min(pw_r, w)
    ph_r = min(ph_r, h)
    patch = patch[:ph_r, :pw_r]

    # Destination
    dx = rng.randint(0, w - pw_r)
    dy = rng.randint(0, h - ph_r)

    augmented = image.copy()
    augmented[dy : dy + ph_r, dx : dx + pw_r] = patch

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[dy : dy + ph_r, dx : dx + pw_r] = 1

    return augmented, mask


# ──────────────────────────────────────────────
# Perlin Noise
# ──────────────────────────────────────────────

def _generate_perlin_noise_2d(
    shape: tuple[int, int],
    res: tuple[int, int],
    np_rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a 2-D Perlin noise array."""
    np_rng = np_rng or np.random.default_rng()
    def _fade(t: np.ndarray) -> np.ndarray:
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1
    angles = 2 * np.pi * np_rng.random((res[0] + 1, res[1] + 1))
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    # Tile corners
    g00 = gradients[:-1, :-1].repeat(d[0], axis=0).repeat(d[1], axis=1)
    g10 = gradients[1:, :-1].repeat(d[0], axis=0).repeat(d[1], axis=1)
    g01 = gradients[:-1, 1:].repeat(d[0], axis=0).repeat(d[1], axis=1)
    g11 = gradients[1:, 1:].repeat(d[0], axis=0).repeat(d[1], axis=1)

    # Dot products
    n00 = np.sum(np.stack([grid[:, :, 0], grid[:, :, 1]], axis=-1) * g00, axis=-1)
    n10 = np.sum(np.stack([grid[:, :, 0] - 1, grid[:, :, 1]], axis=-1) * g10, axis=-1)
    n01 = np.sum(np.stack([grid[:, :, 0], grid[:, :, 1] - 1], axis=-1) * g01, axis=-1)
    n11 = np.sum(np.stack([grid[:, :, 0] - 1, grid[:, :, 1] - 1], axis=-1) * g11, axis=-1)

    t = _fade(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11

    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def perlin_anomaly(
    image: np.ndarray,
    min_scale: int = 0,
    max_scale: int = 6,
    threshold: float = 0.5,
    rng: random.Random | None = None,
    np_rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a Perlin-noise-based synthetic anomaly.

    Creates a smooth random mask and applies colour perturbation.

    Args:
        image: (H, W, 3) uint8 image.
        min_scale: minimum octave for Perlin resolution.
        max_scale: maximum octave for Perlin resolution.
        threshold: binarisation threshold for the noise mask.

    Returns:
        augmented_image: (H, W, 3) uint8.
        mask: (H, W) binary mask.
    """
    h, w = image.shape[:2]
    rng = rng or random
    np_rng = np_rng or np.random.default_rng()
    scale = 2 ** rng.randint(min_scale, max_scale)
    # Resolution must divide the image dimensions
    res_h = max(1, h // scale)
    res_w = max(1, w // scale)
    # Ensure shape is divisible by res
    noise_h = res_h * (h // res_h)
    noise_w = res_w * (w // res_w)

    noise = _generate_perlin_noise_2d((noise_h, noise_w), (res_h, res_w), np_rng=np_rng)
    noise = cv2.resize(noise, (w, h))

    # Normalize → [0, 1]
    nmin, nmax = noise.min(), noise.max()
    if nmax - nmin > 1e-8:
        noise = (noise - nmin) / (nmax - nmin)

    mask = (noise > threshold).astype(np.uint8)

    # Generate random colour perturbation
    perturbation = np_rng.integers(0, 255, size=image.shape, dtype=np.uint8)
    augmented = image.copy()
    augmented[mask == 1] = cv2.addWeighted(
        image[mask == 1], 0.5, perturbation[mask == 1], 0.5, 0
    )

    return augmented, mask


# ──────────────────────────────────────────────
# Patch-level mask
# ──────────────────────────────────────────────

def mask_to_patch_labels(
    mask: np.ndarray,
    patch_size: int = 14,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Convert a pixel mask to patch-level binary labels.

    A patch is labelled anomalous if more than `threshold` fraction of its
    pixels are anomalous.

    Args:
        mask: (H, W) binary mask.
        patch_size: ViT patch size.
        threshold: fraction of anomalous pixels to mark a patch as positive.

    Returns:
        (N_patches,) float tensor of patch labels in {0, 1}.
    """
    h, w = mask.shape
    gh, gw = h // patch_size, w // patch_size
    labels = []
    for i in range(gh):
        for j in range(gw):
            patch = mask[
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
            ]
            ratio = patch.mean()
            labels.append(1.0 if ratio > threshold else 0.0)
    return torch.tensor(labels, dtype=torch.float32)  # (N_patches,)

