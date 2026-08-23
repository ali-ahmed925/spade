"""Dense multi-scale tiling — the Phase 1 ceiling test.

This module answers one question and nothing else:

    Does high-resolution evidence improve localization for this backbone at all?

MVTec images are 1024x1024 natively and are resized to 224 before encoding
(data/mvtec_dataset.py), so ~21x the pixel area is discarded and every patch
score covers a 64x64 native region. Dense tiling re-encodes overlapping crops of
the ORIGINAL image at the same 224 input size, giving each patch a 4x smaller
native footprint.

Whatever dense tiling achieves is the UPPER BOUND for any adaptive scheme that
refines only a subset of regions — you cannot beat exhaustive refinement by
doing less of it. If this does not beat the coarse baseline, adaptive refinement
(ARR) is dead and should not be built. See the plan's G1 gate.

Deliberately parameter-free: the only fitted quantities are the fine-scale
Mahalanobis statistics, estimated in closed form by scripts/fit_fine_statistics.py.

NOTE ON SCALE: patch statistics do NOT transfer across scale. A crop resized to
224 has different feature statistics from a whole image resized to 224, so
scoring tiles with the coarse mu/Sigma produces meaningless distances. The fine
scale needs its own statistics; `require_fine_stats` enforces this rather than
letting it fail silently.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


def tile_boxes(
    native_size: int,
    grid: int = 3,
    overlap: float = 0.5,
) -> list[tuple[int, int, int, int]]:
    """Compute overlapping tile boxes covering a square image.

    Args:
        native_size: side length of the source image in pixels.
        grid: number of tiles per axis (grid x grid tiles total).
        overlap: fraction of a tile shared with its neighbour, in [0, 1).

    Returns:
        List of (x0, y0, x1, y1) boxes in native pixel coordinates, covering the
        whole image. Boxes are clamped to the image bounds, so edge tiles may be
        marginally smaller than interior ones.
    """
    if grid < 1:
        raise ValueError(f"grid must be >= 1, got {grid}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    if grid == 1:
        return [(0, 0, native_size, native_size)]

    # With `grid` tiles of side t and stride s = t * (1 - overlap):
    #   (grid - 1) * s + t = native_size
    tile = native_size / ((grid - 1) * (1.0 - overlap) + 1.0)
    stride = tile * (1.0 - overlap)

    boxes: list[tuple[int, int, int, int]] = []
    for gy in range(grid):
        for gx in range(grid):
            x0 = int(round(gx * stride))
            y0 = int(round(gy * stride))
            x1 = min(int(round(x0 + tile)), native_size)
            y1 = min(int(round(y0 + tile)), native_size)
            # keep the tile size stable at the far edge
            x0 = max(0, x1 - int(round(tile)))
            y0 = max(0, y1 - int(round(tile)))
            boxes.append((x0, y0, x1, y1))
    return boxes


def crops_from_image(
    image_rgb: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    out_size: int = 224,
) -> np.ndarray:
    """Crop and resize regions of a native-resolution image.

    Each crop is resized to `out_size`, so the encoder always sees its native
    input resolution and no position-embedding interpolation is involved.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image at native resolution.
        boxes: (x0, y0, x1, y1) boxes in native coordinates.
        out_size: encoder input size.

    Returns:
        (len(boxes), out_size, out_size, 3) uint8 array.
    """
    crops = np.empty((len(boxes), out_size, out_size, 3), dtype=np.uint8)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        patch = image_rgb[y0:y1, x0:x1]
        crops[i] = cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return crops


def stitch_tile_scores(
    tile_scores: torch.Tensor | np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    native_size: int,
    canvas_size: int = 224,
) -> np.ndarray:
    """Fuse per-tile patch scores into a single anomaly map.

    Each tile contributes a (grid x grid) score map covering its box. Scores are
    upsampled to the box's extent on the canvas and averaged where tiles overlap,
    which also suppresses tile-boundary discontinuities.

    Args:
        tile_scores: (T, N_patches) raw patch scores, one row per box.
        boxes: the boxes the scores came from, in native coordinates.
        native_size: side length of the source image.
        canvas_size: side length of the output map (224 to match the GT masks).

    Returns:
        (canvas_size, canvas_size) fused raw score map. Raw — never per-image
        normalized (EVALUATION_FIX.md).
    """
    if isinstance(tile_scores, torch.Tensor):
        tile_scores = tile_scores.detach().cpu().float().numpy()
    if len(tile_scores) != len(boxes):
        raise ValueError(f"{len(tile_scores)} score rows for {len(boxes)} boxes")

    accum = np.zeros((canvas_size, canvas_size), dtype=np.float64)
    counts = np.zeros((canvas_size, canvas_size), dtype=np.float64)
    scale = canvas_size / float(native_size)

    for scores, (x0, y0, x1, y1) in zip(tile_scores, boxes):
        side = int(round(np.sqrt(scores.size)))
        grid_map = scores.reshape(side, side).astype(np.float64)

        cx0, cy0 = int(round(x0 * scale)), int(round(y0 * scale))
        cx1, cy1 = int(round(x1 * scale)), int(round(y1 * scale))
        cx1, cy1 = min(cx1, canvas_size), min(cy1, canvas_size)
        w, h = cx1 - cx0, cy1 - cy0
        if w <= 0 or h <= 0:
            continue

        resized = cv2.resize(grid_map, (w, h), interpolation=cv2.INTER_LINEAR)
        accum[cy0:cy1, cx0:cx1] += resized
        counts[cy0:cy1, cx0:cx1] += 1.0

    # Any pixel no tile covered (possible only with pathological box sets)
    # falls back to the map mean rather than zero, which would read as "normal".
    uncovered = counts == 0
    counts[uncovered] = 1.0
    fused = accum / counts
    if uncovered.any():
        fused[uncovered] = float(fused[~uncovered].mean()) if (~uncovered).any() else 0.0
    return fused


def image_score_from_map(
    fused_map: np.ndarray,
    pool_grid: int = 32,
    top_k: int = 3,
) -> float:
    """Image-level score from a fused map, mirroring SPADE.get_image_score.

    The coarse model takes the top-3 mean over 256 patch scores. A fused map has
    finer granularity, so it is average-pooled to `pool_grid` x `pool_grid` cells
    first; without pooling, top-3 over ~50k pixels would sample a single hot spot
    and would not be comparable to the coarse image score.

    Args:
        fused_map: (H, W) raw score map.
        pool_grid: pooling grid side.
        top_k: number of cells averaged.

    Returns:
        Scalar image-level anomaly score.
    """
    pooled = cv2.resize(
        fused_map.astype(np.float32), (pool_grid, pool_grid), interpolation=cv2.INTER_AREA
    ).reshape(-1)
    k = min(top_k, pooled.size)
    return float(np.sort(pooled)[-k:].mean())


def require_fine_stats(scorer, context: str = "dense tiling") -> None:
    """Fail loudly if a scorer is being used without scale-matched statistics.

    Scoring zoomed crops with whole-image statistics silently produces garbage
    distances; the run completes and the metrics are meaningless. This turns that
    into an error.
    """
    if not bool(scorer.is_initialized):
        raise RuntimeError(
            f"{context} requires fine-scale Mahalanobis statistics, but the scorer "
            "is uninitialized. Run scripts/fit_fine_statistics.py for this "
            "category/grid/overlap first — whole-image statistics do not transfer "
            "to zoomed crops."
        )


def _validate_stats_entry(key: str, entry: dict) -> None:
    """Reject statistics that were never actually fitted.

    A freshly constructed MahalanobisScoring has mu=0 and Sigma^-1=I, which
    scores every patch by its raw squared norm — plausible-looking numbers that
    mean nothing. Because the swap forces is_initialized=True, this must be
    checked on the incoming statistics, not on the scorer afterwards.
    """
    mu = entry.get("mu")
    sigma_inv = entry.get("sigma_inv")
    if mu is None or sigma_inv is None:
        raise RuntimeError(f"fine-scale statistics for '{key}' are missing mu/sigma_inv")

    if not torch.isfinite(mu).all() or not torch.isfinite(sigma_inv).all():
        raise RuntimeError(f"fine-scale statistics for '{key}' contain non-finite values")

    if float(mu.abs().max()) == 0.0:
        raise RuntimeError(
            f"fine-scale statistics for '{key}' are unfitted (mu is all zeros). "
            "Run scripts/fit_fine_statistics.py — whole-image statistics do not "
            "transfer to zoomed crops, and unfitted ones are worse still."
        )

    identity = torch.eye(sigma_inv.shape[0], dtype=sigma_inv.dtype)
    if torch.allclose(sigma_inv.cpu(), identity, atol=1e-6):
        raise RuntimeError(
            f"fine-scale statistics for '{key}' are unfitted (Sigma^-1 is the "
            "identity). Run scripts/fit_fine_statistics.py."
        )


class fine_statistics:
    """Context manager swapping a model's Mahalanobis statistics to fine scale.

    The comparison "coarse vs dense tiling" must change exactly two things: the
    input scale, and the statistics matched to that scale. Everything else — the
    Q-Former, the attention term, the frequency stream, the score weights — has
    to stay identical, or the experiment measures a mixture of effects.

    So rather than building a second model, this temporarily swaps mu/Sigma^-1
    inside the existing one and restores them on exit.

    Usage::

        with fine_statistics(model, torch.load("fine_stats.pt")):
            out = model(crops)      # identical code path, fine-scale statistics
    """

    def __init__(self, model, stats: dict):
        self.model = model
        self.stats = stats
        self._saved: dict[str, torch.Tensor] = {}

    def _targets(self):
        pairs = [("spatial", self.model.mahalanobis_scorer)]
        if getattr(self.model, "freq_mahalanobis_scorer", None) is not None:
            pairs.append(("frequency", self.model.freq_mahalanobis_scorer))
        return pairs

    def __enter__(self):
        for key, scorer in self._targets():
            if key not in self.stats:
                if key == "frequency":
                    raise RuntimeError(
                        "the model has a frequency stream but the fine statistics "
                        "file has no 'frequency' entry — refit with the same "
                        "frequency setting, or the frequency scores will be "
                        "computed against whole-image statistics"
                    )
                raise RuntimeError(f"fine statistics missing '{key}' entry")
            saved = {
                "mu": scorer.mu.detach().clone(),
                "sigma_inv": scorer.sigma_inv.detach().clone(),
                "is_initialized": scorer.is_initialized.detach().clone(),
            }
            _validate_stats_entry(key, self.stats[key])
            self._saved[key] = saved
            device = scorer.mu.device
            scorer.mu.data = self.stats[key]["mu"].to(device)
            scorer.sigma_inv.data = self.stats[key]["sigma_inv"].to(device)
            scorer.is_initialized.data = torch.tensor(True, device=device)
        return self.model

    def __exit__(self, *exc):
        for key, scorer in self._targets():
            saved = self._saved.get(key)
            if saved is None:
                continue
            scorer.mu.data = saved["mu"]
            scorer.sigma_inv.data = saved["sigma_inv"]
            scorer.is_initialized.data = saved["is_initialized"]
        self._saved.clear()
        return False
