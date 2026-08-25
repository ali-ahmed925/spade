"""Geometry and fusion tests for dense tiling (Phase 1 ceiling test).

Pure numpy/cv2 — no BLIP-2, no GPU.
"""

import numpy as np
import pytest
import torch

from models.tiling import (
    crops_from_image,
    image_score_from_map,
    require_fine_stats,
    stitch_tile_scores,
    tile_boxes,
)


# ── Box geometry ─────────────────────────────────────────────────────────────
def test_tile_boxes_cover_the_whole_image():
    boxes = tile_boxes(1024, grid=3, overlap=0.5)
    covered = np.zeros((1024, 1024), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        covered[y0:y1, x0:x1] = True
    assert covered.all(), "tiling must not leave uncovered pixels"


def test_tile_boxes_count_and_bounds():
    boxes = tile_boxes(1024, grid=3, overlap=0.5)
    assert len(boxes) == 9
    for x0, y0, x1, y1 in boxes:
        assert 0 <= x0 < x1 <= 1024
        assert 0 <= y0 < y1 <= 1024


def test_tile_boxes_overlap_is_honoured():
    """With 50% overlap, adjacent tiles share half their extent."""
    boxes = tile_boxes(1024, grid=3, overlap=0.5)
    first, second = boxes[0], boxes[1]  # same row, adjacent columns
    tile_w = first[2] - first[0]
    stride = second[0] - first[0]
    assert stride == pytest.approx(tile_w * 0.5, rel=0.05)


def test_single_tile_is_the_whole_image():
    assert tile_boxes(1024, grid=1) == [(0, 0, 1024, 1024)]


def test_tile_boxes_rejects_bad_parameters():
    with pytest.raises(ValueError):
        tile_boxes(1024, grid=0)
    with pytest.raises(ValueError):
        tile_boxes(1024, grid=3, overlap=1.0)


# ── Cropping ─────────────────────────────────────────────────────────────────
def test_crops_have_encoder_input_size():
    image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    boxes = tile_boxes(1024, grid=3, overlap=0.5)
    crops = crops_from_image(image, boxes, out_size=224)
    assert crops.shape == (9, 224, 224, 3)
    assert crops.dtype == np.uint8


def test_crop_content_matches_its_box():
    """A crop must contain the pixels of its own box, not another's."""
    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    image[0:512, 0:512] = 255  # top-left quadrant white
    boxes = [(0, 0, 512, 512), (512, 512, 1024, 1024)]
    crops = crops_from_image(image, boxes)
    assert crops[0].mean() > 250
    assert crops[1].mean() < 5


# ── Stitching ────────────────────────────────────────────────────────────────
def test_stitch_constant_scores_reproduces_the_constant():
    boxes = tile_boxes(1024, grid=3, overlap=0.5)
    scores = torch.full((len(boxes), 256), 7.0)
    fused = stitch_tile_scores(scores, boxes, native_size=1024, canvas_size=224)
    assert fused.shape == (224, 224)
    assert np.allclose(fused, 7.0, atol=1e-6), "averaging overlaps must be unbiased"


def test_stitch_places_a_hot_tile_in_the_right_corner():
    boxes = [(0, 0, 512, 512), (512, 0, 1024, 512), (0, 512, 512, 1024), (512, 512, 1024, 1024)]
    scores = torch.zeros(4, 256)
    scores[3] = 10.0  # bottom-right tile
    fused = stitch_tile_scores(scores, boxes, native_size=1024, canvas_size=224)
    h = 224 // 2
    assert fused[h:, h:].mean() > 9.0
    assert fused[:h, :h].mean() < 1.0


def test_stitch_rejects_mismatched_inputs():
    boxes = tile_boxes(1024, grid=3, overlap=0.5)
    with pytest.raises(ValueError):
        stitch_tile_scores(torch.zeros(3, 256), boxes, native_size=1024)


def test_stitch_is_higher_resolution_than_the_coarse_grid():
    """The point of tiling: finer spatial detail than one 16x16 map.

    A single hot patch in one tile must occupy a smaller fraction of the canvas
    than a single hot patch of the coarse whole-image grid would.
    """
    boxes = tile_boxes(1024, grid=3, overlap=0.5)
    scores = torch.zeros(len(boxes), 256)
    scores[4, 128] = 100.0  # one patch of the centre tile
    fused = stitch_tile_scores(scores, boxes, native_size=1024, canvas_size=224)
    hot_fraction = float((fused > fused.max() * 0.5).mean())
    coarse_patch_fraction = 1.0 / 256
    assert hot_fraction < coarse_patch_fraction, (
        f"hot region covers {hot_fraction:.4f} of the canvas; a coarse patch "
        f"covers {coarse_patch_fraction:.4f} — tiling gained no resolution"
    )


# ── Image score ──────────────────────────────────────────────────────────────
def test_image_score_responds_to_a_localized_defect():
    clean = np.zeros((224, 224), dtype=np.float32)
    defective = clean.copy()
    defective[100:110, 100:110] = 50.0
    assert image_score_from_map(defective) > image_score_from_map(clean)


def test_image_score_ignores_a_single_stray_pixel():
    """Pooling must stop one hot pixel from dominating the image score."""
    stray = np.zeros((224, 224), dtype=np.float32)
    stray[50, 50] = 1000.0
    blob = np.zeros((224, 224), dtype=np.float32)
    blob[40:80, 40:80] = 20.0
    assert image_score_from_map(blob) > image_score_from_map(stray)


# ── Scale-statistics guard ───────────────────────────────────────────────────
def test_require_fine_stats_raises_on_uninitialized_scorer():
    from models.mahalanobis_scoring import MahalanobisScoring

    scorer = MahalanobisScoring(feature_dim=8)
    with pytest.raises(RuntimeError, match="fine-scale"):
        require_fine_stats(scorer)

    scorer.update_statistics(torch.randn(64, 8))
    require_fine_stats(scorer)  # must not raise once fitted


# ── Metric regression ────────────────────────────────────────────────────────
def test_compute_pro_works_without_numpy_trapz(monkeypatch):
    """numpy 2 removed np.trapz; PRO must not reference it eagerly.

    The first version used getattr(np, "trapezoid", np.trapz), whose default
    argument is evaluated immediately and raised AttributeError on numpy 2.
    """
    import numpy as np

    from utils.metrics import compute_pro

    monkeypatch.delattr(np, "trapz", raising=False)

    masks = np.zeros((2, 32, 32), dtype=np.uint8)
    masks[0, 8:16, 8:16] = 1
    scores = masks.astype(float) + np.random.default_rng(0).normal(0, 0.01, masks.shape)
    assert compute_pro(masks, scores) == pytest.approx(1.0, abs=0.05)
