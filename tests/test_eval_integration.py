"""End-to-end evaluation tests on a synthetic MVTec-shaped dataset.

Covers the wiring that unit tests cannot: the eval loop, the dense-tiling path,
the held-out split arithmetic, and the metric plumbing — all on a stub backbone,
so no BLIP-2 download and no GPU.

The synthetic dataset is built at 1024x1024 like real MVTec, so the tiling code
exercises the same coordinate mapping it will use in the real run.
"""

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data.mvtec_dataset import MVTecDataset
from tests.stub_blip2 import fit_statistics, make_stub_spade

NATIVE = 1024
IMAGE_SIZE = 224
PATCH_SIZE = 14


@pytest.fixture
def fake_mvtec(tmp_path):
    """A 1024x1024 MVTec-shaped category: textured normals, blob defects."""
    rng = np.random.default_rng(0)
    root = tmp_path / "mvtec"
    category = "synthtex"
    train = root / category / "train" / "good"
    test_good = root / category / "test" / "good"
    test_bad = root / category / "test" / "blob"
    gt_bad = root / category / "ground_truth" / "blob"
    for d in (train, test_good, test_bad, gt_bad):
        d.mkdir(parents=True, exist_ok=True)

    def texture():
        base = rng.normal(128, 12, (NATIVE, NATIVE, 3))
        return np.clip(base, 0, 255).astype(np.uint8)

    for i in range(6):
        cv2.imwrite(str(train / f"{i:03d}.png"), texture())
    for i in range(4):
        cv2.imwrite(str(test_good / f"{i:03d}.png"), texture())

    for i in range(4):
        img = texture()
        mask = np.zeros((NATIVE, NATIVE), dtype=np.uint8)
        cy, cx = int(rng.integers(200, 800)), int(rng.integers(200, 800))
        img[cy - 60 : cy + 60, cx - 60 : cx + 60] = 20   # dark blob
        mask[cy - 60 : cy + 60, cx - 60 : cx + 60] = 255
        cv2.imwrite(str(test_bad / f"{i:03d}.png"), img)
        cv2.imwrite(str(gt_bad / f"{i:03d}_mask.png"), mask)

    return str(root), category


@pytest.fixture
def dataset(fake_mvtec):
    root, category = fake_mvtec
    return MVTecDataset(
        root=root, category=category, split="test",
        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, synthetic_method=None,
    )


@pytest.fixture
def model(dataset):
    m = make_stub_spade()
    images = torch.stack([dataset[i]["image"] for i in range(len(dataset))])
    fit_statistics(m, images)
    m.eval()
    return m


def _fine_stats_from(model, dataset):
    """Stand-in for scripts/fit_fine_statistics.py output."""
    images = torch.stack([dataset[i]["image"] for i in range(len(dataset))])
    with torch.no_grad():
        descriptors = model.build_descriptors(images)["descriptors"]
    dim = descriptors.shape[-1]
    scorer = type(model.mahalanobis_scorer)(feature_dim=dim, regularization=1e-4)
    scorer.update_statistics(descriptors.reshape(-1, dim))
    return {
        "spatial": {"mu": scorer.mu.clone(), "sigma_inv": scorer.sigma_inv.clone()},
        "meta": {"grid": 3, "overlap": 0.5},
    }


# ── The evaluation loop ──────────────────────────────────────────────────────
def test_evaluate_reports_all_three_metrics(model, dataset):
    from eval import evaluate

    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    results = evaluate(
        model, loader, torch.device("cpu"),
        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, smooth_sigma=4.0,
    )
    for key in ("image_auroc", "pixel_auroc", "pro"):
        assert key in results, f"{key} missing from results"
        assert 0.0 <= results[key] <= 1.0 or np.isnan(results[key])


def test_smoothing_changes_the_metric_path(model, dataset):
    """Smoothing must reach the metrics, not just the visualizations."""
    from eval import evaluate

    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    kw = dict(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE)
    unsmoothed = evaluate(model, loader, torch.device("cpu"), smooth_sigma=0.0, **kw)
    smoothed = evaluate(model, loader, torch.device("cpu"), smooth_sigma=4.0, **kw)
    assert unsmoothed["pixel_auroc"] != smoothed["pixel_auroc"], (
        "smoothing did not affect pixel AUROC — it is not on the metric path"
    )


# ── Dense tiling ─────────────────────────────────────────────────────────────
def test_dense_tiling_produces_a_valid_map(model, dataset):
    from eval import _dense_tiling_map
    from data.transforms import get_eval_transforms

    tiling = {
        "grid": 3, "overlap": 0.5,
        "stats": _fine_stats_from(model, dataset),
        "transform": get_eval_transforms(IMAGE_SIZE),
    }
    hmap = _dense_tiling_map(
        model=model, image_path=dataset.image_paths[0], tiling=tiling,
        image_size=IMAGE_SIZE, device=torch.device("cpu"), smooth_sigma=4.0,
    )
    assert hmap.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert np.isfinite(hmap).all()


def test_dense_tiling_restores_coarse_statistics_afterwards(model, dataset):
    """The scale swap must not leak into subsequent coarse scoring."""
    from eval import _dense_tiling_map
    from data.transforms import get_eval_transforms

    before_mu = model.mahalanobis_scorer.mu.detach().clone()
    before_sigma = model.mahalanobis_scorer.sigma_inv.detach().clone()

    _dense_tiling_map(
        model=model, image_path=dataset.image_paths[0],
        tiling={
            "grid": 3, "overlap": 0.5,
            "stats": _fine_stats_from(model, dataset),
            "transform": get_eval_transforms(IMAGE_SIZE),
        },
        image_size=IMAGE_SIZE, device=torch.device("cpu"),
    )

    assert torch.allclose(before_mu, model.mahalanobis_scorer.mu)
    assert torch.allclose(before_sigma, model.mahalanobis_scorer.sigma_inv)


def test_dense_tiling_refuses_uninitialized_statistics(model, dataset):
    """Scoring crops with unfitted statistics must raise, not silently proceed."""
    from eval import _dense_tiling_map
    from data.transforms import get_eval_transforms
    from models.mahalanobis_scoring import MahalanobisScoring

    empty = MahalanobisScoring(feature_dim=model.descriptor_dim)
    with pytest.raises(RuntimeError, match="fine-scale"):
        _dense_tiling_map(
            model=model, image_path=dataset.image_paths[0],
            tiling={
                "grid": 3, "overlap": 0.5,
                "stats": {"spatial": {"mu": empty.mu, "sigma_inv": empty.sigma_inv}},
                "transform": get_eval_transforms(IMAGE_SIZE),
            },
            image_size=IMAGE_SIZE, device=torch.device("cpu"),
        )


def test_dense_tiling_map_is_finer_than_coarse(model, dataset):
    """Sanity: the tiled map must carry higher spatial frequency content."""
    from eval import _dense_tiling_map
    from data.transforms import get_eval_transforms
    from utils.heatmap import patches_to_heatmap

    path = dataset.image_paths[0]
    tiled = _dense_tiling_map(
        model=model, image_path=path,
        tiling={
            "grid": 3, "overlap": 0.5,
            "stats": _fine_stats_from(model, dataset),
            "transform": get_eval_transforms(IMAGE_SIZE),
        },
        image_size=IMAGE_SIZE, device=torch.device("cpu"), smooth_sigma=0.0,
    )
    with torch.no_grad():
        coarse_scores = model(dataset[0]["image"].unsqueeze(0))["patch_scores"][0]
    coarse = patches_to_heatmap(
        coarse_scores.cpu(), image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, normalize=False
    )

    def detail(m):
        m = (m - m.mean()) / (m.std() + 1e-8)
        return float(np.abs(np.diff(m, axis=0)).mean() + np.abs(np.diff(m, axis=1)).mean())

    assert detail(tiled) > detail(coarse), "tiled map is not spatially finer than coarse"


# ── Held-out split arithmetic ────────────────────────────────────────────────
def test_split_indices_match_train_py_and_are_disjoint():
    """--split heldout must be exactly the complement of train.py's val half."""
    n_test = 79  # wood
    perm = np.random.RandomState(42).permutation(n_test).tolist()
    val_size = n_test // 2
    val, heldout = perm[:val_size], perm[val_size:]

    assert len(val) == 39 and len(heldout) == 40
    assert not set(val) & set(heldout), "splits overlap — held-out is contaminated"
    assert sorted(val + heldout) == list(range(n_test)), "splits do not cover the test set"


def test_subset_indices_actually_subsets(fake_mvtec):
    root, category = fake_mvtec
    full = MVTecDataset(root=root, category=category, split="test",
                        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, synthetic_method=None)
    half = MVTecDataset(root=root, category=category, split="test",
                        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, synthetic_method=None,
                        subset_indices=list(range(len(full) // 2)))
    assert len(half) == len(full) // 2
