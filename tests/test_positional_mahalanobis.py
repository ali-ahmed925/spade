"""Tests for position-conditioned Mahalanobis scoring.

The central test is `test_positional_detects_swap_that_global_cannot`: it builds
the exact failure mode we believe explains transistor/misplaced and cable_swap —
a patch that is perfectly normal *as a patch* but sits in the wrong *position* —
and shows the global scorer is blind to it while the positional one is not.

If that test ever fails, the P1 hypothesis is wrong and the design should change.
"""

import pytest
import torch

from models.mahalanobis_scoring import MahalanobisScoring
from models.positional_mahalanobis import (
    PositionalMahalanobisScoring,
    PositionalStatsAccumulator,
    build_positional_scorer,
)

N_POS, DIM = 16, 8


def _structured_images(n_images: int, seed: int = 0) -> torch.Tensor:
    """(n, N_POS, DIM) where each position has its own characteristic mean.

    Stands in for an aligned object: position 0 is always 'board', position 5 is
    always 'copper', etc. Every patch is drawn from the same overall family, so
    the pooled distribution looks identical regardless of position.
    """
    g = torch.Generator().manual_seed(seed)
    # A DISTINCT mean per position. Drawn from a fixed generator so every
    # position's mean is itself a plausible draw from the pooled distribution —
    # that is what makes a swapped patch invisible to a global scorer.
    mean_gen = torch.Generator().manual_seed(12345)
    position_means = torch.randn(N_POS, DIM, generator=mean_gen) * 3.0
    noise = torch.randn(n_images, N_POS, DIM, generator=g) * 0.25
    return position_means.unsqueeze(0) + noise


def _fit(images: torch.Tensor, **kw) -> dict:
    acc = PositionalStatsAccumulator(feature_dim=DIM, num_positions=N_POS)
    acc.update(images)
    return acc.finalize(**kw)


# ── The hypothesis ───────────────────────────────────────────────────────────
def test_positional_detects_swap_that_global_cannot():
    """A misplaced-but-otherwise-normal patch: invisible globally, visible positionally."""
    train = _structured_images(200, seed=0)
    stats = _fit(train)

    clean = _structured_images(1, seed=99)
    swapped = clean.clone()
    # swap two positions' contents — every patch value remains one the training
    # set contains, only the arrangement is wrong
    swapped[0, 3], swapped[0, 11] = clean[0, 11].clone(), clean[0, 3].clone()

    positional = build_positional_scorer(stats, blend=1.0)
    global_scorer = MahalanobisScoring(feature_dim=DIM, regularization=1e-4, gamma=1.0)
    global_scorer.update_statistics(train.reshape(-1, DIM))

    pos_clean = positional(clean).max()
    pos_swapped = positional(swapped).max()
    glob_clean = global_scorer(clean).max()
    glob_swapped = global_scorer(swapped).max()

    pos_ratio = float(pos_swapped / pos_clean)
    glob_ratio = float(glob_swapped / glob_clean)

    assert pos_ratio > 5.0, (
        f"positional scorer barely reacted to a swap (ratio {pos_ratio:.2f}) — "
        "the P1 hypothesis does not hold"
    )
    assert glob_ratio < 1.5, (
        f"global scorer reacted to a pure position swap (ratio {glob_ratio:.2f}); "
        "the test data is not position-neutral enough to prove the point"
    )
    assert pos_ratio > glob_ratio * 3


def test_blend_zero_reproduces_position_agnostic_behaviour():
    """blend=0 is the control arm: global mean everywhere."""
    train = _structured_images(150, seed=1)
    stats = _fit(train)

    control = build_positional_scorer(stats, blend=0.0)
    assert torch.allclose(control.mu[0], control.mu[7], atol=1e-6)
    assert torch.allclose(control.mu[0], stats["mu_global"], atol=1e-6)

    positional = build_positional_scorer(stats, blend=1.0)
    assert not torch.allclose(positional.mu[0], positional.mu[7], atol=1e-3)


def test_blend_interpolates():
    stats = _fit(_structured_images(100, seed=2))
    half = build_positional_scorer(stats, blend=0.5)
    expected = 0.5 * stats["mu_positional"][4] + 0.5 * stats["mu_global"]
    assert torch.allclose(half.mu[4], expected, atol=1e-6)


# ── Statistics correctness ───────────────────────────────────────────────────
def test_means_match_a_direct_computation():
    images = _structured_images(64, seed=3)
    stats = _fit(images)
    assert torch.allclose(stats["mu_positional"], images.mean(dim=0), atol=1e-4)
    assert torch.allclose(stats["mu_global"], images.mean(dim=(0, 1)), atol=1e-4)


def test_pooled_covariance_matches_two_pass_computation():
    """The one-pass identity must equal the naive centered computation."""
    images = _structured_images(80, seed=4)
    stats = _fit(images, regularization=0.0)

    centered = (images - images.mean(dim=0, keepdim=True)).reshape(-1, DIM).double()
    dof = images.shape[0] * N_POS - N_POS
    sigma_ref = (centered.T @ centered) / dof
    sigma_ref = sigma_ref + 0.0 * torch.eye(DIM, dtype=torch.float64)

    sigma_from_inv = torch.linalg.inv(stats["sigma_inv"].double())
    assert torch.allclose(sigma_from_inv, sigma_ref, atol=1e-3), (
        "one-pass scatter identity disagrees with the direct two-pass covariance"
    )


def test_batched_updates_equal_single_update():
    images = _structured_images(40, seed=5)
    single = _fit(images)

    acc = PositionalStatsAccumulator(feature_dim=DIM, num_positions=N_POS)
    for i in range(0, 40, 7):
        acc.update(images[i : i + 7])
    chunked = acc.finalize()

    assert torch.allclose(single["mu_positional"], chunked["mu_positional"], atol=1e-5)
    assert torch.allclose(single["sigma_inv"], chunked["sigma_inv"], atol=1e-3)


def test_shrinkage_improves_conditioning():
    """Shrinkage must reduce the condition number, not just change numbers."""
    images = _structured_images(20, seed=6)   # deliberately few samples
    plain = _fit(images, regularization=1e-6, shrinkage=0.0)
    shrunk = _fit(images, regularization=1e-6, shrinkage=0.5)

    cond_plain = torch.linalg.cond(torch.linalg.inv(plain["sigma_inv"].double()))
    cond_shrunk = torch.linalg.cond(torch.linalg.inv(shrunk["sigma_inv"].double()))
    assert cond_shrunk < cond_plain


def test_reports_sample_to_dimension_ratio():
    stats = _fit(_structured_images(30, seed=7))
    assert stats["samples_per_dim"] == pytest.approx(30 * N_POS / DIM)


# ── Interface compatibility with MahalanobisScoring ──────────────────────────
def test_uninitialized_scorer_returns_zeros():
    scorer = PositionalMahalanobisScoring(feature_dim=DIM, num_positions=N_POS)
    out = scorer(torch.randn(3, N_POS, DIM))
    assert out.shape == (3, N_POS)
    assert torch.all(out == 0)


def test_rejects_wrong_patch_count():
    stats = _fit(_structured_images(50, seed=8))
    scorer = build_positional_scorer(stats)
    with pytest.raises(ValueError, match="patches"):
        scorer(torch.randn(2, N_POS + 1, DIM))


def test_is_drop_in_for_spade():
    """Must satisfy the same contract SPADE expects of mahalanobis_scorer."""
    stats = _fit(_structured_images(50, seed=9))
    scorer = build_positional_scorer(stats)
    x = torch.randn(2, N_POS, DIM)

    out = scorer(x)
    assert out.shape == (2, N_POS)
    assert torch.all(out >= 0), "scores must be non-negative like squared distances"
    assert bool(scorer.is_initialized)
    assert "mu" in dict(scorer.named_buffers())
    assert "sigma_inv" in dict(scorer.named_buffers())


def test_gamma_amplification_applied():
    stats = _fit(_structured_images(50, seed=10))
    x = torch.randn(2, N_POS, DIM) * 3
    linear = build_positional_scorer(stats, gamma=1.0)(x)
    amplified = build_positional_scorer(stats, gamma=2.0)(x)
    assert torch.allclose(amplified, linear.pow(2.0), rtol=1e-3)


def test_rejects_invalid_blend():
    stats = _fit(_structured_images(20, seed=11))
    with pytest.raises(ValueError, match="blend"):
        build_positional_scorer(stats, blend=1.5)


# ── End-to-end wiring ────────────────────────────────────────────────────────
def test_swaps_into_spade_and_changes_scores():
    """The scorer must work as SPADE's mahalanobis_scorer without adaptation."""
    from tests.stub_blip2 import fit_statistics, make_stub_spade

    torch.manual_seed(0)
    images = torch.randn(2, 3, 224, 224)
    model = make_stub_spade()
    fit_statistics(model, images)
    model.eval()

    with torch.no_grad():
        before = model(images)["patch_scores"].clone()
        embeds = model.vision_encoder(images)[:, 1:, :].float()

    acc = PositionalStatsAccumulator(
        feature_dim=embeds.shape[-1], num_positions=embeds.shape[1]
    )
    acc.update(embeds)
    acc.update(embeds + torch.randn_like(embeds) * 0.1)   # >=2 images required
    stats = acc.finalize(regularization=1e-4)

    model.mahalanobis_scorer = build_positional_scorer(stats, blend=1.0)
    with torch.no_grad():
        after = model(images)["patch_scores"]

    assert after.shape == before.shape
    assert torch.isfinite(after).all()
    assert not torch.allclose(before, after), "swapping the scorer changed nothing"


def test_accumulator_rejects_single_image():
    acc = PositionalStatsAccumulator(feature_dim=DIM, num_positions=N_POS)
    acc.update(torch.randn(1, N_POS, DIM))
    with pytest.raises(RuntimeError, match="at least 2"):
        acc.finalize()
