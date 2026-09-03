"""Tests for the local detection pathway: neighbourhood pooling + memory bank.

The pathway exists to remove three structural defects, and each is pinned here:

  * the normal model was ONE Gaussian over all patches of all positions;
  * its covariance was an EMA of window-covariances, not the covariance of the
    training set;
  * nothing in the descriptor path was spatially local, and no raw local
    appearance reached the score without global mixing.
"""

import numpy as np
import pytest
import torch

from models.mahalanobis_scoring import MahalanobisScoring
from models.memory_bank import CoresetMemoryBank, greedy_coreset_indices
from models.neighborhood import NeighborhoodAggregator


# ── neighbourhood aggregation ────────────────────────────────────────────
def test_pooling_preserves_shape():
    agg = NeighborhoodAggregator(grid_size=8, kernel_size=3)
    out = agg(torch.randn(2, 64, 16))
    assert out.shape == (2, 64, 16)


def test_kernel_one_is_exactly_the_identity():
    """The ablation control must change nothing at all."""
    agg = NeighborhoodAggregator(grid_size=8, kernel_size=1)
    x = torch.randn(2, 64, 16)
    assert torch.equal(agg(x), x)
    assert not agg.enabled


def test_pooling_averages_the_actual_neighbourhood():
    agg = NeighborhoodAggregator(grid_size=3, kernel_size=3)
    x = torch.arange(9, dtype=torch.float32).reshape(1, 9, 1)
    out = agg(x).reshape(3, 3)
    # centre cell sees all nine values
    assert out[1, 1] == pytest.approx(4.0)
    # top-left sees only the four real neighbours 0,1,3,4
    assert out[0, 0] == pytest.approx((0 + 1 + 3 + 4) / 4)


def test_border_cells_are_not_shrunk_toward_zero():
    """count_include_pad=True would dim every border patch, which then reads as
    anomalous under any distance score -- a border artefact indistinguishable
    from a real detection."""
    agg = NeighborhoodAggregator(grid_size=6, kernel_size=3)
    x = torch.ones(1, 36, 4)
    out = agg(x)
    assert torch.allclose(out, torch.ones_like(out)), "a constant field must stay constant"


def test_even_kernel_is_rejected():
    with pytest.raises(ValueError, match="odd"):
        NeighborhoodAggregator(grid_size=8, kernel_size=2)


def test_wrong_grid_size_is_caught():
    agg = NeighborhoodAggregator(grid_size=8, kernel_size=3)
    with pytest.raises(ValueError, match="expected 64 patches"):
        agg(torch.randn(1, 63, 4))


# ── coreset selection ────────────────────────────────────────────────────
def test_coreset_is_deterministic_given_a_seed():
    """The bank ships inside the checkpoint, so selection must be reproducible."""
    x = torch.randn(300, 12)
    a = greedy_coreset_indices(x, 20, projection_dim=8, seed=7)
    b = greedy_coreset_indices(x, 20, projection_dim=8, seed=7)
    assert torch.equal(a, b)


def test_coreset_indices_are_unique():
    idx = greedy_coreset_indices(torch.randn(200, 8), 30, seed=1)
    assert len(set(idx.tolist())) == 30


def test_coreset_covers_separated_modes():
    """The point of k-center over random sampling: keep the outlying modes.

    Random sampling from a 99/1 split would usually miss the small cluster
    entirely, and a legitimate but rare normal patch would then match nothing.
    """
    torch.manual_seed(0)
    big = torch.randn(500, 4) * 0.1
    small = torch.randn(5, 4) * 0.1 + 50.0
    x = torch.cat([big, small])
    idx = set(greedy_coreset_indices(x, 12, projection_dim=None, seed=0).tolist())
    assert idx & set(range(500, 505)), "k-center must reach the far mode"


def test_coreset_returns_everything_when_asked_for_more_than_exists():
    idx = greedy_coreset_indices(torch.randn(10, 4), 50, seed=0)
    assert len(idx) == 10


# ── memory bank ──────────────────────────────────────────────────────────
def test_unfitted_bank_scores_zero_rather_than_raising():
    """Training runs before the bank exists; it must contribute nothing then."""
    bank = CoresetMemoryBank(feature_dim=8)
    assert not bank.fitted
    out = bank(torch.randn(2, 5, 8))
    assert out.shape == (2, 5) and torch.count_nonzero(out) == 0


def test_banked_vectors_score_near_zero_and_far_ones_score_high():
    """cdist's matrix-multiply path costs precision near zero (~5e-4), so this
    asserts the separation that matters rather than an exact zero."""
    bank = CoresetMemoryBank(feature_dim=4, coreset_ratio=1.0, k=1)
    normal = torch.randn(50, 4)
    bank.fit(normal)

    on_bank = float(bank(normal[:3].unsqueeze(0)).max())
    far = float(bank((normal[:3] + 100.0).unsqueeze(0)).min())
    assert on_bank < 1e-2
    assert far > 100.0
    assert far / max(on_bank, 1e-9) > 1e4, "separation must be orders of magnitude"


def test_scores_are_finite_and_non_negative():
    """The mm formula can go slightly negative under the sqrt; cdist clamps, but
    a NaN here would silently poison every downstream metric."""
    torch.manual_seed(0)
    bank = CoresetMemoryBank(feature_dim=16, coreset_ratio=1.0)
    normal = torch.randn(200, 16) * 1e3
    bank.fit(normal)
    out = bank(torch.cat([normal[:5], normal[:5] * (1 + 1e-7)]).unsqueeze(0))
    assert torch.isfinite(out).all()
    assert (out >= 0).all()


def test_score_equals_true_nearest_neighbour_distance():
    torch.manual_seed(0)
    bank = CoresetMemoryBank(feature_dim=6, coreset_ratio=1.0, k=1, query_chunk=3)
    normal = torch.randn(40, 6)
    bank.fit(normal)

    query = torch.randn(2, 7, 6)
    got = bank(query)
    expected = torch.cdist(query.reshape(-1, 6), normal).min(dim=1).values.view(2, 7)
    assert torch.allclose(got, expected, atol=1e-5)


def test_chunking_does_not_change_the_answer():
    torch.manual_seed(0)
    normal = torch.randn(60, 5)
    query = torch.randn(1, 20, 5)
    scores = []
    for chunk in (1, 7, 1000):
        bank = CoresetMemoryBank(feature_dim=5, coreset_ratio=1.0, query_chunk=chunk)
        bank.fit(normal)
        scores.append(bank(query))
    assert torch.allclose(scores[0], scores[1], atol=1e-6)
    assert torch.allclose(scores[0], scores[2], atol=1e-6)


def test_coreset_ratio_controls_bank_size():
    bank = CoresetMemoryBank(feature_dim=4, coreset_ratio=0.1)
    info = bank.fit(torch.randn(1000, 4))
    assert info["selected"] == 100
    assert bank.bank.shape == (100, 4)


def test_max_size_caps_the_bank():
    bank = CoresetMemoryBank(feature_dim=4, coreset_ratio=1.0, max_size=25)
    assert bank.fit(torch.randn(400, 4))["selected"] == 25


def test_k_greater_than_one_averages_neighbours():
    bank = CoresetMemoryBank(feature_dim=2, coreset_ratio=1.0, k=3)
    normal = torch.tensor([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0], [90.0, 0.0]])
    bank.fit(normal)
    got = float(bank(torch.zeros(1, 1, 2)))
    assert got == pytest.approx((0.0 + 3.0 + 6.0) / 3, abs=1e-4)


def test_fit_rejects_empty_and_misshapen_input():
    bank = CoresetMemoryBank(feature_dim=4)
    with pytest.raises(ValueError, match="zero patches"):
        bank.fit(torch.zeros(0, 4))
    with pytest.raises(ValueError, match="expected"):
        bank.fit(torch.randn(10, 5))


def test_bank_gradients_reach_the_query():
    """Scoring must stay differentiable w.r.t. the features, so the fusion that
    produces them is not cut off from any future objective."""
    bank = CoresetMemoryBank(feature_dim=4, coreset_ratio=1.0)
    bank.fit(torch.randn(20, 4))
    query = torch.randn(1, 3, 4, requires_grad=True)
    bank(query).sum().backward()
    assert query.grad is not None and float(query.grad.abs().sum()) > 0


# ── closed-form Mahalanobis fit ──────────────────────────────────────────
def test_closed_form_fit_matches_numpy():
    """The EMA path did not estimate the covariance of the training set; this
    one must, exactly."""
    torch.manual_seed(0)
    dim = 12
    x = torch.randn(4000, dim) @ torch.randn(dim, dim) + 2.0

    scorer = MahalanobisScoring(feature_dim=dim, regularization=1e-4, gamma=1.0)
    scorer.fit_from_normal_patches(x)

    expected_cov = np.cov(x.numpy().T) + 1e-4 * np.eye(dim)
    assert np.allclose(scorer.mu.numpy(), x.numpy().mean(0), atol=1e-5)
    assert np.allclose(scorer.sigma_inv.numpy(), np.linalg.inv(expected_cov), atol=1e-3)


def test_closed_form_fit_refuses_an_underdetermined_covariance():
    scorer = MahalanobisScoring(feature_dim=64, regularization=1e-4)
    with pytest.raises(ValueError, match="cannot determine"):
        scorer.fit_from_normal_patches(torch.randn(40, 64))


def test_closed_form_fit_is_order_invariant():
    """A property the EMA path did NOT have: its result depended on batch order."""
    torch.manual_seed(0)
    x = torch.randn(2000, 8) @ torch.randn(8, 8)

    a = MahalanobisScoring(feature_dim=8, regularization=1e-4)
    a.fit_from_normal_patches(x)
    b = MahalanobisScoring(feature_dim=8, regularization=1e-4)
    b.fit_from_normal_patches(x[torch.randperm(2000)])

    assert torch.allclose(a.mu, b.mu, atol=1e-4)
    assert torch.allclose(a.sigma_inv, b.sigma_inv, atol=1e-2)


# ── feature geometry diagnostics ─────────────────────────────────────────
def test_effective_rank_detects_removed_directions():
    """The failure kNN cannot tolerate: variance concentrated into a subspace,
    so a defect along a removed direction becomes invisible."""
    from models.normal_fit import feature_geometry

    torch.manual_seed(0)
    isotropic = feature_geometry(torch.randn(4000, 32))
    degenerate = feature_geometry(torch.randn(4000, 3) @ torch.randn(3, 32))

    assert isotropic["effective_rank"] > 30
    assert degenerate["effective_rank"] < 4


def test_norm_and_rank_are_independent_signals():
    """Uniform shrinkage and direction removal are different failures and must
    not be confused: the score is normalised by scale, so only rank loss is
    fatal."""
    from models.normal_fit import feature_geometry

    torch.manual_seed(0)
    base = torch.randn(4000, 32)
    full = feature_geometry(base)
    shrunk = feature_geometry(base * 0.01)

    assert shrunk["norm"] < full["norm"] / 50, "shrinkage must show in the norm"
    assert abs(shrunk["effective_rank"] - full["effective_rank"]) < 1.0, (
        "uniform shrinkage must NOT register as rank loss"
    )


def test_mahalanobis_loss_cannot_be_reduced_by_reshaping_features():
    """The identity that invalidates the 'collapse objective' reading:
    E[(x-mu)' S^-1 (x-mu)] = trace(S^-1 S) = D, for ANY distribution, when S is
    the sample covariance of the same data. Scale and multimodality are both
    irrelevant. What the loss CAN exploit is stale statistics."""
    torch.manual_seed(0)
    dim = 24

    def mean_distance(x):
        scorer = MahalanobisScoring(feature_dim=dim, regularization=1e-8, gamma=1.0)
        scorer.fit_from_normal_patches(x)
        return float(scorer(x.unsqueeze(0)).mean())

    base = torch.randn(8000, dim) @ torch.randn(dim, dim)
    two_modes = torch.cat([torch.randn(4000, dim), torch.randn(4000, dim) + 30.0])

    assert mean_distance(base) == pytest.approx(dim, rel=0.02)
    assert mean_distance(base * 100) == pytest.approx(dim, rel=0.02)
    assert mean_distance(two_modes) == pytest.approx(dim, rel=0.02)


def test_stale_statistics_create_a_shrinkage_incentive():
    """The gradient that DOES reach fusion: with Sigma_inv refit only every 100
    steps, halving the features between refits drops the loss ~4x."""
    torch.manual_seed(0)
    dim = 24
    x = torch.randn(8000, dim) @ torch.randn(dim, dim)

    scorer = MahalanobisScoring(feature_dim=dim, regularization=1e-8, gamma=1.0)
    scorer.fit_from_normal_patches(x)                     # statistics fitted on x

    matched = float(scorer(x.unsqueeze(0)).mean())
    halved = float(scorer((x * 0.5).unsqueeze(0)).mean())
    assert halved == pytest.approx(matched / 4, rel=0.05)
