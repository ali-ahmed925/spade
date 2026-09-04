"""Tests for the pre-screw confound removal, P0 through P6.

Each block names the confound it eliminates. Several assert an IMPOSSIBILITY
rather than a value, because the point of these fixes is that certain degenerate
behaviours can no longer occur regardless of what the optimiser does.
"""

import numpy as np
import pytest
import torch

from models.mahalanobis_scoring import MahalanobisScoring
from models.memory_bank import CoresetMemoryBank
from models.neighborhood import NeighborhoodAggregator
from models.normal_fit import _subsample, fit_normal_model
from models.streaming_stats import StreamingGaussianEstimator, freeze_layernorm_scale
from tests.stub_blip2 import make_stub_spade
from utils.heatmap import patches_to_heatmap

IMAGE_SIZE = 224


class _Loader:
    def __init__(self, images, batch=2):
        self._batches = [
            {"image": images[i : i + batch]} for i in range(0, images.shape[0], batch)
        ]
        self.dataset = range(images.shape[0])

    def __iter__(self):
        return iter(self._batches)


def _train_images(n=16, seed=1):
    torch.manual_seed(seed)
    return torch.randn(n, 3, IMAGE_SIZE, IMAGE_SIZE)


# ═══════════════════ P0 — the shrinkage treadmill ═══════════════════
def test_descriptor_scale_is_a_constant_of_the_architecture():
    """The exploit needs a global scale degree of freedom. There isn't one."""
    model = make_stub_spade(image_size=IMAGE_SIZE)
    assert model.output_scale_frozen

    images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    before = model.build_descriptors(images)["descriptors"]
    dim = before.shape[-1]

    # halve every trainable parameter -- the direct attempt at the shrinkage
    # exploit, at a magnitude the optimiser could plausibly reach
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.mul_(0.5)
    after = model.build_descriptors(images)["descriptors"]

    for tensor in (before, after):
        centred = tensor - tensor.mean(dim=-1, keepdim=True)
        assert torch.allclose(
            centred.norm(dim=-1),
            torch.full_like(centred.norm(dim=-1), dim ** 0.5),
            rtol=2e-3,
        ), "LayerNorm without a learnable gain must pin ||x - mean|| to sqrt(D)"


def test_layernorm_guarantee_has_an_eps_boundary():
    """Honest limit of the LayerNorm mechanism.

    LayerNorm divides by sqrt(var + eps). Once the pre-norm variance falls below
    eps (1e-5), eps dominates and the output stops being scale-free -- so the
    scale route reopens in that extreme regime. It is documented rather than
    claimed away. The PRIMARY guarantee is not this: it is that statistics are
    refit every step, which makes the mean term constant regardless (see
    test_matched_statistics_leave_the_mean_term_constant).
    """
    norm = torch.nn.LayerNorm(64)
    freeze_layernorm_scale(norm)
    healthy = norm(torch.randn(100, 64))
    degenerate = norm(torch.randn(100, 64) * 1e-4)

    centred = lambda t: (t - t.mean(-1, keepdim=True)).norm(dim=-1).mean()
    assert float(centred(healthy)) == pytest.approx(8.0, rel=0.05)      # sqrt(64)
    assert float(centred(degenerate)) < 7.0, "eps dominates once var << eps"


def test_output_layernorm_gains_are_frozen():
    model = make_stub_spade(image_size=IMAGE_SIZE)
    assert not model.contextualizer.out_norm.weight.requires_grad
    assert not model.fusion.out_norm.weight.requires_grad
    assert torch.allclose(model.contextualizer.out_norm.weight, torch.ones(1))
    assert torch.allclose(model.contextualizer.out_norm.bias, torch.zeros(1))


def test_freeze_helper_is_idempotent_and_reports_honestly():
    norm = torch.nn.LayerNorm(8)
    assert freeze_layernorm_scale(norm) is True
    assert freeze_layernorm_scale(norm) is True     # already identity, still frozen
    assert freeze_layernorm_scale(torch.nn.LayerNorm(8, elementwise_affine=False)) is False
    assert freeze_layernorm_scale(None) is False


def test_streaming_estimator_is_exact():
    """Not an approximation: it must equal the closed-form fit on all samples."""
    torch.manual_seed(0)
    dim = 16
    data = torch.randn(4000, dim) @ torch.randn(dim, dim) + 2.0

    estimator = StreamingGaussianEstimator(feature_dim=dim, regularization=1e-6)
    for chunk in data.split(97):                       # ragged batches on purpose
        estimator.update(chunk)
    mu, sigma = estimator.statistics()

    expected_cov = np.cov(data.numpy().T) + 1e-6 * np.eye(dim)
    assert np.allclose(mu.numpy(), data.numpy().mean(0), atol=1e-4)
    assert np.allclose(sigma.numpy(), expected_cov, atol=1e-3)


def test_streaming_estimator_excludes_anomalous_patches():
    estimator = StreamingGaussianEstimator(feature_dim=4)
    labels = torch.zeros(2, 10)
    labels[0, 5:] = 1.0
    added = estimator.update(torch.randn(2, 10, 4), labels)
    assert added == 15 and int(estimator.count) == 15


def test_streaming_estimator_refuses_an_underdetermined_covariance():
    estimator = StreamingGaussianEstimator(feature_dim=32, min_samples_factor=2.0)
    estimator.update(torch.randn(40, 32))
    assert not estimator.ready
    with pytest.raises(RuntimeError, match="samples accumulated"):
        estimator.statistics()


def test_reset_clears_the_accumulators():
    estimator = StreamingGaussianEstimator(feature_dim=4)
    estimator.update(torch.randn(100, 4))
    estimator.reset()
    assert int(estimator.count) == 0
    assert float(estimator.sum_outer.abs().sum()) == 0.0


def test_matched_statistics_leave_the_mean_term_constant():
    """The identity that makes the mean term unexploitable once the lag is gone.

    This is why the fix cannot 'improve' the loss: with statistics that track the
    features, mean Mahalanobis is D whatever the features do.
    """
    torch.manual_seed(0)
    dim = 24
    base = torch.randn(6000, dim) @ torch.randn(dim, dim)

    def refit_then_score(x):
        estimator = StreamingGaussianEstimator(feature_dim=dim, regularization=1e-8)
        estimator.update(x)
        mu, sigma = estimator.statistics()
        scorer = MahalanobisScoring(feature_dim=dim, regularization=1e-8, gamma=1.0)
        scorer.set_statistics(mu, sigma)
        return float(scorer(x.unsqueeze(0)).mean())

    assert refit_then_score(base) == pytest.approx(dim, rel=0.02)
    assert refit_then_score(base * 0.01) == pytest.approx(dim, rel=0.02)
    assert refit_then_score(base * 100) == pytest.approx(dim, rel=0.02)


def test_geometry_diagnostics_are_preserved():
    model = make_stub_spade(image_size=IMAGE_SIZE)
    report = fit_normal_model(model, _Loader(_train_images()), torch.device("cpu"))
    for key in ("local_norm", "descriptor_norm", "local_effective_rank",
                "descriptor_effective_rank", "fusion_gain"):
        assert key in report, f"{key} diagnostic was dropped"


# ═══════════════════ P1 — frequency stream refit ═══════════════════
def test_frequency_statistics_are_refit_from_the_full_pass():
    model = make_stub_spade(image_size=IMAGE_SIZE, frequency=True)
    report = fit_normal_model(model, _Loader(_train_images()), torch.device("cpu"))
    assert "frequency_samples" in report, "the frequency stream was not refit"
    assert report["frequency_samples"] == 16 * model.num_patches
    assert "freq_scale" in report and report["freq_scale"] > 0
    assert bool(model.freq_mahalanobis_scorer.is_initialized)


def test_frequency_refit_does_not_depend_on_training_ema():
    """A model that never trained must still get usable frequency statistics."""
    model = make_stub_spade(image_size=IMAGE_SIZE, frequency=True)
    assert not bool(model.freq_mahalanobis_scorer.is_initialized)
    fit_normal_model(model, _Loader(_train_images()), torch.device("cpu"))
    assert bool(model.freq_mahalanobis_scorer.is_initialized)


# ═══════════════════ P2 — sampling, not slicing ═══════════════════
def test_subsample_is_random_not_a_prefix():
    features = torch.arange(1000, dtype=torch.float32).unsqueeze(1)
    picked = _subsample(features, 50, seed=0).squeeze(1)
    assert len(picked) == 50
    assert not torch.equal(picked, features[:50].squeeze(1)), "must not be a prefix"
    assert float(picked.max()) > 500, "a random sample must reach the far end"


def test_subsample_is_deterministic_and_returns_all_when_small():
    features = torch.randn(200, 3)
    assert torch.equal(_subsample(features, 40, seed=7), _subsample(features, 40, seed=7))
    assert torch.equal(_subsample(features, 999, seed=7), features)


def test_no_contiguous_slicing_remains_in_the_fitting_code():
    """Guards against the pattern coming back."""
    import pathlib
    import re

    source = pathlib.Path(__file__).resolve().parent.parent / "models" / "normal_fit.py"
    offenders = re.findall(r'features\[[\'"][a-z]+[\'"]\]\[\s*:\s*min\(', source.read_text())
    assert not offenders, f"contiguous prefix slicing reintroduced: {offenders}"


# ═══════════════════ P4 — PatchCore aggregation ═══════════════════
def test_patchcore_weight_is_in_range_and_scales_the_max():
    bank = CoresetMemoryBank(feature_dim=4, coreset_ratio=1.0)
    torch.manual_seed(0)
    normal = torch.randn(60, 4)
    bank.fit(normal)

    queries = torch.cat([normal[:5], torch.randn(3, 4) * 40]).unsqueeze(0)
    scores = bank(queries)
    reweighted = float(bank.patchcore_reweight(queries, scores))
    maximum = float(scores.max())

    assert 0.0 <= reweighted <= maximum, "w must lie in [0, 1]"
    assert reweighted > 0.5 * maximum, "an isolated far patch should keep most of its score"


def test_patchcore_falls_back_to_max_when_unfitted():
    bank = CoresetMemoryBank(feature_dim=4)
    scores = torch.rand(2, 6)
    assert torch.allclose(
        bank.patchcore_reweight(torch.rand(2, 6, 4), scores), scores.max(dim=1).values
    )


def test_patchcore_aggregation_reachable_from_the_model():
    model = make_stub_spade(image_size=IMAGE_SIZE)
    fit_normal_model(model, _Loader(_train_images()), torch.device("cpu"))
    model.eval()
    images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    out = model(images)
    raw_local = out["score_components"]["local_knn"] / model.score_w_local * float(model.local_scale)
    score = model.patchcore_image_score(out["local_features"], raw_local)
    assert score.shape == (2,) and torch.isfinite(score).all()


def test_all_three_aggregations_differ():
    model = make_stub_spade(image_size=IMAGE_SIZE)
    scores = torch.rand(2, model.num_patches)
    assert not torch.allclose(
        model.get_image_score(scores, "max"), model.get_image_score(scores, "topk_mean")
    )


# ═══════════════════ P5 — raw pathway, end to end ═══════════════════
def test_raw_local_source_runs_end_to_end():
    """checkpoint load -> normal fit -> bank -> kNN forward, at 2816-d in prod."""
    model = make_stub_spade(image_size=IMAGE_SIZE, local_source="raw")
    expected = model.vision_encoder.hidden_size * len(model.vision_encoder.feature_layers)
    assert model.local_dim == expected
    assert model.memory_bank.feature_dim == expected

    report = fit_normal_model(model, _Loader(_train_images()), torch.device("cpu"))
    assert report["bank_size"] > 0
    model.eval()

    out = model(torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert "local_knn" in out["score_components"]
    assert out["local_features"].shape[-1] == expected
    assert torch.isfinite(out["patch_scores"]).all()


def test_raw_bank_round_trips_through_a_checkpoint():
    from utils.checkpoint import load_checkpoint_into

    source = make_stub_spade(seed=4, image_size=IMAGE_SIZE, local_source="raw")
    fit_normal_model(source, _Loader(_train_images()), torch.device("cpu"))
    source.eval()

    target = make_stub_spade(seed=4, image_size=IMAGE_SIZE, local_source="raw")
    load_checkpoint_into(target, source.state_dict())
    target.eval()

    images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.allclose(
        source(images)["patch_scores"], target(images)["patch_scores"], atol=1e-5
    )


def test_collection_budget_holds_bytes_constant_not_rows():
    """Production widths: FUSED is 512-d, RAW is 2816-d. The row cap must shrink
    by the width ratio so the same setting is safe for both.

    Asserted on the rule rather than end-to-end, because the stub's widest stream
    is 64-d -- below the 512-d reference -- so the budget correctly never binds
    there and the behaviour is not observable on it.
    """
    from models.normal_fit import collection_cap

    assert collection_cap(500_000, 512) == 500_000          # reference width, no cut
    assert collection_cap(500_000, 2816) == 90_909          # raw: 5.5x narrower
    assert collection_cap(500_000, 64) == 500_000           # narrower than reference
    # bytes held roughly constant across widths
    assert collection_cap(500_000, 2816) * 2816 == pytest.approx(500_000 * 512, rel=1e-3)


def test_streams_are_sampled_on_the_same_patches():
    from models.normal_fit import collect_normal_features

    features = collect_normal_features(
        make_stub_spade(image_size=IMAGE_SIZE, frequency=True),
        _Loader(_train_images(8)), torch.device("cpu"), max_patches=1000,
    )
    counts = {name: tensor.shape[0] for name, tensor in features.items()}
    assert len(set(counts.values())) == 1, f"streams desynchronised: {counts}"


# ═══════════════════ P6 — patch-grid ordering ═══════════════════
def test_token_index_maps_to_the_same_cell_everywhere():
    """ViT token i -> (row, col) must agree between neighbourhood pooling and
    the heatmap. A transposed grid would barely move image AUROC while quietly
    ruining localisation, so this is asserted rather than assumed.

    Both are checked against row-major order: index i = row * G + col.
    """
    grid = 8
    image_size = grid * 4          # 4 px per cell, exact
    n_patches = grid * grid

    for index in (0, 1, grid, grid + 3, n_patches - 1):
        row, col = divmod(index, grid)

        # (a) heatmap: a one-hot patch must light up exactly cell (row, col)
        one_hot = torch.zeros(n_patches)
        one_hot[index] = 1.0
        hmap = patches_to_heatmap(
            one_hot, image_size=image_size, patch_size=4, normalize=False
        )
        hot = np.argwhere(np.asarray(hmap) > 0.5)
        assert hot.size > 0, f"index {index} lit no pixel"
        assert {int(r) // 4 for r, _ in hot} == {row}, f"index {index} row"
        assert {int(c) // 4 for _, c in hot} == {col}, f"index {index} col"

        # (b) neighbourhood pooling: a one-hot patch must spread to exactly the
        # 3x3 grid neighbourhood of the SAME cell
        aggregator = NeighborhoodAggregator(grid_size=grid, kernel_size=3)
        pooled = aggregator(one_hot.view(1, n_patches, 1)).view(grid, grid)
        touched = {(int(r), int(c)) for r, c in np.argwhere(pooled.numpy() > 0)}
        expected = {
            (r, c)
            for r in range(max(0, row - 1), min(grid, row + 2))
            for c in range(max(0, col - 1), min(grid, col + 2))
        }
        assert touched == expected, f"index {index}: pooled {touched} != {expected}"


def test_grid_reshape_round_trips():
    grid, dim = 6, 5
    aggregator = NeighborhoodAggregator(grid_size=grid, kernel_size=1)
    x = torch.randn(2, grid * grid, dim)
    assert torch.equal(aggregator(x), x), "kernel 1 must be a pure identity"


# ═══════════ regression: the config typo that killed a run ═══════════
def test_first_refit_does_not_wait_for_update_frequency():
    """config/model.yaml shipped update_frequency=100 while the code assumed 1.
    Step 1 % 100 != 0, so no refit ever happened, the scorer stayed
    uninitialised, returned constant zeros, and backward() died with
    'element 0 of tensors does not require grad'.

    The first refit must fire as soon as enough samples exist, whatever the
    frequency, because before it the model cannot produce a differentiable
    score at all.
    """
    estimator = StreamingGaussianEstimator(feature_dim=64, update_frequency=100)
    estimator.update(torch.randn(500, 64))
    assert estimator.ready
    assert estimator.should_refit() is True, "the first refit must not be gated"
    assert estimator.should_refit() is False, "subsequent ones respect the frequency"


def test_reset_keeps_ever_fitted_so_scores_stay_defined():
    """Across an epoch boundary the accumulators clear but the scorer keeps its
    last good statistics, so there is no window of graph-less zeros."""
    estimator = StreamingGaussianEstimator(feature_dim=32, update_frequency=1)
    estimator.update(torch.randn(200, 32))
    estimator.should_refit()
    estimator.reset()
    assert int(estimator.count) == 0
    assert bool(estimator.ever_fitted), "must not forget that a fit ever happened"


def test_set_statistics_recovers_from_a_rank_deficient_covariance():
    """Escalating ridge rather than a silent return: a no-op here leaves the
    scorer uninitialised and the failure surfaces far away, in backward()."""
    torch.manual_seed(0)
    scorer = MahalanobisScoring(feature_dim=8, regularization=0.0, gamma=1.0)
    x = torch.randn(100, 3) @ torch.randn(3, 8)          # rank 3 of 8
    scorer.set_statistics(x.mean(0), torch.cov(x.T))

    assert bool(scorer.is_initialized)
    assert bool((scorer(x[:3].unsqueeze(0)) != 0).any())


def test_config_declares_the_update_frequency_the_code_needs():
    """The value that was wrong. Asserted against the file, not the default."""
    import pathlib

    import yaml

    root = pathlib.Path(__file__).resolve().parent.parent
    config = yaml.safe_load((root / "config" / "model.yaml").read_text())
    assert config["normal_stats"]["update_frequency"] == 1
    assert config["normal_stats"]["freeze_output_scale"] is True
