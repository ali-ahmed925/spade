"""The local pathway inside SPADE, end to end on the stub backbone.

Three properties have to hold together, and the value of the redesign depends
on all three:

  * the local stream reaches the score, and is reported standalone so
    local-only vs contextual-only is readable from one eval run;
  * disabling it reproduces the previous behaviour exactly, so it is a clean
    ablation rather than a confounded rewrite;
  * the Q-Former, the contextualizer, the grounding attention and the language
    path are untouched.
"""

import pytest
import torch

from models.normal_fit import fit_normal_model
from tests.stub_blip2 import make_stub_spade


BATCH, IMAGE_SIZE = 2, 224


@pytest.fixture
def images():
    torch.manual_seed(0)
    return torch.randn(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE)


class _Loader:
    """Minimal stand-in for a DataLoader over train/good."""

    def __init__(self, images, batch=2):
        self._batches = [
            {"image": images[i : i + batch]} for i in range(0, images.shape[0], batch)
        ]
        self.dataset = range(images.shape[0])

    def __iter__(self):
        return iter(self._batches)


def _fitted_model(images, **kw):
    model = make_stub_spade(image_size=IMAGE_SIZE, **kw)
    torch.manual_seed(1)
    train_images = torch.randn(16, 3, IMAGE_SIZE, IMAGE_SIZE)
    fit_normal_model(model, _Loader(train_images), torch.device("cpu"), max_patches=100_000)
    model.eval()
    return model


# ── the stream reaches the score ─────────────────────────────────────────
def test_local_stream_is_reported_standalone(images):
    model = _fitted_model(images)
    out = model(images)
    assert "local_knn" in out["score_components"], (
        "eval reads per-stream AUROC from score_components; without this entry "
        "local-only vs contextual-only cannot be compared without tuning weights"
    )
    assert out["score_components"]["local_knn"].shape == (BATCH, model.num_patches)


def test_local_stream_actually_moves_the_score(images):
    model = _fitted_model(images)
    with_local = model(images)["patch_scores"]
    model.memory_bank.reset()
    without_local = model(images)["patch_scores"]
    assert not torch.allclose(with_local, without_local), "the stream must be load-bearing"


def test_local_features_bypass_the_contextualizer(images):
    """The whole point: raw local appearance must reach the score without
    passing through global mixing over the 32 image-level query tokens."""
    model = _fitted_model(images)
    built = model.build_descriptors(images)
    assert built["local_features"].shape == (BATCH, model.num_patches, model.local_dim)
    assert model.local_dim == model.fusion.output_dim
    # the contextual descriptor is a different, wider space
    assert built["descriptors"].shape[-1] == model.descriptor_dim


def test_neighbourhood_pooling_is_applied_to_local_features(images):
    pooled = _fitted_model(images).build_descriptors(images)["local_features"]
    unpooled = _fitted_model(images, local_neighborhood=1).build_descriptors(images)[
        "local_features"
    ]
    assert not torch.allclose(pooled, unpooled)


# ── disabling it is a clean control ──────────────────────────────────────
def test_disabled_pathway_produces_no_local_stream(images):
    model = make_stub_spade(image_size=IMAGE_SIZE, local_enabled=False)
    out = model(images)
    assert "local_features" not in out
    assert "local_knn" not in out["score_components"]


def test_unfitted_bank_leaves_the_score_untouched(images):
    """Enabling the pathway must not change a training run: the bank does not
    exist until fit_normal_model runs after the last epoch."""
    enabled = make_stub_spade(seed=3, image_size=IMAGE_SIZE, local_enabled=True)
    disabled = make_stub_spade(seed=3, image_size=IMAGE_SIZE, local_enabled=False)
    disabled.load_state_dict(
        {k: v for k, v in enabled.state_dict().items() if k in disabled.state_dict()},
        strict=False,
    )
    enabled.eval(); disabled.eval()
    assert not enabled.memory_bank.fitted
    assert torch.allclose(
        enabled(images)["patch_scores"], disabled(images)["patch_scores"], atol=1e-6
    )


# ── the Q-Former and language path survive ───────────────────────────────
def test_qformer_and_language_path_are_intact(images):
    model = _fitted_model(images)
    out = model(images, return_attention=True)

    # the stub uses fewer query tokens than production's 32; read the count from
    # the model so the test pins the invariant, not the fixture's size
    n_queries = out["query_embeds"].shape[1]
    assert n_queries > 0, "query tokens preserved"
    assert out["visual_tokens"].shape[:2] == (BATCH, n_queries), "language path preserved"

    attention = out["patch_query_attention"]
    assert attention.shape == (BATCH, model.num_patches, n_queries)
    assert torch.allclose(
        attention.sum(dim=-1), torch.ones_like(attention.sum(dim=-1)), atol=1e-4
    ), "grounding depends on this being a softmax over queries"


def test_contextual_stream_still_contributes(images):
    """The Q-Former must keep its own route into the score, not be replaced."""
    model = _fitted_model(images)
    components = model(images)["score_components"]
    assert "contextual_mahalanobis" in components
    assert float(components["contextual_mahalanobis"].detach().abs().sum()) > 0


# ── the full fit ─────────────────────────────────────────────────────────
def test_fit_reports_what_it_used(images):
    model = make_stub_spade(image_size=IMAGE_SIZE)
    torch.manual_seed(2)
    report = fit_normal_model(
        model, _Loader(torch.randn(16, 3, IMAGE_SIZE, IMAGE_SIZE)),
        torch.device("cpu"), max_patches=100_000,
    )
    assert report["mahalanobis_samples"] == 16 * model.num_patches, (
        "the fit must see every patch, not a 20k rolling window"
    )
    assert report["bank_size"] > 0
    assert report["local_scale"] > 0


def test_fit_subsamples_without_truncating(images):
    """Over the cap, patches are kept by Bernoulli draw. Taking the first N
    would bias the fit toward whichever images sort first."""
    model = make_stub_spade(image_size=IMAGE_SIZE)
    torch.manual_seed(3)
    report = fit_normal_model(
        model, _Loader(torch.randn(16, 3, IMAGE_SIZE, IMAGE_SIZE)),
        torch.device("cpu"), max_patches=1000,
    )
    kept = report["mahalanobis_samples"]
    assert 500 < kept < 2000, f"expected ~1000 sampled patches, got {kept}"


def test_streams_are_calibrated_by_the_fit(images):
    model = _fitted_model(images)
    assert float(model.local_scale) > 0
    assert float(model.mahal_scale) > 0
    assert bool(model.stream_scales_initialized)
