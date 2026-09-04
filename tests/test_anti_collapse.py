"""Anti-collapse regularisation (Fix 1).

The failure it addresses, measured on screw: descriptor effective rank fell
9.3 -> 2.3 over five epochs while image AUROC fell 0.866 -> 0.774. The cause is
that every term in the normal-only objective is minimised by discarding
dimensions -- mean Mahalanobis EQUALS the effective rank.

The important test here is not that the penalty computes; it is that under the
REAL collapsing objective, optimisation ends up at a higher rank with the
regulariser than without.
"""

import pytest
import torch
import torch.nn.functional as F

from losses.collapse_loss import AntiCollapseLoss, effective_rank
from losses.total_loss import TotalLoss
from tests.stub_blip2 import fit_statistics, make_stub_spade

DIM, N = 64, 512


def _collapsing_objective(z, reg=1e-4):
    """Mean Mahalanobis with statistics refit to match -- the real mechanism.

    Equals the number of eigenvalues above the ridge, so minimising it is
    literally minimising how many dimensions the representation uses.
    """
    centered = z - z.mean(0, keepdim=True)
    sigma = (centered.T @ centered) / (z.shape[0] - 1) + reg * torch.eye(z.shape[1])
    return (centered @ torch.linalg.inv(sigma) * centered).sum(1).mean()


def _optimise(anti, steps=300, seed=1):
    torch.manual_seed(0)
    data = torch.randn(N, DIM)
    torch.manual_seed(seed)
    layer = torch.nn.Linear(DIM, DIM, bias=False)
    optimiser = torch.optim.Adam(layer.parameters(), lr=1e-2)
    for _ in range(steps):
        z = F.layer_norm(layer(data), (DIM,))
        loss = _collapsing_objective(z)
        if anti is not None:
            loss = loss + anti(z)[0]
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        return effective_rank(F.layer_norm(layer(data), (DIM,)))


# ── the property that matters ────────────────────────────────────────────
def test_regulariser_resists_the_real_collapsing_objective():
    without = _optimise(None)
    with_reg = _optimise(AntiCollapseLoss(variance_weight=5.0, covariance_weight=0.2))
    assert with_reg > without + 10.0, (
        f"rank {with_reg:.1f} with the regulariser vs {without:.1f} without -- "
        "it must materially resist collapse, not merely be present"
    )


def test_stronger_weights_resist_more():
    """Monotone in the weight, so the knob means something."""
    weak = _optimise(AntiCollapseLoss(variance_weight=1.0, covariance_weight=0.04))
    strong = _optimise(AntiCollapseLoss(variance_weight=5.0, covariance_weight=0.2))
    assert strong > weak


# ── the penalty itself ───────────────────────────────────────────────────
def test_collapsed_features_are_penalised_more_than_isotropic_ones():
    torch.manual_seed(0)
    loss = AntiCollapseLoss(variance_weight=5.0, covariance_weight=0.2)
    isotropic = torch.randn(400, DIM)
    collapsed = torch.randn(400, 2) @ torch.randn(2, DIM)      # rank 2 of 64
    assert float(loss(collapsed)[0]) > float(loss(isotropic)[0])


def test_variance_term_fires_on_a_dead_dimension():
    loss = AntiCollapseLoss(variance_weight=1.0, covariance_weight=0.0)
    healthy = torch.randn(200, 8)
    dead = healthy.clone()
    dead[:, 3] = 0.0
    assert float(loss(dead)[1]["collapse/variance"]) > float(
        loss(healthy)[1]["collapse/variance"]
    )


def test_covariance_term_fires_on_duplicated_dimensions():
    loss = AntiCollapseLoss(variance_weight=0.0, covariance_weight=1.0)
    torch.manual_seed(0)
    independent = torch.randn(400, 8)
    duplicated = independent.clone()
    duplicated[:, 4:] = duplicated[:, :4]                      # half are copies
    assert float(loss(duplicated)[1]["collapse/covariance"]) > float(
        loss(independent)[1]["collapse/covariance"]
    )


def test_no_nan_on_an_already_dead_dimension():
    """eps inside the sqrt: a dimension at exactly zero must still give a
    gradient rather than a NaN."""
    loss = AntiCollapseLoss(variance_weight=1.0, covariance_weight=1.0)
    z = torch.randn(100, 8, requires_grad=True)
    with torch.no_grad():
        z[:, 2] = 0.0
    value, _ = loss(z)
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(z.grad).all()


def test_rejects_negative_weights():
    with pytest.raises(ValueError, match="non-negative"):
        AntiCollapseLoss(variance_weight=-1.0)


def test_single_sample_is_a_no_op_not_a_crash():
    loss = AntiCollapseLoss(variance_weight=1.0, covariance_weight=1.0)
    value, _ = loss(torch.randn(1, 8))
    assert float(value) == 0.0


# ── integration: zero weights are an exact control ───────────────────────
def _step(model, images, weights, seed=7):
    # The pseudo-anomaly term perturbs the descriptors with random noise, so two
    # forwards differ unless the RNG is pinned. Without this the control below
    # compares two different perturbations rather than two objectives.
    torch.manual_seed(seed)
    criterion = TotalLoss(
        patch_weight=1.0, use_normal_only=True, var_weight=0.1,
        use_pseudo=True, pseudo_epsilon=0.05, pseudo_margin=0.1,
        collapse_variance_weight=weights[0], collapse_covariance_weight=weights[1],
    )
    labels = torch.zeros(images.shape[0], model.num_patches)
    out = model(images, patch_labels=labels, update_stats=False, perturb_epsilon=0.05)
    losses = criterion(
        patch_scores=out["patch_scores"], patch_targets=labels,
        query_embeds=out["query_embeds"], labels=torch.zeros(images.shape[0]),
        patch_scores_perturbed=out["patch_scores_perturbed"],
        descriptors=out["descriptors"], local_features=out.get("local_features"),
    )
    return out, losses, criterion


@pytest.fixture
def model():
    torch.manual_seed(0)
    m = make_stub_spade(image_size=224)
    fit_statistics(m, torch.randn(2, 3, 224, 224))
    m.train()
    return m


def test_zero_weights_construct_nothing_and_change_nothing(model):
    images = torch.randn(2, 3, 224, 224)
    _, off, criterion = _step(model, images, (0.0, 0.0))
    assert criterion.collapse_loss_fn is None
    assert "collapse" not in off
    assert torch.allclose(off["total"], off["detection"])


def test_detection_term_is_untouched_by_the_regulariser(model):
    images = torch.randn(2, 3, 224, 224)
    _, off, _ = _step(model, images, (0.0, 0.0))
    _, on, _ = _step(model, images, (5.0, 0.2))
    assert torch.allclose(off["detection"], on["detection"], atol=1e-5)
    assert on["total"] > on["detection"]


def test_it_constrains_both_scored_representations(model):
    images = torch.randn(2, 3, 224, 224)
    _, losses, _ = _step(model, images, (5.0, 0.2))
    keys = losses["collapse_diagnostics"].keys()
    assert any("descriptor" in k for k in keys), "the Mahalanobis input"
    assert any("local" in k for k in keys), "the memory-bank input"


def test_gradients_reach_the_trainable_modules(model):
    images = torch.randn(2, 3, 224, 224)
    _, losses, _ = _step(model, images, (5.0, 0.2))
    model.zero_grad()
    losses["collapse"].backward()
    reached = {
        n.split(".")[0] for n, p in model.named_parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    }
    assert "fusion" in reached
    assert "contextualizer" in reached
