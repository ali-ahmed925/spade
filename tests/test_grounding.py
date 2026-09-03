"""Tests for the auxiliary query-grounding objective.

The contract this must satisfy, because the whole point is to gain grounding
WITHOUT paying for it in detection:

  * lambda = 0 is the exact control — no attention computed, no synthetic
    anomalies, total loss identical to the detection loss alone;
  * the detection term is byte-identical whether grounding is on or off;
  * gradients reach the shared Q-Former, not just a bolted-on head, so the
    query tokens genuinely learn;
  * synthetic-anomalous patches are excluded from the detection loss and from
    the normal statistics.
"""

import pytest
import torch

from losses.grounding_loss import QueryGroundingLoss
from losses.patch_loss_normal import MahalanobisPatchLoss, PseudoAnomalyLoss
from losses.total_loss import TotalLoss
from tests.stub_blip2 import fit_statistics, make_stub_spade

BATCH, N_PATCHES, N_QUERIES = 2, 256, 4


@pytest.fixture
def images():
    torch.manual_seed(0)
    return torch.randn(BATCH, 3, 224, 224)


@pytest.fixture
def model(images):
    m = make_stub_spade()
    fit_statistics(m, images)
    m.train()
    return m


@pytest.fixture
def labels():
    """A contiguous block of synthetic-anomalous patches in the first image."""
    lab = torch.zeros(BATCH, N_PATCHES)
    lab[0, 100:140] = 1.0
    return lab


def _step(model, images, labels, lmbda):
    criterion = TotalLoss(
        patch_weight=1.0, use_normal_only=True, var_weight=0.1,
        use_pseudo=True, pseudo_epsilon=0.05, pseudo_margin=0.1,
        grounding_weight=lmbda, grounding_queries=2,
    )
    out = model(
        images, patch_labels=labels, update_stats=False, perturb_epsilon=0.05,
        return_attention=criterion.grounding_loss_fn is not None,
    )
    losses = criterion(
        patch_scores=out["patch_scores"], patch_targets=labels,
        query_embeds=out["query_embeds"], labels=torch.zeros(BATCH),
        patch_scores_perturbed=out["patch_scores_perturbed"],
        patch_query_attention=out.get("patch_query_attention"),
    )
    return out, losses, criterion


# ── lambda = 0 is the exact control ──────────────────────────────────────────
def test_lambda_zero_disables_grounding_entirely(model, images, labels):
    out, losses, criterion = _step(model, images, labels, 0.0)
    assert criterion.grounding_loss_fn is None
    assert "grounding" not in losses
    assert "patch_query_attention" not in out, "attention should not even be computed"
    assert torch.allclose(losses["total"], losses["detection"])


def test_grounding_does_not_change_the_detection_term(model, images, labels):
    _, off, _ = _step(model, images, labels, 0.0)
    _, on, _ = _step(model, images, labels, 0.5)
    assert torch.allclose(off["detection"], on["detection"], atol=1e-5), (
        "the detection objective must be untouched by the auxiliary loss"
    )
    assert on["total"] > on["detection"], "grounding should add to the total"


def test_grounding_requires_attention(model, images, labels):
    criterion = TotalLoss(use_normal_only=True, grounding_weight=0.1)
    with pytest.raises(ValueError, match="return_attention"):
        criterion(
            patch_scores=torch.rand(BATCH, N_PATCHES),
            patch_targets=labels,
            query_embeds=torch.rand(BATCH, N_QUERIES, 8),
            labels=torch.zeros(BATCH),
            patch_query_attention=None,
        )


# ── gradients must reach the shared Q-Former ─────────────────────────────────
def test_grounding_gradients_reach_the_qformer(model, images, labels):
    _, losses, _ = _step(model, images, labels, 0.1)
    model.zero_grad()
    losses["grounding"].backward()
    reached = {
        n.split(".")[0] for n, p in model.named_parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    }
    assert "qformer" in reached, (
        "the query tokens must learn from this loss — otherwise it is a "
        f"bolted-on head, not grounding. reached: {sorted(reached)}"
    )
    assert "contextualizer" in reached


# ── the loss must respect the softmax-over-queries structure ─────────────────
def test_attention_is_a_distribution_over_queries(model, images, labels):
    out, _, _ = _step(model, images, labels, 0.1)
    attn = out["patch_query_attention"]
    assert torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)), atol=1e-4)


def test_anomaly_mass_is_the_reserved_queries_share():
    loss = QueryGroundingLoss(n_anomaly_queries=2)
    attn = torch.rand(2, 16, 8)
    attn = attn / attn.sum(dim=-1, keepdim=True)
    mass = loss.anomaly_mass(attn)
    assert torch.allclose(mass, attn[..., :2].sum(-1))
    assert bool((mass >= 0).all() and (mass <= 1).all())


def test_rejects_more_anomaly_queries_than_exist():
    loss = QueryGroundingLoss(n_anomaly_queries=99)
    with pytest.raises(ValueError, match="exceeds"):
        loss(torch.rand(1, 4, 8).softmax(-1), torch.zeros(1, 4))


def test_pos_weight_survives_a_batch_with_no_anomalies():
    loss = QueryGroundingLoss(n_anomaly_queries=2)
    attn = torch.rand(2, 16, 8).softmax(-1)
    value, diag = loss(attn, torch.zeros(2, 16))
    assert torch.isfinite(value)
    assert diag["grounding/pos_weight"] == 1.0


def test_loss_decreases_when_mass_moves_onto_anomalies():
    """Sanity: the objective must reward the behaviour we want."""
    loss = QueryGroundingLoss(n_anomaly_queries=2, pos_weight=1.0)
    target = torch.zeros(1, 4)
    target[0, :2] = 1.0

    wrong = torch.tensor([[[0.05, 0.05, 0.45, 0.45]] * 4])           # mass away from anomaly queries
    right = torch.tensor([[[0.45, 0.45, 0.05, 0.05],
                           [0.45, 0.45, 0.05, 0.05],
                           [0.05, 0.05, 0.45, 0.45],
                           [0.05, 0.05, 0.45, 0.45]]])               # mass on anomalous patches only
    assert float(loss(right, target)[0]) < float(loss(wrong, target)[0])


# ── synthetic patches excluded from the detection objective ──────────────────
def test_detection_loss_ignores_synthetic_anomalous_patches():
    scores = torch.ones(1, 10)
    scores[0, 5:] = 100.0                       # "anomalous" patches score high
    lab = torch.zeros(1, 10)
    lab[0, 5:] = 1.0

    fn = MahalanobisPatchLoss(clamp_max=1000.0, lambda_var=0.0)
    masked = float(fn(scores, lab))
    normal_only = float(fn(scores[:, :5], torch.zeros(1, 5)))
    assert masked == pytest.approx(normal_only, abs=1e-6), (
        "minimizing the score of a deliberately anomalous patch would teach the "
        "model that the anomaly is normal"
    )


def test_detection_loss_unchanged_when_all_patches_normal():
    """The exclusion must be a no-op at lambda=0, or the control is not exact."""
    scores = torch.rand(2, 32) * 10
    fn = MahalanobisPatchLoss()
    assert float(fn(scores, torch.zeros(2, 32))) == pytest.approx(float(fn(scores, None)))


def test_pseudo_loss_also_excludes_anomalous_patches():
    clean = torch.zeros(1, 10)
    perturbed = torch.zeros(1, 10)
    perturbed[0, :5] = 1.0                      # normal patches satisfy the margin
    lab = torch.zeros(1, 10)
    lab[0, 5:] = 1.0

    fn = PseudoAnomalyLoss(margin=0.5)
    assert float(fn(clean, perturbed, lab)) == pytest.approx(0.0, abs=1e-6)
    assert float(fn(clean, perturbed, None)) > 0.0


def test_statistics_still_exclude_anomalous_patches(model, images, labels):
    """Normal-only Mahalanobis statistics must not absorb synthetic defects."""
    stats = model.normal_stats
    stats.reset()
    descriptors = torch.randn(BATCH, N_PATCHES, model.descriptor_dim)
    accumulated = stats.update(descriptors, labels)
    expected = int((labels == 0).sum())
    assert accumulated == expected
    assert int(stats.count) == expected
