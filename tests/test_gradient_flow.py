"""Gradient-flow and dead-path regression tests.

These tests exist because a whole training pipeline ran for months in which:
  * the LLM projection never received a gradient,
  * the "contrastive" pseudo-anomaly loss had an algebraically zero gradient,
  * the only gradient-bearing score term was ~1e-5 of the score, and
  * the attention importance had a structurally constant patch-mean.

The feature-space redesign removed that additive attention term entirely — the
Q-Former now enters through the descriptor Mahalanobis scores — but the
invariants below still hold and still guard the same failure modes.

None of these raised an error. Each one below is pinned by a test so it cannot
come back silently.

They run on a tiny stub backbone (tests/stub_blip2.py) — no BLIP-2 download.
"""

import pytest
import torch

from losses.patch_loss_normal import PseudoAnomalyLoss
from losses.total_loss import TotalLoss
from tests.stub_blip2 import fit_statistics, make_stub_spade
from utils.grad_audit import (
    classify_parameters,
    component_contributions,
    dead_parameters,
    output_sensitivity,
)

BATCH = 2
N_PATCHES = 256   # 224px stub -> 16x16 grid


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


def _train_step(model, images, use_pseudo=True):
    """One forward+backward, returning (outputs, losses)."""
    criterion = TotalLoss(
        patch_weight=1.0,
        use_normal_only=True,
        var_weight=0.1,
        use_pseudo=use_pseudo,
        pseudo_epsilon=0.05,
        pseudo_margin=0.1,
    )
    eps = criterion.pseudo_loss_fn.epsilon if criterion.pseudo_loss_fn else None
    outputs = model(
        images,
        patch_labels=torch.zeros(BATCH, N_PATCHES),
        update_stats=True,
        perturb_epsilon=eps,
    )
    losses = criterion(
        patch_scores=outputs["patch_scores"],
        patch_targets=torch.zeros(BATCH, N_PATCHES),
        query_embeds=outputs["query_embeds"],
        labels=torch.zeros(BATCH),
        patch_scores_perturbed=outputs.get("patch_scores_perturbed"),
    )
    losses["total"].backward()
    return outputs, losses


# ──────────────────────────────────────────────────────────────────────────
# 1. No trainable parameter may be disconnected from the loss
# ──────────────────────────────────────────────────────────────────────────
def test_no_trainable_parameter_is_dead(model, images):
    _train_step(model, images)
    report = classify_parameters(model)
    dead = dead_parameters(report)
    assert not dead, (
        "these parameters require grad but received none — either connect them "
        f"to the loss or freeze them: {dead}"
    )


def test_projection_is_frozen_while_no_text_loss_supervises_it(model):
    """The projection has no loss. It must be frozen, not fake-trainable."""
    assert all(not p.requires_grad for p in model.projection.parameters())
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert not any(n.startswith("projection.") for n in trainable)


def test_projection_can_be_unfrozen_explicitly():
    m = make_stub_spade(projection_trainable=True)
    assert all(p.requires_grad for p in m.projection.parameters())


# ──────────────────────────────────────────────────────────────────────────
# 2. The trainable modules must actually move the output
# ──────────────────────────────────────────────────────────────────────────
def test_trainable_modules_affect_the_output(model, images):
    model.eval()
    sens = output_sensitivity(model, lambda: model(images)["patch_scores"])
    for name, value in sens.items():
        assert value > 1e-6, (
            f"perturbing {name!r} changes the output by only {value:.2e} — it is "
            "trainable in name only"
        )


def test_stream_normalization_makes_weights_meaningful(model, images):
    """alpha/beta/gamma should map to comparable shares of the score."""
    model.train()
    # prime the scale buffers
    for _ in range(3):
        model(images, patch_labels=torch.zeros(BATCH, N_PATCHES), update_stats=True)
    model.eval()
    out = model(images)
    shares = component_contributions(out["score_components"])
    assert shares["contextual_mahalanobis"] > 1e-3, (
        f"the contextual Mahalanobis term is {shares['contextual_mahalanobis']:.2e} "
        "of the score — the configured weight is not what is actually applied"
    )


# ──────────────────────────────────────────────────────────────────────────
# 3. The pseudo-anomaly loss must have a real gradient
# ──────────────────────────────────────────────────────────────────────────
def test_pseudo_loss_requires_separately_scored_perturbation():
    loss = PseudoAnomalyLoss(epsilon=0.01)
    with pytest.raises(ValueError, match="perturbed_scores"):
        loss(torch.randn(2, 8), None)


def test_pseudo_loss_gradient_is_nonzero():
    """The old implementation cancelled algebraically and had exactly 0 grad."""
    clean = torch.randn(2, 16, requires_grad=True)
    perturbed = (clean.detach() + 0.01).requires_grad_(True)
    PseudoAnomalyLoss(epsilon=0.01, margin=0.5)(clean, perturbed).backward()
    assert clean.grad is not None and clean.grad.abs().sum() > 0
    assert perturbed.grad is not None and perturbed.grad.abs().sum() > 0


def test_old_pseudo_loss_formulation_would_have_zero_gradient():
    """Pins the exact bug: perturbing the SCORES cancels."""
    scores = torch.randn(2, 16, requires_grad=True)
    old_style = torch.relu((scores + 0.01 * torch.randn_like(scores)) - scores).mean()
    old_style.backward()
    assert scores.grad is not None
    assert scores.grad.abs().max() == 0.0, (
        "score-level perturbation must cancel — if this ever fails the bug "
        "description in PseudoAnomalyLoss is wrong"
    )


def test_perturbed_patches_score_differently_from_clean(model, images):
    out = model(
        images,
        patch_labels=torch.zeros(BATCH, N_PATCHES),
        update_stats=True,
        perturb_epsilon=0.5,
    )
    assert "patch_scores_perturbed" in out
    assert not torch.allclose(out["patch_scores"], out["patch_scores_perturbed"])


# ──────────────────────────────────────────────────────────────────────────
# 5. Optimizer / checkpoint integrity
# ──────────────────────────────────────────────────────────────────────────
def test_optimizer_only_receives_parameters_that_get_gradients(model, images):
    from optim.optimizer import build_optimizer

    optimizer = build_optimizer(model, lr=1e-4)
    _train_step(model, images)
    optimized = {id(p) for g in optimizer.param_groups for p in g["params"]}
    for name, p in model.named_parameters():
        if id(p) in optimized:
            assert p.grad is not None, f"{name} is being optimized but has no gradient"


def test_checkpoint_keeps_stream_scale_buffers(model, images):
    """The old allowlist-based save silently dropped newly added state."""
    _train_step(model, images)
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("vision_encoder.")}
    for key in ("mahal_scale", "freq_scale", "stream_scales_initialized"):
        assert key in state, f"{key} would not survive a save/load round trip"


def test_training_actually_changes_parameters(model, images):
    """End-to-end: a step must move the weights that claim to be learning."""
    from optim.optimizer import build_optimizer

    optimizer = build_optimizer(model, lr=1e-2)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    _train_step(model, images)
    optimizer.step()
    moved = [
        n for n, p in model.named_parameters()
        if p.requires_grad and not torch.allclose(before[n], p.detach())
    ]
    assert moved, "no trainable parameter moved after an optimizer step"
