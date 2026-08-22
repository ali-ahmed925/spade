"""Gradient / dead-path auditing utilities.

Purpose: prove — rather than assume — that every module we call "trainable"
actually receives a non-zero gradient and actually influences the model output.

Three independent checks are provided, because each catches a different failure:

1. `classify_parameters`      — after backward(), which params got no gradient?
                                Catches modules that are disconnected from the loss.
2. `output_sensitivity`       — does perturbing a module's weights change the
                                output? Catches modules whose output is computed
                                but then discarded (grad can be non-None yet the
                                module still not matter if it is scaled to ~0).
3. `component_contributions`  — relative magnitude of each additive score term.
                                Catches terms that are mathematically present but
                                numerically irrelevant.

Used by tests/test_gradient_flow.py and by train.py's startup self-check.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Parameter status codes
FROZEN = "FROZEN"      # requires_grad=False — intentional, not an error
DEAD = "DEAD"          # requires_grad=True but grad is None: not in the loss graph
ZERO = "ZERO"          # grad exists but is exactly zero: no learning signal
TINY = "TINY"          # grad non-zero but below `tiny_threshold`
OK = "OK"              # grad present and meaningful


def classify_parameters(
    model: nn.Module,
    tiny_threshold: float = 1e-12,
) -> dict[str, dict]:
    """Classify every parameter by the gradient it received.

    Call AFTER loss.backward() and BEFORE optimizer.zero_grad().

    Returns:
        {param_name: {"status": str, "grad_norm": float, "requires_grad": bool}}
    """
    report: dict[str, dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            status, gnorm = FROZEN, 0.0
        elif p.grad is None:
            status, gnorm = DEAD, 0.0
        else:
            gnorm = float(p.grad.detach().float().norm())
            if gnorm == 0.0:
                status = ZERO
            elif gnorm < tiny_threshold:
                status = TINY
            else:
                status = OK
        report[name] = {
            "status": status,
            "grad_norm": gnorm,
            "requires_grad": bool(p.requires_grad),
        }
    return report


def summarize_by_module(param_report: dict[str, dict]) -> dict[str, dict]:
    """Aggregate a parameter report by top-level module name."""
    modules: dict[str, dict] = {}
    for name, info in param_report.items():
        top = name.split(".")[0]
        m = modules.setdefault(
            top, {"n_params": 0, "statuses": {}, "max_grad_norm": 0.0}
        )
        m["n_params"] += 1
        m["statuses"][info["status"]] = m["statuses"].get(info["status"], 0) + 1
        m["max_grad_norm"] = max(m["max_grad_norm"], info["grad_norm"])
    return modules


def dead_parameters(param_report: dict[str, dict]) -> list[str]:
    """Names of parameters that claim to be trainable but receive no signal."""
    return [
        n for n, i in param_report.items() if i["status"] in (DEAD, ZERO)
    ]


@torch.no_grad()
def output_sensitivity(
    model: nn.Module,
    forward_fn,
    module_names: list[str] | None = None,
    epsilon: float = 1e-2,
    seed: int = 0,
) -> dict[str, float]:
    """Measure how much the output moves when each module's weights are perturbed.

    A module can receive a non-None gradient and still be irrelevant if its
    contribution is scaled to ~0 downstream. This catches that case: it reports
    the relative change in the output when the module's parameters are nudged.

    Args:
        model: the model under test.
        forward_fn: zero-arg callable returning the output tensor to watch.
        module_names: top-level module names to test (default: all with params).
        epsilon: relative perturbation size.

    Returns:
        {module_name: relative_l1_change_in_output}
    """
    baseline = forward_fn().detach().clone()
    denom = baseline.abs().mean().clamp_min(1e-12)

    if module_names is None:
        module_names = sorted(
            {n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad}
        )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    results: dict[str, float] = {}

    for mod_name in module_names:
        saved = {}
        for name, p in model.named_parameters():
            if name.split(".")[0] != mod_name or not p.requires_grad:
                continue
            saved[name] = p.detach().clone()
            noise = torch.randn(
                p.shape, generator=generator, dtype=torch.float32
            ).to(p.device, p.dtype)
            p.add_(noise * epsilon * p.detach().abs().mean().clamp_min(1e-8))

        if not saved:
            results[mod_name] = 0.0
            continue

        perturbed = forward_fn().detach()
        results[mod_name] = float(((perturbed - baseline).abs().mean()) / denom)

        # restore
        params = dict(model.named_parameters())
        for name, value in saved.items():
            params[name].copy_(value)

    return results


def component_contributions(components: dict[str, torch.Tensor]) -> dict[str, float]:
    """Relative magnitude of each additive term in a score.

    Args:
        components: {term_name: tensor} — the *weighted* terms as they are summed.

    Returns:
        {term_name: fraction_of_total_absolute_magnitude}
    """
    mags = {
        k: float(v.detach().abs().mean()) for k, v in components.items()
    }
    total = sum(mags.values()) or 1e-12
    return {k: v / total for k, v in mags.items()}


def format_report(
    param_report: dict[str, dict],
    sensitivity: dict[str, float] | None = None,
    contributions: dict[str, float] | None = None,
) -> str:
    """Render a human-readable audit report."""
    lines = []
    lines.append("=" * 72)
    lines.append("GRADIENT AUDIT")
    lines.append("=" * 72)

    modules = summarize_by_module(param_report)
    lines.append(f"{'module':<28} {'params':>7}  {'max|grad|':>12}  statuses")
    lines.append("-" * 72)
    for name in sorted(modules):
        m = modules[name]
        statuses = ", ".join(f"{k}:{v}" for k, v in sorted(m["statuses"].items()))
        lines.append(
            f"{name:<28} {m['n_params']:>7}  {m['max_grad_norm']:>12.3e}  {statuses}"
        )

    dead = dead_parameters(param_report)
    lines.append("")
    if dead:
        lines.append(f"!! {len(dead)} parameter tensor(s) trainable but receiving NO signal:")
        for n in dead[:10]:
            lines.append(f"   - {n} ({param_report[n]['status']})")
        if len(dead) > 10:
            lines.append(f"   ... and {len(dead) - 10} more")
    else:
        lines.append("OK: every trainable parameter received a non-zero gradient.")

    if sensitivity is not None:
        lines.append("")
        lines.append("OUTPUT SENSITIVITY (relative output change when weights perturbed)")
        lines.append("-" * 72)
        for name, val in sorted(sensitivity.items(), key=lambda kv: -kv[1]):
            flag = "  <-- NO EFFECT ON OUTPUT" if val < 1e-9 else ""
            lines.append(f"{name:<28} {val:>12.3e}{flag}")

    if contributions is not None:
        lines.append("")
        lines.append("SCORE COMPOSITION (share of total magnitude)")
        lines.append("-" * 72)
        for name, val in sorted(contributions.items(), key=lambda kv: -kv[1]):
            flag = "  <-- NEGLIGIBLE" if val < 1e-4 else ""
            lines.append(f"{name:<28} {val:>11.6%}{flag}")

    lines.append("=" * 72)
    return "\n".join(lines)
