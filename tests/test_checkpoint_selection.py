"""Checkpoint selection must neither slide down nor deadlock.

Both failure modes were observed in real runs and each silently discards
training while the log claims progress, so both are pinned here.
"""

import math

from utils.selection import should_save_checkpoint


def replay(epochs, min_delta=0.0, tie_tol=1e-6):
    """Run the real selection rule over a sequence of (primary, secondary)."""
    best_p, best_s, saved = -math.inf, -math.inf, []
    for epoch, (p, s) in enumerate(epochs, start=1):
        save, improved = should_save_checkpoint(p, s, best_p, best_s, min_delta, tie_tol)
        if save:
            best_p = max(best_p, p)
            best_s = s if improved else max(best_s, s)
            saved.append((epoch, p, s))
    return saved, best_p


# ── deadlock: the failure that made the lambda sweep meaningless ─────────────
def test_saturated_primary_does_not_freeze_selection():
    """wood reaches val image AUROC 1.0000 at epoch 1.

    Selecting on the primary alone keeps the epoch-1 weights forever, so 19
    further epochs train into a checkpoint nobody ever loads.
    """
    epochs = [(1.0, 0.7329), (1.0, 0.7510), (1.0, 0.7480), (1.0, 0.8102)]
    saved, _ = replay(epochs)
    assert [e for e, _, _ in saved] == [1, 2, 4]
    assert saved[-1] == (4, 1.0, 0.8102), "the best secondary at the saturated primary must win"


def test_saturated_primary_ignores_secondary_regressions():
    saved, _ = replay([(1.0, 0.80), (1.0, 0.60), (1.0, 0.70)])
    assert [e for e, _, _ in saved] == [1]


# ── ratchet: the earlier bug, which must stay fixed ──────────────────────────
def test_primary_bar_never_slides_downward():
    """The original rule saved every epoch here, drifting 0.9125 -> 0.8565."""
    epochs = [(0.8100, 0.70), (0.9125, 0.71), (0.8800, 0.90), (0.8565, 0.95)]
    saved, best = replay(epochs)
    assert [e for e, _, _ in saved] == [1, 2]
    assert best == 0.9125
    assert saved[-1][1] == 0.9125, "a worse primary must not be saved for a better secondary"


def test_strict_improvement_resets_the_secondary_bar():
    """A higher primary level starts a fresh race; a low secondary must not
    be blocked by a high secondary recorded at a lower primary level."""
    saved, _ = replay([(0.90, 0.99), (0.95, 0.10), (0.95, 0.20)])
    assert [e for e, _, _ in saved] == [1, 2, 3]


# ── numerical and NaN edges ──────────────────────────────────────────────────
def test_float_noise_counts_as_a_tie_not_a_regression():
    save, improved = should_save_checkpoint(1.0 - 1e-12, 0.9, 1.0, 0.5)
    assert save and not improved


def test_min_delta_suppresses_trivial_gains():
    assert should_save_checkpoint(0.9001, 0.5, 0.9000, 0.9, min_delta=0.01)[0] is False


def test_nan_primary_never_saves():
    assert should_save_checkpoint(float("nan"), 0.9, 0.5, 0.5) == (False, False)


def test_nan_secondary_cannot_win_a_tie_but_does_not_block_a_gain():
    assert should_save_checkpoint(0.9, float("nan"), 0.9, 0.5)[0] is False
    assert should_save_checkpoint(0.95, float("nan"), 0.9, 0.5)[0] is True
