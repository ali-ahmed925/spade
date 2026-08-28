"""Checkpoint selection policy.

Deliberately free of torch and of any model state: choosing WHICH epoch to keep
is a decision about two numbers, and keeping it that way means the rule can be
replayed against a real run's metric sequence in a unit test.
"""

from __future__ import annotations


def should_save_checkpoint(
    primary: float,
    secondary: float,
    best_primary: float,
    best_secondary: float,
    min_delta: float = 0.0,
    tie_tol: float = 1e-6,
) -> tuple[bool, bool]:
    """Lexicographic checkpoint selection on (primary, secondary).

    Two failure modes this exists to prevent, both observed:

    1. A ratchet that slides DOWN. The original rule saved whenever the
       secondary improved and the primary degraded "within tolerance", then
       reassigned the reference to the degraded value, so a run drifted
       0.9125 -> 0.8565 on image AUROC while logging "new best" every epoch.
       Here `best_primary` is only ever raised by the caller.

    2. A ratchet that DEADLOCKS. Selecting on the primary alone freezes the
       checkpoint the moment that metric saturates: wood hits val image AUROC
       1.0000 at epoch 1, nothing can exceed it, and 19 further epochs of
       training are silently discarded. Ties are therefore broken by the
       secondary metric, which is not saturated.

    Returns:
        (save, improved) — `improved` distinguishes a strict primary gain from
        a tie broken on the secondary, which the caller needs because a strict
        gain RESETS the secondary bar (a new primary level starts a fresh race)
        while a tie-break only raises it.
    """
    if primary != primary:                      # NaN — nothing to select on
        return False, False
    if secondary != secondary:                  # NaN cannot win a tie-break
        secondary = -float("inf")

    improved = primary > best_primary + min_delta
    tied = (not improved) and primary >= best_primary - tie_tol
    broke_tie = tied and secondary > best_secondary + min_delta
    return improved or broke_tie, improved
