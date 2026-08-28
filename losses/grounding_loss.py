"""Auxiliary loss that makes specific query tokens attend to anomalous regions.

WHY
---
D1 measured that the Q-Former queries do not localize defects: on wood, mean
per-query pixel AUROC was 0.5093 (chance), the best of 32 reached 0.6309, and
max-over-queries saliency was 0.5765. The redesign made the queries load-bearing
for DETECTION — they shape the descriptor Mahalanobis scores — but nothing in
the objective asks them to point AT anything, so any language claim about
"where" the defect is would currently be unfounded.

This supplies that signal, using synthetic CutPaste/Crack anomalies purely as a
source of patch-level masks. The detection objective is untouched: the anomaly
score never sees a synthetic label.

FORM
----
The contextualizer computes MultiheadAttention(query=patches, key/value=queries),
so `attention` is (B, N_patches, N_queries) and is a SOFTMAX OVER QUERIES: each
patch's 32 weights sum to 1. That rules out the obvious formulation — supervising
max-over-queries toward {0, 1} is ill-posed, since the max is bounded below by
1/32 and above by 1 regardless of anomaly.

What IS well-posed given that structure: reserve a few queries as "anomaly
queries" and supervise the attention MASS they receive.

    p_anom(patch) = sum over the reserved queries of attention[patch, q]   in [0, 1]
    loss          = BCE(p_anom, patch_is_anomalous)

A patch routes its (fixed) attention budget either to the anomaly queries or to
the rest, so the loss is a genuine competition and cannot be satisfied by
inflating everything. It also produces exactly what goal 3 needs: a small set of
tokens that mean "defect", whose attention map IS the localization, rather than
a diffuse signal spread over all 32.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QueryGroundingLoss(nn.Module):
    """BCE on the attention mass received by the reserved anomaly queries."""

    def __init__(
        self,
        n_anomaly_queries: int = 4,
        pos_weight: float | str = "auto",
        max_pos_weight: float = 50.0,
        eps: float = 1e-6,
    ):
        """
        Args:
            n_anomaly_queries: how many leading query tokens are reserved for
                anomalies. Taking the first k is arbitrary but deterministic,
                which matters for interpreting a trained checkpoint.
            pos_weight: weight on the positive (anomalous) term. Anomalous
                patches are a small minority of a batch, so unweighted BCE is
                minimised by predicting "normal" everywhere. "auto" computes
                n_negative / n_positive per batch.
            max_pos_weight: cap on the automatic weight; a batch with two
                anomalous patches would otherwise produce a weight in the
                hundreds and destabilise training.
        """
        super().__init__()
        if n_anomaly_queries < 1:
            raise ValueError(f"n_anomaly_queries must be >= 1, got {n_anomaly_queries}")
        self.n_anomaly_queries = n_anomaly_queries
        self.pos_weight = pos_weight
        self.max_pos_weight = max_pos_weight
        self.eps = eps

    def anomaly_mass(self, attention: torch.Tensor) -> torch.Tensor:
        """(B, N, Q) attention -> (B, N) mass on the reserved anomaly queries."""
        if attention.shape[-1] < self.n_anomaly_queries:
            raise ValueError(
                f"n_anomaly_queries={self.n_anomaly_queries} exceeds the "
                f"{attention.shape[-1]} queries the model provides"
            )
        return attention[..., : self.n_anomaly_queries].sum(dim=-1)

    def forward(
        self,
        attention: torch.Tensor,
        patch_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            attention: (B, N, Q) patch-to-query attention, softmax over Q.
            patch_labels: (B, N) 1 for anomalous patches (from the synthetic
                mask), 0 for normal.

        Returns:
            (scalar loss, diagnostics dict). The diagnostics report the mean
            anomaly-query mass on each class, which is what actually has to
            separate for the queries to be groundable.
        """
        p = self.anomaly_mass(attention).clamp(self.eps, 1.0 - self.eps)
        target = patch_labels.to(p.dtype)

        n_pos = float(target.sum())
        n_neg = float(target.numel() - n_pos)

        if self.pos_weight == "auto":
            weight = min(n_neg / max(n_pos, 1.0), self.max_pos_weight) if n_pos > 0 else 1.0
        else:
            weight = float(self.pos_weight)

        pos_term = -weight * target * torch.log(p)
        neg_term = -(1.0 - target) * torch.log(1.0 - p)
        loss = (pos_term + neg_term).mean()

        with torch.no_grad():
            diagnostics = {
                "grounding/pos_weight": float(weight),
                "grounding/anomalous_patch_fraction": n_pos / max(target.numel(), 1),
                "grounding/mass_on_normal": float(p[target == 0].mean()) if n_neg > 0 else float("nan"),
                "grounding/mass_on_anomalous": float(p[target == 1].mean()) if n_pos > 0 else float("nan"),
            }
        return loss, diagnostics
