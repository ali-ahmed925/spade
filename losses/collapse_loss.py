"""Anti-collapse regularisation for the normal-only detection objective.

WHY THIS IS NEEDED
------------------
Measured on screw: over five epochs the 512-d descriptor space fell to an
effective rank of 2.3, and image AUROC fell with it, 0.866 -> 0.774.

The cause is not a bug in any single term. With statistics refit to match the
features, the Mahalanobis score satisfies

    E[s] = trace((Sigma_data + lambda I)^-1 Sigma_data)
         = sum_i  sigma_i / (sigma_i + lambda)
         ~= the number of eigenvalues above the ridge
         =  the EFFECTIVE RANK

measured directly: rank 512 -> loss 505, rank 64 -> 64, rank 8 -> 8, rank 2 ->
2. So "minimise the mean Mahalanobis distance of normal patches" literally means
"use fewer dimensions". And it is not only the mean term:

  * var(s) ~= 2 x effective rank, for the same reason;
  * the pseudo-anomaly margin gets EASIER under collapse -- once the features
    occupy 2 of 512 directions, Sigma^-1 is enormous in the 510 empty ones, so
    any perturbation scores hugely and the margin is satisfied for free.

Every term rewards collapse. That is not specific to this loss: ANY objective of
the form "make normal features close together" is minimised by discarding
dimensions. It is the degeneracy BYOL and SimSiam face, and every method in that
family ships an explicit anti-collapse mechanism. This one had none.

WHAT THIS IMPLEMENTS
--------------------
The VICReg mechanism (Bardes, Ponce & LeCun, ICLR 2022), which is the standard
and citable answer to exactly this failure:

    variance    (1/D) sum_j max(0, gamma - std(z_j))     each dimension must
                                                          keep spread
    covariance  (1/D) sum_{i != j} Cov(z)_ij^2            dimensions must stay
                                                          decorrelated

Together they make discarding a dimension cost something, so collapse stops
being free. Neither term looks at anomalies, labels, or test data -- they
constrain the representation only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AntiCollapseLoss(nn.Module):
    """VICReg variance + covariance terms on a batch of patch features."""

    def __init__(
        self,
        variance_weight: float = 1.0,
        covariance_weight: float = 0.04,
        target_std: float = 1.0,
        eps: float = 1e-4,
    ):
        """
        Args:
            variance_weight: weight on the per-dimension variance hinge.
            covariance_weight: weight on the decorrelation term. The 25:1 ratio
                between them is VICReg's; the absolute scale is set so both
                terms are comparable to the detection loss, which runs at ~2.
            target_std: the per-dimension standard deviation below which the
                hinge activates. The scored descriptor leaves a LayerNorm, so an
                isotropic representation has per-dimension std ~1 across the
                batch, which makes 1.0 the natural target rather than a tuned
                value.
            eps: added inside the square root, so a dimension that has already
                collapsed to exactly zero still produces a gradient rather than
                a NaN.
        """
        super().__init__()
        if variance_weight < 0 or covariance_weight < 0:
            raise ValueError("collapse weights must be non-negative")
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.target_std = target_std
        self.eps = eps

    @property
    def active(self) -> bool:
        return self.variance_weight > 0 or self.covariance_weight > 0

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            features: (B, N, D) or (M, D) patch features. Flattened to (M, D):
                the constraint is on the DIMENSIONS of the representation, so
                every patch is an independent sample of them.

        Returns:
            (weighted scalar loss, diagnostics). Diagnostics report the two
            terms unweighted, so their magnitudes can be compared against the
            detection loss without unpicking the weights.
        """
        z = features.reshape(-1, features.shape[-1])
        n_samples, dim = z.shape
        if n_samples < 2:
            zero = features.sum() * 0.0
            return zero, {"collapse/variance": 0.0, "collapse/covariance": 0.0}

        z = z.float()
        centered = z - z.mean(dim=0, keepdim=True)

        # ── variance: every dimension must keep spread ──
        std = torch.sqrt(centered.var(dim=0) + self.eps)
        variance_term = F.relu(self.target_std - std).mean()

        # ── covariance: dimensions must not duplicate one another ──
        # A rank-2 representation in 512 dimensions is 510 dimensions that are
        # exact linear functions of the other two, which shows up here as large
        # off-diagonal covariance.
        covariance = (centered.T @ centered) / (n_samples - 1)
        off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
        covariance_term = off_diagonal.pow(2).sum() / dim

        loss = (
            self.variance_weight * variance_term
            + self.covariance_weight * covariance_term
        )
        return loss, {
            "collapse/variance": float(variance_term.detach()),
            "collapse/covariance": float(covariance_term.detach()),
        }

    def extra_repr(self) -> str:
        return (
            f"variance_weight={self.variance_weight}, "
            f"covariance_weight={self.covariance_weight}, "
            f"target_std={self.target_std}"
        )


@torch.no_grad()
def effective_rank(features: torch.Tensor) -> float:
    """Participation ratio of the covariance spectrum, (sum L)^2 / sum L^2.

    Equals D for an isotropic representation and falls toward 1 as variance
    concentrates. The same quantity `models/normal_fit.feature_geometry` reports,
    duplicated here so tests can assert on it without a full fit.
    """
    z = features.reshape(-1, features.shape[-1]).float()
    centered = z - z.mean(dim=0, keepdim=True)
    eigenvalues = torch.linalg.svdvals(centered).pow(2) / max(z.shape[0] - 1, 1)
    total = eigenvalues.sum().clamp_min(1e-12)
    return float(total.pow(2) / eigenvalues.pow(2).sum().clamp_min(1e-12))
