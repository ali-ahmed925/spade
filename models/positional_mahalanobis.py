"""Position-conditioned Mahalanobis scoring.

WHY
---
`MahalanobisScoring` fits ONE Gaussian over every patch of every training image
pooled together: `mu` has shape (D,). Position is discarded by
`NormalStatisticsTracker.add_normal_patches`, which masks with
`patch_embeds[normal_mask]` and flattens.

That makes a whole class of defects invisible by construction. When a transistor
is *misplaced*, every individual patch still looks like a legitimate transistor
patch — copper, plastic, board — and only the arrangement is wrong. A
position-agnostic model cannot see it. Same for cable_swap (right wires, wrong
positions) and zipper misalignment.

It also creates false positives in the other direction: a patch that is unusual
*globally* but perfectly normal *for its location* scores high, which inflates
the scores of clean images. That matches the measured failure — cable localizes
well (pixel 0.935) yet fails to detect (image 0.875).

WHAT
----
PaDiM fits one Gaussian per patch position. With ~250 training images per
category, a full per-position covariance (1408x1408 from 250 samples) is wildly
under-determined, so this module uses the sample-efficient form:

    per-position means mu_p  +  ONE shared, pooled within-position covariance

The shared covariance is estimated from M*N samples rather than M, so it stays
well-conditioned, while the means carry the positional information.

CAVEAT
------
Position-conditioning assumes the object is aligned across images. That holds
for transistor, cable, zipper, toothbrush. It does NOT hold for screw, whose
images are arbitrarily rotated — there, positional means should help less or
hurt. `blend` interpolates toward the global mean so this is a measurable knob
rather than an assumption.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PositionalMahalanobisScoring(nn.Module):
    """Mahalanobis scoring with per-position means and a shared covariance.

    Drop-in replacement for `MahalanobisScoring`: same forward signature, same
    `is_initialized` flag, so it can be swapped into `SPADE.mahalanobis_scorer`.
    """

    def __init__(
        self,
        feature_dim: int,
        num_positions: int,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_positions = num_positions
        self.gamma = gamma

        self.register_buffer("mu", torch.zeros(num_positions, feature_dim))
        self.register_buffer("sigma_inv", torch.eye(feature_dim))
        self.register_buffer("is_initialized", torch.tensor(False))

    def forward(
        self,
        patch_embeds: torch.Tensor,
        apply_amplification: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            patch_embeds: (B, N, D) patch embeddings, N in the same raster order
                the statistics were fitted with.
            apply_amplification: apply the gamma power, matching MahalanobisScoring.

        Returns:
            (B, N) squared Mahalanobis distances to the per-position mean.
        """
        if not self.is_initialized:
            return torch.zeros(
                patch_embeds.shape[0], patch_embeds.shape[1],
                device=patch_embeds.device, dtype=patch_embeds.dtype,
            )

        B, N, D = patch_embeds.shape
        if N != self.num_positions:
            raise ValueError(
                f"got {N} patches but statistics were fitted for "
                f"{self.num_positions} positions — image_size/patch_size mismatch"
            )

        centered = patch_embeds - self.mu.unsqueeze(0)      # (B, N, D)
        flat = centered.reshape(B * N, D)
        quad = (torch.mm(flat, self.sigma_inv) * flat).sum(dim=1)
        quad = torch.clamp(quad, min=0.0).view(B, N)

        if apply_amplification:
            quad = torch.pow(quad + 1e-8, self.gamma)
        return quad


class PositionalStatsAccumulator:
    """One-pass accumulator for per-position means and a pooled covariance.

    Streams over the training set without holding every patch in memory. Uses
    the identity

        S_within = sum_i sum_p x_ip x_ip^T  -  M * sum_p mu_p mu_p^T

    so both the means and the pooled within-position scatter come from a single
    pass. Accumulates in float64: the sums run to ~1e10 and float32 loses the
    small differences that the covariance is made of.
    """

    def __init__(self, feature_dim: int, num_positions: int, device: str | torch.device = "cpu"):
        self.feature_dim = feature_dim
        self.num_positions = num_positions
        self.device = torch.device(device)
        self.sum_x = torch.zeros(num_positions, feature_dim, dtype=torch.float64, device=self.device)
        self.sum_xx = torch.zeros(feature_dim, feature_dim, dtype=torch.float64, device=self.device)
        self.n_images = 0

    @torch.no_grad()
    def update(self, patch_embeds: torch.Tensor) -> None:
        """Accumulate a batch of (B, N, D) normal patch embeddings."""
        x = patch_embeds.to(self.device, torch.float64)
        B, N, D = x.shape
        if N != self.num_positions or D != self.feature_dim:
            raise ValueError(
                f"expected (B, {self.num_positions}, {self.feature_dim}), got {tuple(x.shape)}"
            )
        self.sum_x += x.sum(dim=0)
        flat = x.reshape(B * N, D)
        self.sum_xx += flat.T @ flat
        self.n_images += B

    @torch.no_grad()
    def finalize(
        self,
        regularization: float = 1e-4,
        shrinkage: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Compute per-position means, the global mean, and the shared Sigma^-1.

        Args:
            regularization: ridge added to the diagonal before inversion.
            shrinkage: in [0, 1]; pulls Sigma toward (trace/D) * I. Ledoit-Wolf
                style. Useful when the sample-to-dimension ratio is small.

        Returns:
            dict with mu_positional (N, D), mu_global (D,), sigma_inv (D, D).
        """
        if self.n_images < 2:
            raise RuntimeError(f"need at least 2 images, got {self.n_images}")

        M, N, D = self.n_images, self.num_positions, self.feature_dim
        mu_pos = self.sum_x / M                       # (N, D)
        mu_global = mu_pos.mean(dim=0)                # (D,)

        # Pooled WITHIN-position scatter: total scatter minus between-position part.
        between = M * (mu_pos.T @ mu_pos)             # (D, D)
        scatter = self.sum_xx - between
        dof = max(M * N - N, 1)
        sigma = scatter / dof

        # Symmetrize — the subtraction above can leave tiny asymmetries that make
        # the inverse complex-valued in edge cases.
        sigma = 0.5 * (sigma + sigma.T)

        # TOTAL covariance: centered on the global mean, i.e. within + between.
        # This is the right Sigma for a position-agnostic model, and having it
        # makes blend=0 a genuine control arm rather than a mismatched hybrid:
        # per-position means need the within-position spread, a single global
        # mean needs the full spread.
        total_scatter = self.sum_xx - (M * N) * torch.outer(mu_global, mu_global)
        sigma_total = total_scatter / max(M * N - 1, 1)
        sigma_total = 0.5 * (sigma_total + sigma_total.T)

        eye = torch.eye(D, dtype=sigma.dtype, device=sigma.device)
        if shrinkage > 0.0:
            sigma = (1.0 - shrinkage) * sigma + shrinkage * (torch.trace(sigma) / D) * eye
            sigma_total = (
                (1.0 - shrinkage) * sigma_total
                + shrinkage * (torch.trace(sigma_total) / D) * eye
            )

        sigma_inv = torch.linalg.inv(sigma + regularization * eye)
        sigma_inv_total = torch.linalg.inv(sigma_total + regularization * eye)

        return {
            "mu_positional": mu_pos.float().cpu(),
            "mu_global": mu_global.float().cpu(),
            "sigma_inv": sigma_inv.float().cpu(),
            "sigma_inv_total": sigma_inv_total.float().cpu(),
            "n_images": M,
            "n_positions": N,
            "feature_dim": D,
            "samples_per_dim": (M * N) / D,
        }


def build_positional_scorer(
    stats: dict,
    blend: float = 1.0,
    gamma: float = 1.0,
    device: str | torch.device = "cpu",
) -> PositionalMahalanobisScoring:
    """Construct a fitted scorer from a statistics dict.

    Args:
        stats: output of `PositionalStatsAccumulator.finalize`.
        blend: 1.0 = fully per-position means; 0.0 = global mean everywhere,
            which reproduces position-agnostic scoring and is the control arm
            for the ablation. Values between interpolate.
        gamma: non-linear amplification, matching the configured scoring.gamma.

    Returns:
        An initialized PositionalMahalanobisScoring.
    """
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"blend must be in [0, 1], got {blend}")

    mu_pos = stats["mu_positional"]
    mu_global = stats["mu_global"]
    n_positions, feature_dim = mu_pos.shape

    scorer = PositionalMahalanobisScoring(
        feature_dim=feature_dim, num_positions=n_positions, gamma=gamma
    )
    mu = blend * mu_pos + (1.0 - blend) * mu_global.unsqueeze(0)
    scorer.mu.data = mu.to(device)

    # Match the covariance to the mean model: a global mean must be paired with
    # the TOTAL covariance, otherwise blend=0 is neither the baseline nor P1 and
    # the ablation measures nothing.
    if blend == 0.0 and "sigma_inv_total" in stats:
        scorer.sigma_inv.data = stats["sigma_inv_total"].to(device)
    else:
        scorer.sigma_inv.data = stats["sigma_inv"].to(device)
    scorer.is_initialized.data = torch.tensor(True, device=device)
    return scorer.to(device)
