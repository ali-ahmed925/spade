"""Combined loss: patch loss for anomaly detection."""

import torch
import torch.nn as nn

from losses.patch_loss import PatchBCELoss, FocalLoss


class TotalLoss(nn.Module):
    """Weighted combination of all training losses."""

    def __init__(
        self,
        patch_weight: float = 1.0,
        use_focal: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.patch_weight = patch_weight

        self.patch_loss_fn = (
            FocalLoss(focal_alpha, focal_gamma) if use_focal else PatchBCELoss()
        )

    def forward(
        self,
        patch_scores: torch.Tensor,  # Changed from patch_logits
        patch_targets: torch.Tensor,
        query_embeds: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            patch_scores:  (B, N) patch anomaly scores (Mahalanobis-based).
            patch_targets: (B, N) binary patch labels.
            query_embeds:  (B, Q, D) Q-Former outputs (unused, kept for API compatibility).
            labels:        (B,) image-level binary labels (unused, kept for API compatibility).

        Returns:
            dict with keys: total, patch — all scalar tensors.
        """
        l_patch = self.patch_loss_fn(patch_scores, patch_targets)

        total = self.patch_weight * l_patch

        return {
            "total": total,
            "patch": l_patch,
        }
