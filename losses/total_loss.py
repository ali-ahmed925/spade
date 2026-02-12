"""Combined loss: weighted sum of patch loss and contrastive loss."""

import torch
import torch.nn as nn

from losses.patch_loss import PatchBCELoss, FocalLoss
from losses.contrastive_loss import ContrastiveLoss


class TotalLoss(nn.Module):
    """Weighted combination of all training losses."""

    def __init__(
        self,
        patch_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        use_focal: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        contrastive_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.patch_weight = patch_weight
        self.contrastive_weight = contrastive_weight

        self.patch_loss_fn = (
            FocalLoss(focal_alpha, focal_gamma) if use_focal else PatchBCELoss()
        )
        self.contrastive_loss_fn = ContrastiveLoss(contrastive_temperature)

    def forward(
        self,
        patch_logits: torch.Tensor,
        patch_targets: torch.Tensor,
        query_embeds: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            patch_logits:  (B, N) raw patch logits.
            patch_targets: (B, N) binary patch labels.
            query_embeds:  (B, Q, D) Q-Former outputs.
            labels:        (B,) image-level binary labels.

        Returns:
            dict with keys: total, patch, contrastive — all scalar tensors.
        """
        l_patch = self.patch_loss_fn(patch_logits, patch_targets)
        l_contra = self.contrastive_loss_fn(query_embeds, labels)

        total = self.patch_weight * l_patch + self.contrastive_weight * l_contra

        return {
            "total": total,
            "patch": l_patch,
            "contrastive": l_contra,
        }



