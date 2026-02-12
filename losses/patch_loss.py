"""Patch-level anomaly detection loss.

Binary Cross-Entropy (optionally Focal) between predicted patch logits
and synthetic patch-level ground-truth masks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchBCELoss(nn.Module):
    """Standard BCE loss for patch anomaly scores."""

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits:  (B, N) raw patch logits.
            targets: (B, N) binary patch labels.

        Returns:
            Scalar loss.
        """
        return F.binary_cross_entropy_with_logits(logits, targets)


class FocalLoss(nn.Module):
    """Focal loss to handle class imbalance in patch labels.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits:  (B, N) raw patch logits.
            targets: (B, N) binary patch labels.

        Returns:
            Scalar focal loss.
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()



