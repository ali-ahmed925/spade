"""Patch-level anomaly detection loss.

Updated to work with patch_scores (Mahalanobis-based) instead of logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchBCELoss(nn.Module):
    """BCE loss for patch anomaly scores.
    
    Since patch_scores are in [0, inf], we normalize them to [0, 1] using sigmoid
    or a normalized version for training stability.
    """

    def forward(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            scores:  (B, N) patch anomaly scores (Mahalanobis-based, [0, inf]).
            targets: (B, N) binary patch labels.
            
        Returns:
            Scalar loss.
        """
        # Normalize scores to [0, 1] using sigmoid of log1p for numerical stability
        normalized_scores = torch.sigmoid(torch.log1p(scores))
        
        return F.binary_cross_entropy(normalized_scores, targets)


class FocalLoss(nn.Module):
    """Focal loss for patch anomaly scores.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 4.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            scores:  (B, N) patch anomaly scores (Mahalanobis-based).
            targets: (B, N) binary patch labels.
            
        Returns:
            Scalar focal loss.
        """
        # Normalize scores to probabilities
        probs = torch.sigmoid(torch.log1p(scores))  # (B, N)
        
        # Compute p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # BCE component
        bce = F.binary_cross_entropy(probs, targets, reduction="none")
        
        return (focal_weight * bce).mean()
