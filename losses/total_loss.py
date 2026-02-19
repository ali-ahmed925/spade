"""Combined loss: patch loss for anomaly detection.

Supports two modes:
1. Synthetic training: BCE/Focal loss with binary targets
2. Normal-only training: Mahalanobis clustering loss
"""

import torch
import torch.nn as nn

from losses.patch_loss import PatchBCELoss, FocalLoss
from losses.patch_loss_normal import MahalanobisPatchLoss, PseudoAnomalyLoss


class TotalLoss(nn.Module):
    """Weighted combination of all training losses.
    
    Automatically selects appropriate loss based on training mode:
    - Synthetic training: BCE/Focal loss with binary targets
    - Normal-only training: Mahalanobis clustering loss
    """

    def __init__(
        self,
        patch_weight: float = 1.0,
        use_focal: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        # Normal-only training parameters
        use_normal_only: bool = False,
        var_weight: float = 0.1,
        use_pseudo: bool = False,
        pseudo_epsilon: float = 0.01,
        clamp_max: float = 100.0,
    ) -> None:
        """
        Args:
            patch_weight: weight for patch loss
            use_focal: use focal loss (for synthetic training)
            focal_alpha: focal loss alpha parameter
            focal_gamma: focal loss gamma parameter
            use_normal_only: if True, use Mahalanobis clustering loss instead of BCE
            var_weight: weight for variance term in Mahalanobis loss
            use_pseudo: use pseudo-anomaly loss for contrastive learning
            pseudo_epsilon: perturbation magnitude for pseudo-anomaly loss
            clamp_max: maximum score for soft clipping
        """
        super().__init__()
        self.patch_weight = patch_weight
        self.use_normal_only = use_normal_only

        if use_normal_only:
            # Normal-only training: Mahalanobis clustering loss
            self.patch_loss_fn = MahalanobisPatchLoss(
                clamp_max=clamp_max,
                lambda_var=var_weight,
            )
            if use_pseudo:
                self.pseudo_loss_fn = PseudoAnomalyLoss(epsilon=pseudo_epsilon)
            else:
                self.pseudo_loss_fn = None
        else:
            # Synthetic training: BCE/Focal loss
            self.patch_loss_fn = (
                FocalLoss(focal_alpha, focal_gamma) if use_focal else PatchBCELoss()
            )
            self.pseudo_loss_fn = None

    def forward(
        self,
        patch_scores: torch.Tensor,
        patch_targets: torch.Tensor,
        query_embeds: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            patch_scores:  (B, N) patch anomaly scores (Mahalanobis-based).
            patch_targets: (B, N) binary patch labels (ignored for normal-only training).
            query_embeds:  (B, Q, D) Q-Former outputs (unused, kept for API compatibility).
            labels:        (B,) image-level binary labels (unused, kept for API compatibility).

        Returns:
            dict with keys: total, patch, [pseudo] — all scalar tensors.
        """
        if self.use_normal_only:
            # Normal-only training: minimize Mahalanobis distances
            l_patch = self.patch_loss_fn(patch_scores, patch_targets)
            total = self.patch_weight * l_patch
            
            result = {
                "total": total,
                "patch": l_patch,
            }
            
            # Add pseudo-anomaly loss if enabled
            if self.pseudo_loss_fn is not None:
                l_pseudo = self.pseudo_loss_fn(patch_scores)
                result["pseudo"] = l_pseudo
                result["total"] = result["total"] + l_pseudo
        else:
            # Synthetic training: BCE/Focal loss with binary targets
            l_patch = self.patch_loss_fn(patch_scores, patch_targets)
            total = self.patch_weight * l_patch
            
            result = {
                "total": total,
                "patch": l_patch,
            }
        
        return result
