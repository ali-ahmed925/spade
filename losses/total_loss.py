"""Combined loss: patch loss for anomaly detection.

Supports two modes:
1. Synthetic training: BCE/Focal loss with binary targets
2. Normal-only training: Mahalanobis clustering loss
"""

import torch
import torch.nn as nn

from losses.grounding_loss import QueryGroundingLoss
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
        pseudo_margin: float = 0.1,
        clamp_max: float = 100.0,
        # ── auxiliary query grounding ──
        grounding_weight: float = 0.0,
        grounding_queries: int = 4,
        grounding_pos_weight: float | str = "auto",
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

        # Auxiliary objective on the patch->query attention map. It does NOT
        # touch the detection term: the anomaly score never sees a synthetic
        # label. weight == 0 disables it entirely, which is the exact control
        # for any comparison.
        self.grounding_weight = float(grounding_weight)
        self.grounding_loss_fn = (
            QueryGroundingLoss(
                n_anomaly_queries=grounding_queries, pos_weight=grounding_pos_weight
            )
            if self.grounding_weight > 0
            else None
        )

        if use_normal_only:
            # Normal-only training: Mahalanobis clustering loss
            self.patch_loss_fn = MahalanobisPatchLoss(
                clamp_max=clamp_max,
                lambda_var=var_weight,
            )
            if use_pseudo:
                self.pseudo_loss_fn = PseudoAnomalyLoss(
                    epsilon=pseudo_epsilon, margin=pseudo_margin
                )
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
        patch_scores_perturbed: torch.Tensor | None = None,
        patch_query_attention: torch.Tensor | None = None,
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
            
            # Add pseudo-anomaly loss if enabled. Requires the model to have
            # scored a perturbed copy of the patches (perturb_epsilon != None);
            # scoring-level perturbation cancels and has zero gradient.
            if self.pseudo_loss_fn is not None:
                l_pseudo = self.pseudo_loss_fn(
                    patch_scores, patch_scores_perturbed, patch_targets
                )
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

        # ── auxiliary grounding, added to BOTH branches ──
        # Gradients flow through the contextualizer into the Q-Former, so the
        # query tokens actually learn; the detection term above is unchanged.
        result["detection"] = result["total"]
        if self.grounding_loss_fn is not None:
            if patch_query_attention is None:
                raise ValueError(
                    "grounding_weight > 0 but no patch_query_attention was passed. "
                    "Call SPADE.forward(..., return_attention=True)."
                )
            l_ground, diagnostics = self.grounding_loss_fn(
                patch_query_attention, patch_targets
            )
            result["grounding"] = l_ground
            result["grounding_diagnostics"] = diagnostics
            result["total"] = result["total"] + self.grounding_weight * l_ground

        return result
