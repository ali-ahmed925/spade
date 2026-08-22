"""Patch-level losses for normal-only training (Mahalanobis-based).

These losses encourage tight clustering of normal patches by minimizing
Mahalanobis distances and their variance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MahalanobisPatchLoss(nn.Module):
    """
    Loss for normal-only training:
    - Push normal patch scores (Mahalanobis distances) as low as possible
    - Encourage tight cluster (low variance)
    
    This loss directly optimizes the Mahalanobis distances computed by the model,
    encouraging normal patches to cluster tightly around the learned mean.
    """

    def __init__(self, clamp_max: float = 100.0, lambda_var: float = 0.1):
        """
        Args:
            clamp_max: maximum distance to avoid exploding loss (soft clipping)
            lambda_var: weight for variance regularization term
        """
        super().__init__()
        self.clamp_max = clamp_max
        self.lambda_var = lambda_var

    def forward(self, scores: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            scores: (B, N) Mahalanobis distances per patch
            targets: not used (placeholder for API compatibility)
        Returns:
            scalar loss
        """
        # Soft clipping instead of hard clamp for better gradient flow
        # scores / (1 + scores/scale) smoothly saturates to scale
        scale = self.clamp_max
        scores_clipped = scores / (1.0 + scores / scale)
        
        # Mean distance (encourages small distances for normal patches)
        mean_loss = scores_clipped.mean()
        
        # Variance of distances (encourages tight cluster - all patches similar distance)
        var_loss = scores_clipped.var()
        
        total_loss = mean_loss + self.lambda_var * var_loss
        return total_loss


class PseudoAnomalyLoss(nn.Module):
    """Margin loss pushing perturbed (pseudo-anomalous) patches above clean ones.

    HISTORY / BUG
    -------------
    The original implementation was::

        perturbed_scores = scores + self.epsilon * torch.randn_like(scores)
        loss = F.relu(perturbed_scores - scores).mean()

    ``scores`` cancels algebraically, leaving ``relu(epsilon * noise)`` — a
    random constant with respect to every parameter in the model. Its gradient
    is *exactly* zero (verified in tests/test_gradient_flow.py), so the term
    contributed nothing but noise to the reported loss value.

    FIX
    ---
    The perturbation must be applied where the score actually comes from — the
    patch embeddings — and the perturbed patch must then be re-scored by the
    model. ``SPADE._score_perturbed`` does that and hands both score tensors
    here. The objective is a margin: a perturbed patch should score at least
    ``margin`` higher than the clean patch it came from.
    """

    def __init__(self, epsilon: float = 0.01, margin: float = 0.1):
        """
        Args:
            epsilon: perturbation magnitude (applied by the model, kept here for
                bookkeeping and for backwards-compatible construction).
            margin: how much higher a perturbed patch should score.
        """
        super().__init__()
        self.epsilon = epsilon
        self.margin = margin

    def forward(
        self,
        scores: torch.Tensor,
        perturbed_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            scores: (B, N) scores for the clean patches.
            perturbed_scores: (B, N) scores for the perturbed copies of the SAME
                patches, produced by re-running the scoring path on perturbed
                embeddings.

        Returns:
            scalar margin loss.
        """
        if perturbed_scores is None:
            raise ValueError(
                "PseudoAnomalyLoss requires perturbed_scores computed from perturbed "
                "embeddings. Pass perturb_epsilon to SPADE.forward() so the model "
                "returns 'patch_scores_perturbed'."
            )
        # want: perturbed >= clean + margin
        violation = self.margin - (perturbed_scores - scores)
        return F.relu(violation).mean()
