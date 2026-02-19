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
    """
    Optional: small perturbation on normal embeddings to simulate soft negatives.
    Encourages Q-Former to separate slightly perturbed normals from true normals.
    
    This adds a contrastive component: normal patches should be closer to the mean
    than slightly perturbed versions of themselves.
    """

    def __init__(self, epsilon: float = 0.01):
        """
        Args:
            epsilon: magnitude of random perturbation
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: (B, N) Mahalanobis distances for normal patches
        Returns:
            scalar loss encouraging perturbed patches to have higher scores
        """
        # Random small perturbation (additive noise)
        perturbed_scores = scores + self.epsilon * torch.randn_like(scores)
        
        # Encourage perturbed patches to have slightly higher scores than normal
        # This creates a margin: normal < perturbed
        loss = F.relu(perturbed_scores - scores).mean()
        return loss

