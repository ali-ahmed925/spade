"""Patch-level anomaly detection head.

Lightweight MLP that predicts an anomaly score for each ViT patch.
"""

import torch
import torch.nn as nn


class PatchAnomalyHead(nn.Module):
    """MLP head producing per-patch anomaly scores.

    Takes ViT patch embeddings (CLS excluded) and predicts a scalar
    anomaly probability per patch.

    Output shape: (B, N_patches)
    """

    def __init__(
        self,
        input_dim: int = 1408,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, patch_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_embeds: (B, N, D) patch embeddings (CLS excluded).

        Returns:
            (B, N) anomaly logits per patch (before sigmoid).
        """
        logits = self.mlp(patch_embeds)  # (B, N, 1)
        return logits.squeeze(-1)         # (B, N)
