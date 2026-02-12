"""Contrastive loss for anomaly detection.

Only enforces consistency among NORMAL samples. Anomalies are NOT pulled together
(they are heterogeneous and should not cluster). This aligns with open-set
anomaly detection geometry.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """InfoNCE-style contrastive loss on Q-Former query embeddings.

    Strategy:
    - Normal ↔ Normal → positive pairs (pulled together)
    - Normal ↔ Anomaly → negative pairs (pushed apart)
    - Anomaly ↔ Anomaly → ignored (not pulled together, as anomalies are heterogeneous)

    This enforces a compact normal manifold while keeping anomalies separate.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_embeds: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_embeds: (B, Q, D) Q-Former outputs.
            labels:       (B,) binary labels (0=normal, 1=anomaly).

        Returns:
            Scalar contrastive loss.
        """
        # Mean-pool over queries → (B, D)
        embeds = query_embeds.mean(dim=1)
        embeds = F.normalize(embeds, dim=-1)

        # Cosine similarity matrix (B, B)
        sim = embeds @ embeds.T / self.temperature

        # Mask out self-similarity
        B = embeds.shape[0]
        mask_self = ~torch.eye(B, dtype=torch.bool, device=embeds.device)

        # Positive pairs: ONLY normal-normal pairs
        # This ensures anomalies are NOT pulled together (they are heterogeneous)
        label_match = (
            (labels.unsqueeze(0) == labels.unsqueeze(1)) &  # same label
            (labels.unsqueeze(0) == 0) &                    # both normal
            (labels.unsqueeze(1) == 0)                      # both normal
        ) & mask_self  # (B, B)

        # If no normal-normal pairs exist, return zero loss to avoid NaN
        if label_match.sum() == 0:
            # Keep graph-friendly zero (so backward() is always safe)
            return embeds.sum() * 0.0

        # SupCon-style: for each anchor, average log-prob over its positives
        exp_sim = torch.exp(sim) * mask_self.float()
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean log-prob over positive pairs (normal-normal only)
        loss = -(log_prob * label_match.float()).sum() / label_match.float().sum()
        return loss



