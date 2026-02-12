"""Pretrained BLIP-2 Q-Former — fine-tuned for anomaly detection.

Wraps the Q-Former and learnable query tokens from a pretrained BLIP-2 model.
All parameters are trainable (fine-tuned from pretrained weights).
"""

import torch
import torch.nn as nn


class Blip2QFormerWrapper(nn.Module):
    """Wrapper around BLIP-2's pretrained Q-Former.

    Learnable query tokens attend to ViT patch embeddings via cross-attention.
    Produces anomaly-aware query representations.

    Output shape: (B, Q, D_qformer)
    """

    def __init__(
        self,
        qformer: nn.Module,
        query_tokens: nn.Parameter,
    ) -> None:
        super().__init__()
        self.qformer = qformer
        # Clone pretrained query tokens so they survive `del blip2`
        self.query_tokens = nn.Parameter(query_tokens.data.clone())

    @property
    def hidden_size(self) -> int:
        """Q-Former hidden dimension (768 for BLIP-2)."""
        return self.qformer.config.hidden_size

    @property
    def num_queries(self) -> int:
        """Number of learnable query tokens (32 for BLIP-2)."""
        return self.query_tokens.shape[1]

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_embeds: (B, N+1, D_vision) ViT embeddings including CLS.

        Returns:
            (B, Q, D_qformer) query outputs.
        """
        B = image_embeds.shape[0]

        # Attention mask: all ones (attend to every vision token)
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device,
        )

        # Expand query tokens for the batch
        query_tokens = self.query_tokens.expand(B, -1, -1)  # (B, Q, D)

        outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
        )
        return outputs.last_hidden_state  # (B, Q, D_qformer)
