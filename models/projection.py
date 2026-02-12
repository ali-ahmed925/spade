"""Linear projection from Q-Former output space to LLM embedding space.

Maps (B, Q, Dq) → (B, Q, D_llm) so that Q-Former query tokens can be
consumed by a frozen LLM as soft visual tokens.
"""

import torch
import torch.nn as nn


class LLMProjection(nn.Module):
    """Single linear projection with LayerNorm."""

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 2560,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, query_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_embeds: (B, Q, Dq) Q-Former outputs.

        Returns:
            (B, Q, D_llm) projected tokens for the LLM.
        """
        return self.proj(query_embeds)  # (B, Q, D_llm)



