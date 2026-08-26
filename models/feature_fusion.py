"""Patch descriptor construction: multi-layer fusion + Q-Former contextualisation.

These two modules build the space that Mahalanobis scores. They replace the old
arrangement where Mahalanobis operated directly on the frozen final-layer ViT
embedding of each patch, independently of every other patch.

Both defect families we measured fail in that old space for different reasons:

  local appearance defects (poke_insulation, crack, damaged_case)
      need fine local detail, destroyed by using only the final semantic block
      -> addressed by MultiLayerPatchFusion, which mixes mid-level blocks in

  role/layout violations (cable_swap, cut_lead, misplaced)
      have individually NORMAL patches — blue insulation is legitimately blue,
      bare board is legitimately board — and are anomalous only relative to the
      rest of the image
      -> addressed by QueryPatchContextualizer, which conditions every patch on
         the 32 Q-Former query tokens summarising the whole image, so "blue
         where the layout expects brown" and "blue where blue belongs" become
         different vectors

Unlike the frozen encoder, these layers are TRAINABLE, which also means the
Mahalanobis term finally has a gradient path (see docs/GRADIENT_AUDIT.md — it
previously had none, because the frozen ViT runs under torch.no_grad).
"""

import torch
import torch.nn as nn


class MultiLayerPatchFusion(nn.Module):
    """Project and concatenate patch features tapped from several ViT blocks.

    Each block is layer-normalised independently before projection: activation
    scale grows substantially with depth in ViT-G, so concatenating raw features
    would let the deepest block dominate the fused descriptor and silently undo
    the point of using mid-level features at all.

    Output dim = len(layers) * proj_dim.
    """

    def __init__(self, in_dim: int, n_layers: int, proj_dim: int = 256):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.n_layers = n_layers
        self.proj_dim = proj_dim
        self.norms = nn.ModuleList([nn.LayerNorm(in_dim) for _ in range(n_layers)])
        self.projections = nn.ModuleList(
            [nn.Linear(in_dim, proj_dim) for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(n_layers * proj_dim)

    @property
    def output_dim(self) -> int:
        return self.n_layers * self.proj_dim

    def forward(self, layer_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            layer_features: list of (B, N, D) patch features, CLS already removed,
                one per tapped block, in the order the encoder produced them.

        Returns:
            (B, N, n_layers * proj_dim) fused patch features.
        """
        if len(layer_features) != self.n_layers:
            raise ValueError(
                f"expected {self.n_layers} tapped layers, got {len(layer_features)}"
            )
        projected = [
            proj(norm(feat.float()))
            for feat, norm, proj in zip(layer_features, self.norms, self.projections)
        ]
        return self.out_norm(torch.cat(projected, dim=-1))


class QueryPatchContextualizer(nn.Module):
    """Condition every patch on the image-level Q-Former query tokens.

    This is what makes the Q-Former load-bearing. Previously its only route into
    the score was an additive "attention importance" term measured at ~1e-5 of
    the score magnitude, and shown to be anti-correlated with anomaly at image
    level. Here the queries instead shape the descriptor that Mahalanobis scores,
    so the Q-Former sits directly on the detection path.

    Patches are the attention QUERIES and the 32 tokens are the KEYS/VALUES: each
    patch asks "given this image's global structure, what should be here?" and
    receives a context vector. The descriptor keeps both halves — local
    appearance and retrieved context — concatenated rather than summed, so
    Mahalanobis can weigh a patch that looks wrong and a patch that is in the
    wrong place independently.

    Output dim = 2 * hidden_dim.
    """

    def __init__(
        self,
        patch_dim: int,
        query_dim: int,
        hidden_dim: int = 256,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by n_heads {n_heads}")
        self.hidden_dim = hidden_dim
        self.patch_proj = nn.Linear(patch_dim, hidden_dim)
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.out_norm = nn.LayerNorm(2 * hidden_dim)

    @property
    def output_dim(self) -> int:
        return 2 * self.hidden_dim

    def forward(
        self,
        patch_features: torch.Tensor,
        query_embeds: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patch_features: (B, N, P) fused patch features.
            query_embeds:   (B, Q, E) Q-Former query outputs for the same images.
            return_attention: also return the (B, N, Q) patch-to-query attention,
                used for localization diagnostics and for grounding the language
                component's visual tokens.

        Returns:
            (B, N, 2 * hidden_dim) contextualised patch descriptors, and
            optionally the attention weights.
        """
        local = self.patch_proj(patch_features)                    # (B, N, H)
        kv = self.query_proj(query_embeds.to(patch_features.dtype))  # (B, Q, H)

        context, attn_weights = self.attn(
            local, kv, kv, need_weights=return_attention, average_attn_weights=True
        )
        context = self.context_norm(context)

        descriptors = self.out_norm(torch.cat([local, context], dim=-1))
        if return_attention:
            return descriptors, attn_weights
        return descriptors
