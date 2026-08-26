"""A weightless stand-in for BLIP-2, for tests.

The real backbone is a ~15 GB download and a ~1 B parameter ViT-G. None of the
gradient-flow properties we need to verify depend on the *size* of the backbone,
only on its interface, so tests build a tiny module exposing the same surface:

    blip2.vision_model   -> .config.{hidden_size,image_size,patch_size}
                            __call__(pixel_values=...) -> .last_hidden_state (B, N+1, D_v)
    blip2.qformer        -> .config.hidden_size
                            __call__(query_embeds=, encoder_hidden_states=,
                                     encoder_attention_mask=) -> .last_hidden_state (B, Q, D_q)
    blip2.query_tokens   -> nn.Parameter (1, Q, D_q)

Pass the result to SPADE(..., blip2_model=make_stub_blip2()).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _Config:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Output:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _StubEmbeddings(nn.Module):
    """Patch-embed + CLS, mirroring Blip2VisionEmbeddings' interface.

    Accepts `interpolate_pos_encoding` because FrozenVisionEncoder passes it
    whenever the working resolution differs from the pretrained one.
    """

    def __init__(self, hidden_size: int, patch_size: int):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            3, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.class_embedding = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False):
        x = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)   # (B, N, D)
        cls = self.class_embedding.expand(x.shape[0], -1, -1)
        return torch.cat([cls, x], dim=1)                                   # (B, 1+N, D)


class _StubBlock(nn.Module):
    """One encoder block: returns a plain tensor, like Blip2EncoderLayer."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.ffn(self.norm(hidden_states))


class _StubEncoder(nn.Module):
    def __init__(self, hidden_size: int, n_blocks: int):
        super().__init__()
        self.layers = nn.ModuleList([_StubBlock(hidden_size) for _ in range(n_blocks)])


class StubVisionModel(nn.Module):
    """Mirrors Blip2VisionModel's structure: embeddings / encoder.layers / post_layernorm.

    FrozenVisionEncoder steps the encoder blocks directly to tap intermediates,
    so the stub must expose the same attribute layout rather than just a forward.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        image_size: int = 224,
        patch_size: int = 14,
        n_blocks: int = 6,
    ):
        super().__init__()
        self.config = _Config(
            hidden_size=hidden_size, image_size=image_size, patch_size=patch_size
        )
        self.embeddings = _StubEmbeddings(hidden_size, patch_size)
        self.encoder = _StubEncoder(hidden_size, n_blocks)
        self.post_layernorm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> _Output:
        h = self.embeddings(pixel_values, interpolate_pos_encoding)
        for block in self.encoder.layers:
            h = block(h)
        return _Output(self.post_layernorm(h))


class StubQFormer(nn.Module):
    """Cross-attention block with real parameters, so gradients are meaningful."""

    def __init__(self, hidden_size: int = 16, vision_dim: int = 32, n_heads: int = 2):
        super().__init__()
        self.config = _Config(hidden_size=hidden_size)
        self.kv_proj = nn.Linear(vision_dim, hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(
        self,
        query_embeds: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> _Output:
        kv = self.kv_proj(encoder_hidden_states)
        attended, _ = self.attn(query_embeds, kv, kv, need_weights=False)
        return _Output(self.ffn(attended + query_embeds))


class StubBlip2(nn.Module):
    def __init__(
        self,
        vision_dim: int = 32,
        qformer_dim: int = 16,
        num_queries: int = 4,
        image_size: int = 224,
        patch_size: int = 14,
        n_blocks: int = 6,
    ):
        super().__init__()
        self.vision_model = StubVisionModel(vision_dim, image_size, patch_size, n_blocks)
        self.qformer = StubQFormer(qformer_dim, vision_dim)
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, qformer_dim) * 0.02)


def make_stub_blip2(seed: int = 0, **kw) -> StubBlip2:
    torch.manual_seed(seed)
    return StubBlip2(**kw)


def make_stub_spade(seed: int = 0, frequency: bool = False, image_size: int = 224, **spade_kw):
    """Build a SPADE model on the stub backbone, with sensible small defaults.

    image_size defaults to 224 (a 16x16 grid) to keep tests fast; the production
    config uses 448. Nothing in the model hard-codes either.
    """
    from models.spade import SPADE

    blip2 = make_stub_blip2(seed=seed, image_size=image_size)
    defaults = dict(
        llm_embed_dim=64,
        image_size=image_size,
        feature_layers=(2, 4),
        fusion_proj_dim=16,
        context_hidden_dim=16,
        context_heads=2,
        score_beta=0.9,
        score_gamma=0.1,
        mahalanobis_gamma=1.0,
        mahalanobis_reg=1e-4,
        normal_stats_buffer_size=2048,
        normal_stats_update_frequency=1,
    )
    defaults.update(spade_kw)
    model = SPADE(blip2_model=blip2, **defaults)
    if frequency:
        model.enable_frequency_features(score_gamma=0.1)
    return model


def fit_statistics(model, images: torch.Tensor) -> None:
    """Populate the Mahalanobis statistics so scores are not identically zero.

    Fitted on CONTEXTUAL DESCRIPTORS, which is the space the scorer operates in.
    """
    with torch.no_grad():
        descriptors = model.build_descriptors(images)["descriptors"]
        flat = descriptors.reshape(-1, descriptors.shape[-1])
        model.mahalanobis_scorer.update_statistics(flat)
