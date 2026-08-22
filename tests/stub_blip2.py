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


class StubVisionModel(nn.Module):
    """Patch-embed + CLS. Mirrors ViT-G's output contract at 1/44th the width."""

    def __init__(self, hidden_size: int = 32, image_size: int = 224, patch_size: int = 14):
        super().__init__()
        self.config = _Config(
            hidden_size=hidden_size, image_size=image_size, patch_size=patch_size
        )
        self.patch_embed = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.cls = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

    def forward(self, pixel_values: torch.Tensor) -> _Output:
        x = self.patch_embed(pixel_values)              # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)                # (B, N, D)
        cls = self.cls.expand(x.shape[0], -1, -1)
        return _Output(torch.cat([cls, x], dim=1))      # (B, N+1, D)


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
    ):
        super().__init__()
        self.vision_model = StubVisionModel(vision_dim, image_size, patch_size)
        self.qformer = StubQFormer(qformer_dim, vision_dim)
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, qformer_dim) * 0.02)


def make_stub_blip2(seed: int = 0, **kw) -> StubBlip2:
    torch.manual_seed(seed)
    return StubBlip2(**kw)


def make_stub_spade(seed: int = 0, hpa: bool = False, frequency: bool = False, **spade_kw):
    """Build a SPADE model on the stub backbone, with sensible small defaults."""
    from models.spade import SPADE

    blip2 = make_stub_blip2(seed=seed)
    defaults = dict(
        llm_embed_dim=64,
        hpa_n_max=256,
        hpa_n_min=32,
        hpa_t_steps=3,
        score_alpha=0.25,
        score_beta=0.65,
        score_lambda=0.001,
        mahalanobis_gamma=1.0,
        mahalanobis_reg=1e-4,
        normal_stats_buffer_size=2048,
        normal_stats_update_frequency=1,
    )
    defaults.update(spade_kw)
    model = SPADE(blip2_model=blip2, **defaults)
    model.use_hpa = hpa
    if frequency:
        model.enable_frequency_features(score_gamma=0.1)
    return model


def fit_statistics(model, images: torch.Tensor) -> None:
    """Populate the Mahalanobis statistics so scores are not identically zero."""
    with torch.no_grad():
        embeds = model.vision_encoder(images)[:, 1:, :].float()
        flat = embeds.reshape(-1, embeds.shape[-1])
        model.mahalanobis_scorer.update_statistics(flat)
