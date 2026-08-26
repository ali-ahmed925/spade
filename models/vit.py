"""Frozen BLIP-2 Vision Encoder — multi-layer, high-resolution patch extractor.

Runs BLIP-2's ViT-G at a configurable input resolution (448 by default, giving a
32x32 = 1024 patch grid instead of 16x16 = 256) and taps INTERMEDIATE encoder
blocks rather than only the final one.

WHY BOTH CHANGES
----------------
Resolution: at 224 with patch_size 14, one patch covers 196 px = 0.39% of the
image. Measured MVTec defects such as capsule crack (0.30% area), poke (0.23%)
and faulty_imprint (0.28%) are therefore SMALLER THAN A SINGLE PATCH — they are
averaged with normal material inside one embedding before anything scores them.
At 448 each patch covers 0.098% and those defects span ~3 patches instead of
~0.7. PaDiM and PatchCore operate at 56x56 and 28x28 grids respectively for the
same reason.

Layer choice: the final ViT-G block is the most semantic representation in the
network — BLIP-2 trained it to answer "what object is this", which is precisely
the invariance we do not want. A bent lead is still semantically a lead; a
hairline crack is still semantically capsule surface. Mid-level blocks retain
local appearance, so both are used and fused downstream.

All parameters stay frozen; only the downstream fusion/contextualisation layers
are trainable.
"""

import torch
import torch.nn as nn


class FrozenVisionEncoder(nn.Module):
    """Wrapper around BLIP-2's vision model with multi-layer, multi-resolution output.

    Output contract:
        forward(x)                    -> (B, 1 + N, D) final hidden states (incl. CLS)
        forward(x, return_layers=[...]) -> dict with:
            "final":  (B, 1 + N, D)   post-layernorm final hidden states
            "layers": list of (B, 1 + N, D), one per requested block, in order
    """

    def __init__(
        self,
        vision_model: nn.Module,
        image_size: int = 448,
        feature_layers: tuple[int, ...] | list[int] = (20, 30),
    ) -> None:
        super().__init__()
        self.vision_model = vision_model
        self.image_size = int(image_size)

        n_blocks = len(self.vision_model.encoder.layers)
        resolved = []
        for idx in feature_layers:
            i = int(idx)
            if i < 0:
                i += n_blocks
            if not 0 <= i < n_blocks:
                raise ValueError(
                    f"feature layer {idx} out of range for a {n_blocks}-block encoder"
                )
            resolved.append(i)
        if not resolved:
            raise ValueError("feature_layers must not be empty")
        self.feature_layers = tuple(sorted(set(resolved)))
        self.num_blocks = n_blocks

        # Freeze all parameters
        for param in self.vision_model.parameters():
            param.requires_grad = False

    # ── geometry ──────────────────────────────────────────────────────────────
    @property
    def hidden_size(self) -> int:
        """Vision encoder hidden dimension (1408 for ViT-G)."""
        return self.vision_model.config.hidden_size

    @property
    def patch_size(self) -> int:
        return self.vision_model.config.patch_size

    @property
    def grid_size(self) -> int:
        """Patches per side at the configured input resolution (32 at 448/14)."""
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        """Spatial patches, excluding CLS (1024 at 448/14)."""
        return self.grid_size ** 2

    @property
    def native_image_size(self) -> int:
        """Resolution the position embeddings were pretrained at (224)."""
        return self.vision_model.config.image_size

    @property
    def needs_interpolation(self) -> bool:
        return self.image_size != self.native_image_size

    # ── forward ───────────────────────────────────────────────────────────────
    @torch.no_grad()
    def forward(
        self,
        pixel_values: torch.Tensor,
        return_layers: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | list[torch.Tensor]]:
        """Extract patch embeddings, optionally from intermediate blocks too.

        The encoder stack is stepped manually rather than calling
        `Blip2VisionModel.forward`, because that method constructs its output as
        `BaseModelOutputWithPooling(last_hidden_state=..., pooler_output=...)`
        and drops `hidden_states` — so `output_hidden_states=True` cannot reach
        the intermediate blocks through it. Stepping the layers is explicit and
        version-independent.

        Args:
            pixel_values: (B, 3, H, W) normalized images at `self.image_size`.
            return_layers: if True, also return the tapped intermediate blocks.

        Returns:
            (B, 1 + N, D) final hidden states, or a dict with "final" and "layers".
        """
        expected = self.image_size
        if pixel_values.shape[-1] != expected or pixel_values.shape[-2] != expected:
            raise ValueError(
                f"expected {expected}x{expected} input for this encoder "
                f"(grid {self.grid_size}x{self.grid_size}), got "
                f"{tuple(pixel_values.shape[-2:])}. The dataset/transform image_size "
                "must match vit.image_size in config/model.yaml."
            )

        # Position embeddings are pretrained for 224; interpolate bicubically for
        # any other resolution (native support in Blip2VisionEmbeddings).
        hidden = self.vision_model.embeddings(
            pixel_values, interpolate_pos_encoding=self.needs_interpolation
        )

        wanted = set(self.feature_layers) if return_layers else set()
        taps: dict[int, torch.Tensor] = {}
        for i, block in enumerate(self.vision_model.encoder.layers):
            hidden = block(hidden)
            if i in wanted:
                taps[i] = hidden

        final = self.vision_model.post_layernorm(hidden)
        if not return_layers:
            return final
        return {"final": final, "layers": [taps[i] for i in self.feature_layers]}
