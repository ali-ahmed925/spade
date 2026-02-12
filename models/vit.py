"""Frozen BLIP-2 Vision Encoder — patch embedding extractor.

Wraps BLIP-2's pretrained ViT-G. All parameters are frozen.
Returns patch embeddings including CLS token (Q-Former expects CLS).
"""

import torch
import torch.nn as nn


class FrozenVisionEncoder(nn.Module):
    """Wrapper around BLIP-2's vision model.

    Output shape: (B, N_patches + 1, D_vision)
        — includes CLS at position 0 (required by BLIP-2 Q-Former)
    """

    def __init__(self, vision_model: nn.Module) -> None:
        super().__init__()
        self.vision_model = vision_model

        # Freeze all parameters
        for param in self.vision_model.parameters():
            param.requires_grad = False

    @property
    def hidden_size(self) -> int:
        """Vision encoder hidden dimension (1408 for ViT-G)."""
        return self.vision_model.config.hidden_size

    @property
    def num_patches(self) -> int:
        """Number of spatial patches (excluding CLS)."""
        img = self.vision_model.config.image_size
        patch = self.vision_model.config.patch_size
        return (img // patch) ** 2

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract patch embeddings from image batch.

        Args:
            pixel_values: (B, 3, H, W) normalised image tensor.

        Returns:
            (B, N_patches + 1, D) embeddings including CLS at index 0.
        """
        outputs = self.vision_model(pixel_values=pixel_values)
        return outputs.last_hidden_state  # (B, N+1, D)
