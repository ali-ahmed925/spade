"""SPADE — top-level model assembling all components.

Loads pretrained BLIP-2 and extracts:
  - Frozen ViT-G vision encoder
  - Fine-tunable Q-Former + query tokens
Then adds custom trainable heads:
  - Patch anomaly head (on ViT patch embeddings)
  - LLM projection head (on Q-Former outputs)
"""

import torch
import torch.nn as nn
from transformers import Blip2Model

from models.vit import FrozenVisionEncoder
from models.qformer import Blip2QFormerWrapper
from models.heads import PatchAnomalyHead
from models.projection import LLMProjection


class SPADE(nn.Module):
    """Full SPADE model.

    Trainable: Q-Former (fine-tuned), PatchAnomalyHead, LLMProjection.
    Frozen:    ViT-G (always), LLM (loaded separately for inference).
    """

    def __init__(
        self,
        blip2_model_name: str = "Salesforce/blip2-opt-2.7b",
        patch_head_hidden: int = 256,
        patch_head_dropout: float = 0.1,
        llm_embed_dim: int = 2560,
    ) -> None:
        super().__init__()

        # Load pretrained BLIP-2 and extract vision + Q-Former components.
        # The language model is temporarily loaded but freed immediately.
        import os
        token = os.environ.get("HF_TOKEN")
        blip2 = Blip2Model.from_pretrained(
            blip2_model_name,
            token=token,
            torch_dtype=torch.float16,  # Use half precision to save memory
        )

        # Frozen vision encoder (ViT-G, 1408-d, patch_size=14)
        self.vision_encoder = FrozenVisionEncoder(blip2.vision_model)

        # Trainable Q-Former (pretrained weights, fine-tuned)
        self.qformer = Blip2QFormerWrapper(blip2.qformer, blip2.query_tokens)

        # Free the language model weights we don't need
        del blip2
        torch.cuda.empty_cache()

        # Dims derived from pretrained model
        vit_dim = self.vision_encoder.hidden_size     # 1408
        qformer_dim = self.qformer.hidden_size        # 768

        # Trainable patch anomaly head (operates on ViT patch embeddings)
        self.patch_head = PatchAnomalyHead(
            input_dim=vit_dim,
            hidden_dim=patch_head_hidden,
            dropout=patch_head_dropout,
        )

        # Trainable projection to LLM embedding space
        self.projection = LLMProjection(
            input_dim=qformer_dim,
            output_dim=llm_embed_dim,
        )

    # ── forward ───────────────────────────────
    def forward(
        self, images: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: (B, 3, H, W) normalised image batch.

        Returns:
            dict with keys:
                patch_logits:   (B, N_patches) anomaly logits per patch.
                query_embeds:   (B, Q, D_qformer) Q-Former outputs.
                visual_tokens:  (B, Q, D_llm) projected tokens for LLM.
        """
        # 1. Frozen ViT patch embeddings (including CLS)
        image_embeds = self.vision_encoder(images)  # (B, N+1, D_v) - may be float16

        # 2. Q-Former: queries attend to vision tokens (needs CLS included)
        # Convert to float32 for trainable components (better training stability)
        query_embeds = self.qformer(image_embeds.float())   # (B, Q, D_q)

        # 3. Patch-level anomaly scores (drop CLS token at index 0)
        patch_embeds = image_embeds[:, 1:, :].float()       # (B, N, D_v) - convert to float32
        patch_logits = self.patch_head(patch_embeds)        # (B, N)

        # 4. Project Q-Former outputs for LLM
        visual_tokens = self.projection(query_embeds)       # (B, Q, D_llm)

        return {
            "patch_logits": patch_logits,
            "query_embeds": query_embeds,
            "visual_tokens": visual_tokens,
        }

    # ── convenience ───────────────────────────
    def get_image_score(self, patch_logits: torch.Tensor) -> torch.Tensor:
        """Aggregate patch logits into an image-level anomaly score.

        Uses max-pool over patch scores after sigmoid.

        Args:
            patch_logits: (B, N) raw logits.

        Returns:
            (B,) image-level anomaly scores.
        """
        patch_scores = torch.sigmoid(patch_logits)      # (B, N)
        image_scores = patch_scores.max(dim=-1).values   # (B,)
        return image_scores
