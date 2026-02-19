"""SPADE — top-level model with Hierarchical Patch Anomaly Detection.

Loads pretrained BLIP-2 and extracts:
  - Frozen ViT-G vision encoder
  - Fine-tunable Q-Former + query tokens
Then adds hierarchical patch refinement with Mahalanobis scoring.
"""

import torch
import torch.nn as nn
from transformers import Blip2Model

from models.vit import FrozenVisionEncoder
from models.qformer import Blip2QFormerWrapper
from models.projection import LLMProjection
from models.hierarchical_patch_refinement import HybridPatchAnnealing, QueryPatchAttention
from models.mahalanobis_scoring import MahalanobisScoring
from models.normal_statistics import NormalStatisticsTracker


class SPADE(nn.Module):
    """Full SPADE model with Hierarchical Patch Refinement.
    
    Trainable: Q-Former, LLMProjection.
    Frozen: ViT-G, LLM.
    Statistical: Mahalanobis scoring (updated during training).
    """
    
    def __init__(
        self,
        blip2_model_name: str = "Salesforce/blip2-opt-2.7b",
        llm_embed_dim: int = 2560,
        # HPA parameters
        hpa_n_max: int = 256,
        hpa_n_min: int = 32,
        hpa_t_steps: int = 15,
        hpa_w: float = 0.4,
        hpa_p1: float = 0.5,
        hpa_p2: float = 2.0,
        # Scoring parameters
        score_alpha: float = 0.4,
        score_beta: float = 0.4,
        score_lambda: float = 0.2,
        mahalanobis_gamma: float = 1.3,
        mahalanobis_reg: float = 1e-6,
        # Normal statistics parameters
        normal_stats_buffer_size: int = 10000,
        normal_stats_update_frequency: int = 100,
    ):
        super().__init__()
        
        # Load pretrained BLIP-2
        import os
        token = os.environ.get("HF_TOKEN")
        blip2 = Blip2Model.from_pretrained(
            blip2_model_name,
            token=token,
            torch_dtype=torch.float16,
        )
        
        # Frozen vision encoder
        self.vision_encoder = FrozenVisionEncoder(blip2.vision_model)
        
        # Trainable Q-Former
        self.qformer = Blip2QFormerWrapper(blip2.qformer, blip2.query_tokens)
        
        del blip2
        torch.cuda.empty_cache()
        
        # Dimensions
        vit_dim = self.vision_encoder.hidden_size  # 1408
        qformer_dim = self.qformer.hidden_size      # 768
        num_queries = self.qformer.num_queries      # 32
        
        # Hierarchical Patch Refinement
        self.hpa = HybridPatchAnnealing(
            n_max=hpa_n_max,
            n_min=hpa_n_min,
            t_steps=hpa_t_steps,
            w=hpa_w,
            p1=hpa_p1,
            p2=hpa_p2,
        )
        
        # Query-Patch Attention
        self.query_patch_attn = QueryPatchAttention(
            query_dim=qformer_dim,
            patch_dim=vit_dim,
        )
        
        # Mahalanobis Scoring
        self.mahalanobis_scorer = MahalanobisScoring(
            feature_dim=vit_dim,
            regularization=mahalanobis_reg,
            gamma=mahalanobis_gamma,
        )
        
        # Normal Statistics Tracker
        self.normal_stats_tracker = NormalStatisticsTracker(
            feature_dim=vit_dim,
            buffer_size=normal_stats_buffer_size,
            update_frequency=normal_stats_update_frequency,
        )
        
        # LLM Projection
        self.projection = LLMProjection(
            input_dim=qformer_dim,
            output_dim=llm_embed_dim,
        )
        
        # Scoring weights
        self.score_alpha = score_alpha
        self.score_beta = score_beta
        self.score_lambda = score_lambda
        
    def forward(
        self,
        images: torch.Tensor,
        patch_labels: torch.Tensor | None = None,
        update_stats: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: (B, 3, H, W) normalized image batch.
            patch_labels: (B, N) binary patch labels (for statistics update).
            update_stats: Whether to update normal statistics.
            
        Returns:
            dict with keys:
                patch_scores: (B, N) final patch anomaly scores.
                query_embeds: (B, Q, D_qformer) Q-Former outputs.
                visual_tokens: (B, Q, D_llm) projected tokens for LLM.
        """
        B = images.shape[0]
        
        # 1. Extract ViT patch embeddings
        image_embeds = self.vision_encoder(images)  # (B, N+1, D_v)
        patch_embeds = image_embeds[:, 1:, :].float()  # (B, N, D_v) - drop CLS
        cls_token = image_embeds[:, 0:1, :].float()  # (B, 1, D_v) - keep CLS
        
        # 2. Initial Q-Former: queries attend to all patches (including CLS)
        query_tokens = self.qformer.query_tokens.expand(B, -1, -1)  # (B, Q, D_q)
        image_atts = torch.ones(
            image_embeds.size()[:-1],
            dtype=torch.long,
            device=image_embeds.device,
        )
        initial_query_embeds = self.qformer.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds.float(),
            encoder_attention_mask=image_atts,
        ).last_hidden_state  # (B, Q, D_q)
        
        # 3. Update normal statistics if training
        if update_stats and patch_labels is not None:
            self.normal_stats_tracker.add_normal_patches(patch_embeds, patch_labels)
            # Periodically update Mahalanobis statistics
            if self.normal_stats_tracker.step_count % self.normal_stats_tracker.update_frequency == 0:
                mu, sigma = self.normal_stats_tracker.get_statistics()
                if mu is not None:
                    normal_patches = torch.stack(list(self.normal_stats_tracker.normal_patch_buffer))
                    # Move to same device as model
                    normal_patches = normal_patches.to(patch_embeds.device)
                    self.mahalanobis_scorer.update_statistics(normal_patches)
                    # Clear cache after statistics update
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            self.normal_stats_tracker.step_count += 1
        
        # 4. Hierarchical Patch Refinement (HPA) with attention re-computation
        # Clear cache before refinement to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        refined_patches, selected_indices = self.hpa(
            patch_embeds=patch_embeds,
            qformer=self.qformer.qformer,  # Pass Q-Former module
            query_tokens=query_tokens,  # Pass query tokens
            query_patch_attn=self.query_patch_attn,
            mahalanobis_scorer=self.mahalanobis_scorer,
            cls_token=cls_token,
            alpha=self.score_alpha,
            beta=self.score_beta,
        )  # (B, N_min, D), (B, N_min)
        
        # Clear cache after refinement
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 5. Final scoring: refined queries attend to ALL 256 patches
        # Re-compute queries on refined patches one more time
        refined_image_embeds = torch.cat([cls_token, refined_patches], dim=1)  # (B, N_min+1, D)
        refined_image_atts = torch.ones(
            refined_image_embeds.size()[:-1],
            dtype=torch.long,
            device=refined_image_embeds.device,
        )
        
        final_query_embeds = self.qformer.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=refined_image_embeds,
            encoder_attention_mask=refined_image_atts,
        ).last_hidden_state  # (B, Q, D_q)
        
        # Attend to ALL original 256 patches with refined queries
        final_attn_weights, final_attn_importance = self.query_patch_attn(
            final_query_embeds, patch_embeds  # All patches, not just refined
        )  # (B, N)
        
        # Final Mahalanobis scores on all patches
        final_deviation = self.mahalanobis_scorer(patch_embeds)  # (B, N)
        
        # Final patch scores with cross-term
        # Normalize attention importance to [0, 1] range to prevent extreme values
        # Attention importance can be very high initially (sum over queries, each in [0,1])
        # With 32 queries, max attention_importance = 32, which is too high
        attn_importance_norm = final_attn_importance / (self.qformer.num_queries + 1e-8)  # Normalize by num queries
        
        patch_scores = (
            self.score_alpha * attn_importance_norm +
            self.score_beta * final_deviation +
            self.score_lambda * (attn_importance_norm * final_deviation)
        )  # (B, N)
        
        # 6. Project for LLM (use initial query embeds for consistency)
        visual_tokens = self.projection(initial_query_embeds)
        
        return {
            "patch_scores": patch_scores,
            "query_embeds": initial_query_embeds,  # Return initial queries
            "visual_tokens": visual_tokens,
        }
    
    def get_image_score(self, patch_scores: torch.Tensor) -> torch.Tensor:
        """Aggregate patch scores into image-level score using top-3 mean.
        
        Args:
            patch_scores: (B, N) patch anomaly scores.
            
        Returns:
            (B,) image-level anomaly scores.
        """
        # Get top-3 patch scores per image
        top3_scores, _ = torch.topk(patch_scores, k=min(3, patch_scores.shape[1]), dim=1)
        image_scores = top3_scores.mean(dim=1)  # (B,)
        return image_scores
