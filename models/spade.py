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
from utils.debug_logger import get_debug_logger

# Initialize debug logger
_debug_logger = get_debug_logger("spade_debug")


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
        
        # HPA enabled flag (can be set via enable_hpa/disable_hpa)
        self.use_hpa = True  # Default enabled
        
        # Frequency feature extractor (optional, enabled via enable_frequency_features)
        self.use_frequency = False
        self.freq_extractor = None
        self.freq_mahalanobis_scorer = None
        self.freq_normal_stats_tracker = None
        self.score_gamma = 0.25  # Default, can be set via enable_frequency_features
        
    def forward(
        self,
        images: torch.Tensor,
        patch_labels: torch.Tensor | None = None,
        update_stats: bool = False,
    ) -> dict[str, torch.Tensor]:
        # ⚠️ CRITICAL: Log training mode and HPA status
        _debug_logger.debug(f"[SPADE DEBUG] Forward pass - training={self.training}, use_hpa={self.use_hpa}")
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
        
        # ═══════════════════════════════════════════════
        # STREAM 2: FREQUENCY FEATURES (parallel)
        # ═══════════════════════════════════════════════
        freq_scores = None
        
        if self.use_frequency and self.freq_extractor is not None:
            from utils.patch_extraction import extract_image_patches_from_tensor
            
            # Extract 14×14 patches
            image_patches = extract_image_patches_from_tensor(images, patch_size=14)
            # (B*256, 14, 14, 3) uint8 numpy
            
            # Convert to tensor
            image_patches_tensor = torch.from_numpy(image_patches).to(images.device)
            
            # Extract frequency features
            freq_features = self.freq_extractor(image_patches_tensor)
            # (B*256, freq_feature_dim)
            
            # Reshape to batch structure
            freq_feature_dim = freq_features.shape[-1]
            freq_features = freq_features.reshape(B, 256, freq_feature_dim)  # (B, 256, freq_feature_dim)
            
            # Update frequency statistics if training
            if update_stats and patch_labels is not None:
                self.freq_normal_stats_tracker.add_normal_patches(freq_features, patch_labels)
                if self.freq_normal_stats_tracker.step_count % self.freq_normal_stats_tracker.update_frequency == 0:
                    mu, sigma = self.freq_normal_stats_tracker.get_statistics()
                    if mu is not None:
                        normal_freq = torch.stack(list(self.freq_normal_stats_tracker.normal_patch_buffer))
                        normal_freq = normal_freq.to(freq_features.device)
                        self.freq_mahalanobis_scorer.update_statistics(normal_freq)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                self.freq_normal_stats_tracker.step_count += 1
            
            # Compute frequency Mahalanobis scores
            freq_scores = self.freq_mahalanobis_scorer(freq_features)  # (B, 256)
        
        # 4. Hierarchical Patch Refinement (HPA) with attention re-computation
        # DEBUG: Store initial query stats
        initial_query_mean = initial_query_embeds.mean().item()
        initial_query_std = initial_query_embeds.std().item()
        initial_query_norm = initial_query_embeds.norm(dim=-1).mean().item()
        
        # ⚠️ CRITICAL VERIFICATION: Check if use_hpa is actually being respected
        _debug_logger.debug(f"[SPADE DEBUG] ═══ FORWARD PASS START ═══")
        _debug_logger.debug(f"[SPADE DEBUG] HPA enabled: {self.use_hpa} (type: {type(self.use_hpa)})")
        _debug_logger.debug(f"[SPADE DEBUG] Initial queries - mean={initial_query_mean:.4f}, "
              f"std={initial_query_std:.4f}, norm={initial_query_norm:.4f}")
        
        # ⚠️ CRITICAL: Double-check use_hpa is actually a boolean True/False
        if not isinstance(self.use_hpa, bool):
            _debug_logger.error(f"[SPADE DEBUG] ❌ CRITICAL BUG: self.use_hpa is not a boolean! Value: {self.use_hpa}, type: {type(self.use_hpa)}")
            # Force it to be a boolean
            self.use_hpa = bool(self.use_hpa)
        
        if self.use_hpa:
            _debug_logger.debug(f"[SPADE DEBUG] ═══ HPA ENABLED - Entering refinement ═══")
            _debug_logger.debug(f"[SPADE DEBUG] ✅ TAKING HPA ENABLED BRANCH - self.use_hpa={self.use_hpa}")
            
            # DEBUG: Compute attention with initial queries BEFORE HPA (for comparison)
            initial_attn_weights, initial_attn_importance = self.query_patch_attn(
                initial_query_embeds, patch_embeds
            )
            initial_attn_mean = initial_attn_importance.mean().item()
            _debug_logger.debug(f"[SPADE DEBUG] Initial attention (before HPA) - mean={initial_attn_mean:.4f}")
            
            # Clear cache before refinement to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            refined_patches, selected_indices, final_refined_queries = self.hpa(
                patch_embeds=patch_embeds,
                qformer=self.qformer.qformer,  # Pass Q-Former module
                query_tokens=query_tokens,  # Pass query tokens
                query_patch_attn=self.query_patch_attn,
                mahalanobis_scorer=self.mahalanobis_scorer,
                cls_token=cls_token,
                alpha=self.score_alpha,
                beta=self.score_beta,
                # NEW: Frequency scoring
                freq_scores=freq_scores,  # Pass precomputed scores
                gamma=self.score_gamma,
            )  # (B, N_min, D), (B, N_min), (B, Q, D_q)
            
            # DEBUG: Compare initial vs refined queries
            query_diff = (final_refined_queries - initial_query_embeds).abs().mean().item()
            final_query_mean = final_refined_queries.mean().item()
            final_query_std = final_refined_queries.std().item()
            final_query_norm = final_refined_queries.norm(dim=-1).mean().item()
            
            _debug_logger.debug(f"[SPADE DEBUG] ═══ HPA COMPLETED ═══")
            _debug_logger.debug(f"[SPADE DEBUG] Final queries - mean={final_query_mean:.4f}, "
                  f"std={final_query_std:.4f}, norm={final_query_norm:.4f}")
            _debug_logger.debug(f"[SPADE DEBUG] Query change (final - initial): {query_diff:.6f}")
            _debug_logger.debug(f"[SPADE DEBUG] Mean change: {abs(final_query_mean - initial_query_mean):.6f}")
            _debug_logger.debug(f"[SPADE DEBUG] Selected {selected_indices.shape[1]} patches (n_min={self.hpa.n_min})")
            
            # Clear cache after refinement
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 5. Final scoring: Use accumulated refined queries to attend to ALL 256 patches
            # No need to recompute - use the queries returned from HPA (already refined through all steps)
            # DEBUG: Verify queries are actually different
            query_diff_final = (final_refined_queries - initial_query_embeds).abs().mean().item()
            _debug_logger.debug(f"[SPADE DEBUG] Using REFINED queries for final scoring (diff from initial: {query_diff_final:.6f})")
            final_attn_weights, final_attn_importance = self.query_patch_attn(
                final_refined_queries, patch_embeds  # ✅ Use accumulated refined queries on all patches
            )  # (B, N)
            
            # DEBUG: Compare final attention with initial attention
            final_attn_mean = final_attn_importance.mean().item()
            attn_diff = (final_attn_importance - initial_attn_importance).abs().mean().item()
            _debug_logger.debug(f"[SPADE DEBUG] Final attention (after HPA) - mean={final_attn_mean:.4f}, diff from initial={attn_diff:.6f}")
        else:
            _debug_logger.debug(f"[SPADE DEBUG] ═══ HPA DISABLED - Using initial queries directly ═══")
            _debug_logger.debug(f"[SPADE DEBUG] ✅ TAKING HPA DISABLED BRANCH - self.use_hpa={self.use_hpa}")
            # HPA disabled: Use initial query embeddings directly to score all patches
            # 5. Final scoring: Use initial queries to attend to ALL 256 patches
            _debug_logger.debug(f"[SPADE DEBUG] Using INITIAL queries for final scoring (no refinement)")
            final_attn_weights, final_attn_importance = self.query_patch_attn(
                initial_query_embeds, patch_embeds  # ✅ Use initial queries on all patches (no refinement)
            )  # (B, N)
            _debug_logger.debug(f"[SPADE DEBUG] Using initial queries (no refinement)")
            _debug_logger.debug(f"[SPADE DEBUG] Initial queries - mean={initial_query_mean:.4f}, "
                  f"std={initial_query_std:.4f}, norm={initial_query_norm:.4f}")
        
        # Final Mahalanobis scores on all patches (RAW - no normalization)
        final_spatial_mahal = self.mahalanobis_scorer(patch_embeds)  # (B, N)
        
        # Final frequency Mahalanobis on all patches (RAW - no normalization)
        if freq_scores is not None:
            final_freq_mahal = freq_scores  # Already computed for all 256 patches
        else:
            final_freq_mahal = torch.zeros_like(final_spatial_mahal)
        
        # DEBUG: Compare attention importance
        attn_mean = final_attn_importance.mean().item()
        attn_std = final_attn_importance.std().item()
        attn_max = final_attn_importance.max().item()
        attn_min = final_attn_importance.min().item()
        _debug_logger.debug(f"[SPADE DEBUG] Final attention importance - mean={attn_mean:.4f}, "
              f"std={attn_std:.4f}, max={attn_max:.4f}, min={attn_min:.4f}")
        
        # DEBUG: Compare spatial and frequency Mahalanobis (RAW values)
        spatial_mean = final_spatial_mahal.mean().item()
        spatial_std = final_spatial_mahal.std().item()
        spatial_max = final_spatial_mahal.max().item()
        spatial_min = final_spatial_mahal.min().item()
        _debug_logger.debug(f"[SPADE DEBUG] Final spatial Mahalanobis - RAW: mean={spatial_mean:.4f}, "
              f"std={spatial_std:.4f}, max={spatial_max:.4f}, min={spatial_min:.4f}")
        
        if freq_scores is not None:
            freq_mean = final_freq_mahal.mean().item()
            freq_std = final_freq_mahal.std().item()
            freq_max = final_freq_mahal.max().item()
            freq_min = final_freq_mahal.min().item()
            _debug_logger.debug(f"[SPADE DEBUG] Final frequency Mahalanobis - RAW: mean={freq_mean:.4f}, "
                  f"std={freq_std:.4f}, max={freq_max:.4f}, min={freq_min:.4f}")
        
        # Final patch scores with cross-term
        # Normalize attention importance to [0, 1] range to prevent extreme values
        # Attention importance can be very high initially (sum over queries, each in [0,1])
        # With 32 queries, max attention_importance = 32, which is too high
        attn_importance_norm = final_attn_importance / (self.qformer.num_queries + 1e-8)  # Normalize by num queries
        
        # ⭐ COMBINED FINAL SCORES (spatial + frequency) - Using RAW Mahalanobis scores
        patch_scores = (
            self.score_alpha * attn_importance_norm +
            self.score_beta * final_spatial_mahal +
            self.score_gamma * final_freq_mahal +
            self.score_lambda * (attn_importance_norm * final_spatial_mahal)
        )  # (B, N)
        
        # DEBUG: Final patch scores statistics
        patch_mean = patch_scores.mean().item()
        patch_std = patch_scores.std().item()
        patch_max = patch_scores.max().item()
        patch_min = patch_scores.min().item()
        _debug_logger.debug(f"[SPADE DEBUG] Final patch scores - mean={patch_mean:.4f}, "
              f"std={patch_std:.4f}, max={patch_max:.4f}, min={patch_min:.4f}")
        
        # DEBUG: Log first few patch scores to verify they're actually different
        # Always compute hash (works for any batch size)
        import hashlib
        patch_scores_bytes = patch_scores.detach().cpu().numpy().tobytes()
        patch_scores_hash = hashlib.md5(patch_scores_bytes).hexdigest()[:16]
        _debug_logger.debug(f"[SPADE DEBUG] ⚠️ CRITICAL: Patch scores hash: {patch_scores_hash}")
        
        # Log first batch's first 5 patches
        if patch_scores.shape[0] >= 1:
            first_5_scores = patch_scores[0, :5].cpu().tolist()
            _debug_logger.debug(f"[SPADE DEBUG] First 5 patch scores: {[f'{s:.6f}' for s in first_5_scores]}")
            # Also log attention and mahal components for first patch (using RAW Mahalanobis)
            first_attn = attn_importance_norm[0, 0].item()
            first_mahal_raw = final_spatial_mahal[0, 0].item()
            first_freq_raw = final_freq_mahal[0, 0].item() if freq_scores is not None else 0.0
            first_cross = (attn_importance_norm * final_spatial_mahal)[0, 0].item()
            first_score = patch_scores[0, 0].item()
            _debug_logger.debug(f"[SPADE DEBUG] First patch breakdown - "
                  f"attn_contrib={self.score_alpha * first_attn:.6f} (alpha={self.score_alpha} * {first_attn:.6f}), "
                  f"mahal_contrib={self.score_beta * first_mahal_raw:.6f} (beta={self.score_beta} * RAW={first_mahal_raw:.2f}), "
                  f"freq_contrib={self.score_gamma * first_freq_raw:.6f} (gamma={self.score_gamma} * RAW={first_freq_raw:.2f}), "
                  f"cross_contrib={self.score_lambda * first_cross:.6f} (lambda={self.score_lambda} * {first_cross:.6f}), "
                  f"TOTAL={first_score:.6f}")
        
        # DEBUG: Save query statistics for comparison (optional)
        if hasattr(self, '_debug_save_stats') and self._debug_save_stats:
            import json
            import os
            from datetime import datetime
            
            stats = {
                'hpa_enabled': self.use_hpa,
                'timestamp': datetime.now().isoformat(),
                'initial_queries': {
                    'mean': initial_query_mean,
                    'std': initial_query_std,
                    'norm': initial_query_norm,
                },
            }
            
            if self.use_hpa:
                # final_query_mean, final_query_std, etc. are defined in the HPA enabled branch
                stats['final_queries'] = {
                    'mean': final_query_mean,
                    'std': final_query_std,
                    'norm': final_query_norm,
                    'change_from_initial': query_diff,
                }
            else:
                # When disabled, final queries = initial queries (no refinement)
                stats['final_queries'] = {
                    'mean': initial_query_mean,
                    'std': initial_query_std,
                    'norm': initial_query_norm,
                    'change_from_initial': 0.0,
                }
            
            stats['attention'] = {
                'mean': attn_mean,
                'std': attn_std,
                'max': attn_max,
            }
            
            stats['patch_scores'] = {
                'mean': patch_scores.mean().item(),
                'std': patch_scores.std().item(),
                'max': patch_scores.max().item(),
                'min': patch_scores.min().item(),
            }
            
            # Save to file
            debug_dir = "debug_stats"
            os.makedirs(debug_dir, exist_ok=True)
            filename = f"query_stats_{'hpa_enabled' if self.use_hpa else 'hpa_disabled'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(debug_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(stats, f, indent=2)
            _debug_logger.debug(f"[SPADE DEBUG] Saved stats to {filepath}")
        
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
        top3_scores, top3_indices = torch.topk(patch_scores, k=min(3, patch_scores.shape[1]), dim=1)
        image_scores = top3_scores.mean(dim=1)  # (B,)
        
        # DEBUG: Log top-3 patch indices and scores
        from utils.debug_logger import get_debug_logger
        _debug_logger = get_debug_logger("spade_debug")
        if patch_scores.shape[0] == 1:  # Only log for single image batches to avoid spam
            top3_vals = top3_scores[0].cpu().tolist()
            top3_idx = top3_indices[0].cpu().tolist()
            _debug_logger.debug(f"[SPADE DEBUG] Image score: {image_scores[0].item():.4f}, "
                  f"top3_indices: {top3_idx}, top3_scores: {[f'{v:.4f}' for v in top3_vals]}")
        
        return image_scores
    
    def enable_frequency_features(
        self,
        freq_num_bands: int = 6,
        freq_use_phase: bool = True,
        freq_feature_dim: int = 32,
        score_gamma: float = 0.25,
    ):
        """Enable frequency feature extraction (call after __init__ or from config).
        
        Args:
            freq_num_bands: Number of frequency bands to extract.
            freq_use_phase: Whether to include phase information.
            freq_feature_dim: Output feature dimension.
            score_gamma: Weight for frequency Mahalanobis in scoring.
        """
        from models.frequency_features import FourierPatchFeatureExtractor
        
        self.use_frequency = True
        self.score_gamma = score_gamma
        
        self.freq_extractor = FourierPatchFeatureExtractor(
            num_freq_bands=freq_num_bands,
            use_phase=freq_use_phase,
            feature_dim=freq_feature_dim,
        )
        
        # Separate Mahalanobis scorer for frequency
        self.freq_mahalanobis_scorer = MahalanobisScoring(
            feature_dim=freq_feature_dim,
            regularization=self.mahalanobis_scorer.regularization,
            gamma=self.mahalanobis_scorer.gamma,
        )
        
        # Frequency statistics tracker
        self.freq_normal_stats_tracker = NormalStatisticsTracker(
            feature_dim=freq_feature_dim,
            buffer_size=self.normal_stats_tracker.buffer_size,
            update_frequency=self.normal_stats_tracker.update_frequency,
        )
