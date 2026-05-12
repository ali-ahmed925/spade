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
from models.gmm_scorer import GMMScorer
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
        score_gamma: float = 0.25,  # Frequency weight (moved from enable_frequency_features)
        score_lambda: float = 0.2,
        mahalanobis_gamma: float = 1.3,
        mahalanobis_reg: float = 1e-6,
        use_mahalanobis: bool = True,  # Enable/disable Mahalanobis scoring
        # Normal statistics parameters
        normal_stats_buffer_size: int = 10000,
        normal_stats_update_frequency: int = 100,
        # Attention verification
        verify_attention: bool = False,  # Enable attention verification (logs and visualizes)
        # Raw attention mode
        use_raw_attention: bool = True,  # Use raw attention scores (larger scale, handles negatives)
        # Image-level score from patch scores (see get_image_score)
        image_score_mode: str = "quantile",
        image_score_quantile: float = 0.99,
        image_score_top_k: int = 3,
        # GMM (optional, replaces single-Gaussian Mahalanobis when enabled)
        use_gmm: bool = False,
        gmm_spatial_components: int = 8,
        gmm_freq_components: int = 5,
        gmm_covariance_type: str = "full",
        gmm_reg_covar: float = 1e-4,
        gmm_fit_max_samples: int | None = 15000,
        gmm_fit_subsample_seed: int = 42,
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
        
        # Density Scoring (GMM or single-Gaussian Mahalanobis)
        self.use_gmm = bool(use_gmm) and use_mahalanobis
        self._gmm_freq_n = int(gmm_freq_components)
        self._gmm_covariance_type = str(gmm_covariance_type)
        self._gmm_reg_covar = float(gmm_reg_covar)
        self._gmm_fit_max_samples = None if gmm_fit_max_samples in (None, 0, "0") else int(gmm_fit_max_samples)
        self._gmm_fit_subsample_seed = int(gmm_fit_subsample_seed)

        self.gmm_scorer: GMMScorer | None = None
        self.mahalanobis_scorer: MahalanobisScoring | None = None
        if self.use_gmm:
            self.gmm_scorer = GMMScorer(
                feature_dim=vit_dim,
                n_components=int(gmm_spatial_components),
                covariance_type=self._gmm_covariance_type,
                reg_covar=self._gmm_reg_covar,
                gamma=mahalanobis_gamma,
                fit_max_samples=self._gmm_fit_max_samples,
                fit_subsample_seed=self._gmm_fit_subsample_seed,
            )
        else:
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
        
        # Scoring weights - Learnable parameters (will be optimized during training)
        self.score_alpha = nn.Parameter(torch.tensor(score_alpha))
        self.score_beta = nn.Parameter(torch.tensor(score_beta))
        self.score_gamma = nn.Parameter(torch.tensor(score_gamma))
        self.score_lambda = nn.Parameter(torch.tensor(score_lambda))
        
        # Mahalanobis enabled flag
        self.use_mahalanobis = use_mahalanobis
        
        # HPA enabled flag (can be set via enable_hpa/disable_hpa)
        self.use_hpa = True  # Default enabled
        
        # Attention verification flag
        self.verify_attention = verify_attention
        self._verification_run_this_epoch = False  # Track if verification has run this epoch
        
        # Raw attention mode: use raw scores instead of softmax probabilities
        self.use_raw_attention = use_raw_attention  # Enable raw attention scores (larger scale, handles negatives)
        
        self.image_score_mode = image_score_mode
        self.image_score_quantile = image_score_quantile
        self.image_score_top_k = image_score_top_k
        
        if self.use_raw_attention:
            _debug_logger.info(f"✅ Raw attention mode ENABLED - using raw scores (before softmax) with ReLU+shift for negatives")
        else:
            _debug_logger.info(f"📊 Softmax attention mode - using probability-based attention (original approach)")
        
        # Frequency feature extractor (optional, enabled via enable_frequency_features)
        self.use_frequency = False
        self.freq_extractor = None
        self.freq_gmm_scorer: GMMScorer | None = None
        self.freq_mahalanobis_scorer = None
        self.freq_normal_stats_tracker = None
        
        # Running statistics for standardization (computed from training data)
        # These accumulate during training and are used for fixed normalization
        self.register_buffer('attn_mean', torch.tensor(0.0))
        self.register_buffer('attn_std', torch.tensor(1.0))
        self.register_buffer('spatial_mean', torch.tensor(0.0))
        self.register_buffer('spatial_std', torch.tensor(1.0))
        self.register_buffer('freq_mean', torch.tensor(0.0))
        self.register_buffer('freq_std', torch.tensor(1.0))
        self.register_buffer('_stats_count', torch.tensor(0))  # Number of batches used for statistics

    def _spatial_density_scorer(self) -> GMMScorer | MahalanobisScoring | None:
        if not self.use_mahalanobis:
            return None
        return self.gmm_scorer if self.use_gmm else self.mahalanobis_scorer

    def _freq_density_scorer(self) -> GMMScorer | MahalanobisScoring | None:
        if not self.use_mahalanobis or not self.use_frequency:
            return None
        return self.freq_gmm_scorer if self.use_gmm else self.freq_mahalanobis_scorer

    def _attention_component(
        self,
        attention_importance: torch.Tensor,
        raw_attention_scores: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return attention-derived anomaly component (higher = more anomalous)."""
        if self.use_raw_attention and raw_attention_scores is not None:
            attention_anomaly = -raw_attention_scores
            attention_min = attention_anomaly.min(dim=1, keepdim=True)[0]
            return attention_anomaly - attention_min
        return attention_importance

    def _standardize_components(
        self,
        attn_component: torch.Tensor,
        spatial_mahal: torch.Tensor,
        freq_mahal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared transform + standardization path used by HPA shortlist and final scoring."""
        spatial_energy = torch.log1p(torch.clamp_min(spatial_mahal, 0.0))
        freq_energy = torch.log1p(torch.clamp_min(freq_mahal, 0.0))

        if self.training:
            batch_attn_mean = attn_component.mean()
            batch_attn_std = torch.clamp(attn_component.std(), min=1e-6)
            batch_spatial_mean = spatial_energy.mean()
            batch_spatial_std = torch.clamp(spatial_energy.std(), min=1e-6)
            batch_freq_mean = freq_energy.mean()
            batch_freq_std = torch.clamp(freq_energy.std(), min=1e-6)

            momentum = 0.1
            self.attn_mean.mul_(1 - momentum).add_(momentum * batch_attn_mean.detach())
            self.attn_std.mul_(1 - momentum).add_(momentum * batch_attn_std.detach())
            self.spatial_mean.mul_(1 - momentum).add_(momentum * batch_spatial_mean.detach())
            self.spatial_std.mul_(1 - momentum).add_(momentum * batch_spatial_std.detach())
            self.freq_mean.mul_(1 - momentum).add_(momentum * batch_freq_mean.detach())
            self.freq_std.mul_(1 - momentum).add_(momentum * batch_freq_std.detach())
            self._stats_count.add_(1)

        attn_std = (attn_component - self.attn_mean) / torch.clamp(self.attn_std, min=1e-6)
        spatial_std = (spatial_energy - self.spatial_mean) / torch.clamp(self.spatial_std, min=1e-6)
        freq_std = (freq_energy - self.freq_mean) / torch.clamp(self.freq_std, min=1e-6)

        attn_std = torch.where(torch.isfinite(attn_std), attn_std, torch.zeros_like(attn_std))
        spatial_std = torch.where(torch.isfinite(spatial_std), spatial_std, torch.zeros_like(spatial_std))
        freq_std = torch.where(torch.isfinite(freq_std), freq_std, torch.zeros_like(freq_std))
        return attn_std, spatial_std, freq_std, spatial_energy, freq_energy
        
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
        
        # 3. Collect normal patches for statistics (only if Mahalanobis is enabled)
        # Statistics will be computed once after collecting all patches
        if self.use_mahalanobis and update_stats and patch_labels is not None:
            self.normal_stats_tracker.add_normal_patches(patch_embeds, patch_labels)
        
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
            
            # Collect frequency normal patches for statistics (only if Mahalanobis is enabled)
            # Statistics will be computed once after collecting all patches
            if self.use_mahalanobis and update_stats and patch_labels is not None:
                self.freq_normal_stats_tracker.add_normal_patches(freq_features, patch_labels)
            
            # Compute frequency Mahalanobis scores (only if Mahalanobis is enabled)
            if self.use_mahalanobis:
                freq_scorer = self._freq_density_scorer()
                freq_scores = freq_scorer(freq_features) if freq_scorer is not None else None  # (B, 256)
            else:
                freq_scores = None  # Disabled
        
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
            if self.use_raw_attention or self.verify_attention:
                initial_attn_weights, initial_attn_importance, _ = self.query_patch_attn(
                    initial_query_embeds, patch_embeds, return_raw_scores=True
                )
            else:
                initial_attn_weights, initial_attn_importance = self.query_patch_attn(
                    initial_query_embeds, patch_embeds
                )
            initial_attn_mean = initial_attn_importance.mean().item()
            _debug_logger.debug(f"[SPADE DEBUG] Initial attention (before HPA) - mean={initial_attn_mean:.4f}")
            
            # Clear cache before refinement to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Use density scorer (GMM or Mahalanobis), or None when disabled
            mahalanobis_scorer_for_hpa = self._spatial_density_scorer() if self.use_mahalanobis else None
            
            refined_patches, selected_indices, final_refined_queries = self.hpa(
                patch_embeds=patch_embeds,
                qformer=self.qformer.qformer,  # Pass Q-Former module
                query_tokens=query_tokens,  # Pass query tokens
                query_patch_attn=self.query_patch_attn,
                mahalanobis_scorer=mahalanobis_scorer_for_hpa,
                cls_token=cls_token,
                alpha=self.score_alpha.item(),  # Convert learnable parameter to float for HPA
                beta=self.score_beta.item(),
                # NEW: Frequency scoring
                freq_scores=freq_scores if self.use_mahalanobis else None,  # Disable freq scores if Mahalanobis disabled
                gamma=self.score_gamma.item(),
                use_raw_attention=self.use_raw_attention,
                attn_mean=float(self.attn_mean.item()),
                attn_std=float(self.attn_std.item()),
                spatial_mean=float(self.spatial_mean.item()),
                spatial_std=float(self.spatial_std.item()),
                freq_mean=float(self.freq_mean.item()),
                freq_std=float(self.freq_std.item()),
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
            # Always get raw scores if using raw attention mode or verification
            if self.use_raw_attention or self.verify_attention:
                final_attn_weights, final_attn_importance, raw_attention_scores = self.query_patch_attn(
                    final_refined_queries, patch_embeds, return_raw_scores=True
                )  # (B, N), (B, N), (B, N)
            else:
                final_attn_weights, final_attn_importance = self.query_patch_attn(
                    final_refined_queries, patch_embeds  # ✅ Use accumulated refined queries on all patches
                )  # (B, N)
                raw_attention_scores = None
            
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
            # Always get raw scores if using raw attention mode or verification
            if self.use_raw_attention or self.verify_attention:
                final_attn_weights, final_attn_importance, raw_attention_scores = self.query_patch_attn(
                    initial_query_embeds, patch_embeds, return_raw_scores=True
                )  # (B, N), (B, N), (B, N)
            else:
                final_attn_weights, final_attn_importance = self.query_patch_attn(
                    initial_query_embeds, patch_embeds  # ✅ Use initial queries on all patches (no refinement)
                )  # (B, N)
                raw_attention_scores = None
            _debug_logger.debug(f"[SPADE DEBUG] Using initial queries (no refinement)")
            _debug_logger.debug(f"[SPADE DEBUG] Initial queries - mean={initial_query_mean:.4f}, "
                  f"std={initial_query_std:.4f}, norm={initial_query_norm:.4f}")
        
        # Final Mahalanobis scores on all patches (RAW - no normalization)
        if self.use_mahalanobis:
            spatial_scorer = self._spatial_density_scorer()
            final_spatial_mahal = spatial_scorer(patch_embeds) if spatial_scorer is not None else torch.zeros(
                patch_embeds.shape[0], patch_embeds.shape[1], device=patch_embeds.device, dtype=patch_embeds.dtype
            )  # (B, N)
        else:
            final_spatial_mahal = torch.zeros_like(patch_embeds[:, :, 0])  # (B, N) - zeros when disabled
        
        # Final frequency Mahalanobis on all patches (RAW - no normalization)
        if self.use_mahalanobis and freq_scores is not None:
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
        
        # Final patch scores: use the same anomaly-component pipeline as HPA shortlist.
        attn_importance_norm = self._attention_component(final_attn_importance, raw_attention_scores)
        attn_std, spatial_std, freq_std, spatial_energy, freq_energy = self._standardize_components(
            attn_component=attn_importance_norm,
            spatial_mahal=final_spatial_mahal,
            freq_mahal=final_freq_mahal,
        )

        if self.use_raw_attention and raw_attention_scores is not None:
            _debug_logger.debug(
                f"[SPADE DEBUG] Using RAW attention scores - raw mean={raw_attention_scores.mean().item():.4f}, "
                f"transformed mean={attn_importance_norm.mean().item():.4f}, "
                f"transformed max={attn_importance_norm.max().item():.4f}"
            )
        else:
            _debug_logger.debug(f"[SPADE DEBUG] Using SOFTMAX attention (original approach)")

        # DEBUG: Log energy values (after log-transform)
        spatial_energy_mean = spatial_energy.mean().item()
        spatial_energy_std = spatial_energy.std().item()
        _debug_logger.debug(f"[SPADE DEBUG] Spatial energy (log1p) - mean={spatial_energy_mean:.4f}, std={spatial_energy_std:.4f}")
        if freq_scores is not None:
            freq_energy_mean = freq_energy.mean().item()
            freq_energy_std = freq_energy.std().item()
            _debug_logger.debug(f"[SPADE DEBUG] Frequency energy (log1p) - mean={freq_energy_mean:.4f}, std={freq_energy_std:.4f}")
        
        # DEBUG: Log standardized values and running statistics
        mode_str = "TRAINING (updating stats)" if self.training else "EVAL (using fixed stats)"
        _debug_logger.debug(f"[SPADE DEBUG] Standardized values ({mode_str}) - "
              f"attn: mean={attn_std.mean().item():.4f}, std={attn_std.std().item():.4f} "
              f"(running: μ={self.attn_mean.item():.4f}, σ={self.attn_std.item():.4f}), "
              f"spatial: mean={spatial_std.mean().item():.4f}, std={spatial_std.std().item():.4f} "
              f"(running: μ={self.spatial_mean.item():.4f}, σ={self.spatial_std.item():.4f}), "
              f"freq: mean={freq_std.mean().item():.4f}, std={freq_std.std().item():.4f} "
              f"(running: μ={self.freq_mean.item():.4f}, σ={self.freq_std.item():.4f})")
        
        # 🔥 Step 3: Combine with learnable weights (all modalities now in same scale)
        # No cross-term needed after standardization - all are already in comparable ranges
        patch_scores = (
            self.score_alpha * attn_std +
            self.score_beta * spatial_std +
            self.score_gamma * freq_std
        )  # (B, N)
        
        # Safety check: Replace any NaN or Inf with zeros
        patch_scores = torch.where(torch.isfinite(patch_scores), patch_scores, torch.zeros_like(patch_scores))
        
        # DEBUG: Log learnable weights
        _debug_logger.debug(f"[SPADE DEBUG] Learnable weights - alpha={self.score_alpha.item():.4f}, "
              f"beta={self.score_beta.item():.4f}, gamma={self.score_gamma.item():.4f}")
        
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
            # Also log attention and mahal components for first patch (standardized values)
            first_attn_std = attn_std[0, 0].item()
            first_spatial_std = spatial_std[0, 0].item()
            first_freq_std = freq_std[0, 0].item() if freq_scores is not None else 0.0
            first_score = patch_scores[0, 0].item()
            
            # Log raw values for reference
            first_attn_raw = attn_importance_norm[0, 0].item()
            first_spatial_raw = final_spatial_mahal[0, 0].item()
            first_freq_raw = final_freq_mahal[0, 0].item() if freq_scores is not None else 0.0
            first_spatial_energy = spatial_energy[0, 0].item()
            
            _debug_logger.debug(f"[SPADE DEBUG] First patch breakdown - "
                  f"attn_contrib={self.score_alpha.item() * first_attn_std:.6f} "
                  f"(alpha={self.score_alpha.item():.4f} * std={first_attn_std:.4f}, raw={first_attn_raw:.2f}), "
                  f"spatial_contrib={self.score_beta.item() * first_spatial_std:.6f} "
                  f"(beta={self.score_beta.item():.4f} * std={first_spatial_std:.4f}, energy={first_spatial_energy:.2f}, raw={first_spatial_raw:.2f}), "
                  f"freq_contrib={self.score_gamma.item() * first_freq_std:.6f} "
                  f"(gamma={self.score_gamma.item():.4f} * std={first_freq_std:.4f}, raw={first_freq_raw:.2f}), "
                  f"TOTAL={first_score:.6f}")
        
        # ═══════════════════════════════════════════════════════════════════
        # ATTENTION VERIFICATION (only if enabled and we have both normal and anomalous patches)
        # Run only once per epoch to avoid too many files
        # ═══════════════════════════════════════════════════════════════════
        if (self.verify_attention and raw_attention_scores is not None and patch_labels is not None 
            and not self._verification_run_this_epoch):
            # Only verify if we have both normal and anomalous patches
            has_normal = (patch_labels == 0).any()
            has_anomaly = (patch_labels == 1).any()
            if has_normal and has_anomaly:
                self._verify_attention_assumption(
                    raw_attention_scores=raw_attention_scores,
                    patch_labels=patch_labels,
                    final_spatial_mahal=final_spatial_mahal,
                    images=images,
                )
                self._verification_run_this_epoch = True  # Mark as run for this epoch
            elif self.training:
                # During training (normal-only), just log that we can't verify
                _debug_logger.debug("[ATTENTION VERIFY] Skipping verification - training data has only normal patches")
        
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
        """Aggregate patch scores into one score per image.
        
        Modes (``self.image_score_mode``):
          - ``quantile``: per-image ``q``-th quantile over all patches (default q=0.99).
          - ``max``: maximum patch score per image.
          - ``topk_mean``: mean of the top-``k`` patch scores (legacy behavior).
        
        Args:
            patch_scores: (B, N) patch anomaly scores.
            
        Returns:
            (B,) image-level anomaly scores.
        """
        from utils.debug_logger import get_debug_logger
        _dbg = get_debug_logger("spade_debug")
        _, n = patch_scores.shape
        mode = (self.image_score_mode or "quantile").lower()
        
        if mode == "quantile":
            q = float(self.image_score_quantile)
            q = max(0.0, min(1.0, q))
            image_scores = torch.quantile(patch_scores, q, dim=1)
            if patch_scores.shape[0] == 1:
                _dbg.debug(
                    f"[SPADE DEBUG] Image score (quantile q={q}): {image_scores[0].item():.4f}"
                )
        elif mode == "max":
            image_scores = patch_scores.max(dim=1).values
            if patch_scores.shape[0] == 1:
                _dbg.debug(f"[SPADE DEBUG] Image score (max): {image_scores[0].item():.4f}")
        elif mode in ("topk_mean", "top_k_mean"):
            k = min(int(self.image_score_top_k), n)
            k = max(1, k)
            topk_scores, topk_indices = torch.topk(patch_scores, k=k, dim=1)
            image_scores = topk_scores.mean(dim=1)
            if patch_scores.shape[0] == 1:
                topk_vals = topk_scores[0].cpu().tolist()
                topk_idx = topk_indices[0].cpu().tolist()
                _dbg.debug(
                    f"[SPADE DEBUG] Image score (top-{k} mean): {image_scores[0].item():.4f}, "
                    f"indices: {topk_idx}, scores: {[f'{v:.4f}' for v in topk_vals]}"
                )
        else:
            raise ValueError(
                f"Unknown image_score_mode={self.image_score_mode!r}; "
                "use 'quantile', 'max', or 'topk_mean'."
            )
        
        return image_scores
    
    @torch.no_grad()
    def compute_statistics_once(self, device: torch.device = None):
        """Compute Mahalanobis statistics once from all collected normal patches.
        
        This should be called after collecting all normal patches (e.g., after first epoch).
        Computes statistics for both spatial and frequency features if enabled.
        
        Args:
            device: Device to move patches to. If None, uses the device of the first parameter.
        """
        if not self.use_mahalanobis:
            return
        
        # Compute spatial statistics
        mu, sigma = self.normal_stats_tracker.compute_statistics_once()
        if mu is not None:
            normal_patches = torch.stack(list(self.normal_stats_tracker.normal_patch_buffer))
            if device is not None:
                normal_patches = normal_patches.to(device)
            else:
                # Use device of first model parameter
                device = next(self.parameters()).device
                normal_patches = normal_patches.to(device)
            
            if self.use_gmm and self.gmm_scorer is not None:
                self.gmm_scorer.fit(normal_patches)
            elif self.mahalanobis_scorer is not None:
                self.mahalanobis_scorer.update_statistics(normal_patches)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Compute frequency statistics if enabled
        if self.use_frequency and self.freq_normal_stats_tracker is not None:
            freq_mu, freq_sigma = self.freq_normal_stats_tracker.compute_statistics_once()
            if freq_mu is not None:
                normal_freq = torch.stack(list(self.freq_normal_stats_tracker.normal_patch_buffer))
                if device is None:
                    # Use device of first model parameter (already set above)
                    device = next(self.parameters()).device
                normal_freq = normal_freq.to(device)
                
                if self.use_gmm and self.freq_gmm_scorer is not None:
                    self.freq_gmm_scorer.fit(normal_freq)
                elif self.freq_mahalanobis_scorer is not None:
                    self.freq_mahalanobis_scorer.update_statistics(normal_freq)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def _verify_attention_assumption(
        self,
        raw_attention_scores: torch.Tensor,
        patch_labels: torch.Tensor,
        final_spatial_mahal: torch.Tensor,
        images: torch.Tensor,
    ) -> None:
        """Verify the assumption: positive attention = normal, negative attention = anomalous.
        
        Logs statistics and saves visualizations to verify if:
        - Normal patches have positive attention (on average)
        - Anomalous patches have negative attention (on average)
        - Attention correlates with Mahalanobis scores
        
        Args:
            raw_attention_scores: (B, N) raw attention scores (before softmax, summed across queries).
            patch_labels: (B, N) binary patch labels (0=normal, 1=anomalous).
            final_spatial_mahal: (B, N) Mahalanobis scores.
            images: (B, 3, H, W) original images for visualization.
        """
        import os
        from datetime import datetime
        import numpy as np
        
        # Create verification directory
        verify_dir = "attention_verification"
        os.makedirs(verify_dir, exist_ok=True)
        
        B, N = raw_attention_scores.shape
        
        # Flatten for analysis
        raw_attn_flat = raw_attention_scores.detach().cpu().flatten()  # (B*N,)
        labels_flat = patch_labels.detach().cpu().flatten()  # (B*N,)
        mahal_flat = final_spatial_mahal.detach().cpu().flatten()  # (B*N,)
        
        # Separate normal and anomalous patches
        normal_mask = labels_flat == 0
        anomaly_mask = labels_flat == 1
        
        if normal_mask.sum() > 0 and anomaly_mask.sum() > 0:
            # Statistics
            normal_attn_mean = raw_attn_flat[normal_mask].mean().item()
            normal_attn_std = raw_attn_flat[normal_mask].std().item()
            normal_attn_min = raw_attn_flat[normal_mask].min().item()
            normal_attn_max = raw_attn_flat[normal_mask].max().item()
            
            anomaly_attn_mean = raw_attn_flat[anomaly_mask].mean().item()
            anomaly_attn_std = raw_attn_flat[anomaly_mask].std().item()
            anomaly_attn_min = raw_attn_flat[anomaly_mask].min().item()
            anomaly_attn_max = raw_attn_flat[anomaly_mask].max().item()
            
            # Correlation with Mahalanobis
            # If assumption is correct: negative correlation (positive attention → low Mahalanobis)
            correlation = torch.corrcoef(torch.stack([raw_attn_flat, mahal_flat]))[0, 1].item()
            
            # Log results
            _debug_logger.info("=" * 80)
            _debug_logger.info("🔍 ATTENTION VERIFICATION RESULTS")
            _debug_logger.info("=" * 80)
            _debug_logger.info(f"Normal patches (N={normal_mask.sum().item()}):")
            _debug_logger.info(f"  Mean attention: {normal_attn_mean:.4f} (std={normal_attn_std:.4f})")
            _debug_logger.info(f"  Range: [{normal_attn_min:.4f}, {normal_attn_max:.4f}]")
            _debug_logger.info(f"Anomalous patches (N={anomaly_mask.sum().item()}):")
            _debug_logger.info(f"  Mean attention: {anomaly_attn_mean:.4f} (std={anomaly_attn_std:.4f})")
            _debug_logger.info(f"  Range: [{anomaly_attn_min:.4f}, {anomaly_attn_max:.4f}]")
            _debug_logger.info(f"Attention-Mahalanobis correlation: {correlation:.4f}")
            _debug_logger.info("")
            
            # Interpretation
            if normal_attn_mean > 0 and anomaly_attn_mean < 0:
                _debug_logger.info("✅ ASSUMPTION CONFIRMED: Positive attention = normal, Negative = anomalous")
                _debug_logger.info("   → Use: attention_anomaly = -raw_attention (flip sign)")
            elif normal_attn_mean < 0 and anomaly_attn_mean > 0:
                _debug_logger.info("⚠️  ASSUMPTION REVERSED: Positive attention = anomalous, Negative = normal")
                _debug_logger.info("   → Use: attention_anomaly = raw_attention (no flip)")
            else:
                _debug_logger.info("❓ ASSUMPTION UNCLEAR: Attention may not correlate with normality")
                _debug_logger.info("   → Consider using absolute value or reducing attention weight")
            
            if correlation < -0.3:
                _debug_logger.info(f"✅ Strong negative correlation ({correlation:.4f}) supports assumption")
            elif correlation > 0.3:
                _debug_logger.info(f"⚠️  Positive correlation ({correlation:.4f}) contradicts assumption")
            else:
                _debug_logger.info(f"❓ Weak correlation ({correlation:.4f}) - attention may be unreliable")
            
            _debug_logger.info("=" * 80)
            
            # Save statistics to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_file = os.path.join(verify_dir, f"attention_stats_{timestamp}.txt")
            with open(stats_file, "w") as f:
                f.write("ATTENTION VERIFICATION STATISTICS\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Normal patches (N={normal_mask.sum().item()}):\n")
                f.write(f"  Mean: {normal_attn_mean:.4f}\n")
                f.write(f"  Std: {normal_attn_std:.4f}\n")
                f.write(f"  Min: {normal_attn_min:.4f}\n")
                f.write(f"  Max: {normal_attn_max:.4f}\n\n")
                f.write(f"Anomalous patches (N={anomaly_mask.sum().item()}):\n")
                f.write(f"  Mean: {anomaly_attn_mean:.4f}\n")
                f.write(f"  Std: {anomaly_attn_std:.4f}\n")
                f.write(f"  Min: {anomaly_attn_min:.4f}\n")
                f.write(f"  Max: {anomaly_attn_max:.4f}\n\n")
                f.write(f"Attention-Mahalanobis correlation: {correlation:.4f}\n")
            
            # Visualize attention maps for first few images
            self._visualize_attention_maps(
                raw_attention_scores=raw_attention_scores,
                patch_labels=patch_labels,
                images=images,
                save_dir=verify_dir,
                timestamp=timestamp,
            )
    
    def _visualize_attention_maps(
        self,
        raw_attention_scores: torch.Tensor,
        patch_labels: torch.Tensor,
        images: torch.Tensor,
        save_dir: str,
        timestamp: str,
    ) -> None:
        """Visualize attention maps for normal and anomalous patches.
        
        Args:
            raw_attention_scores: (B, N) raw attention scores.
            patch_labels: (B, N) binary patch labels.
            images: (B, 3, H, W) original images.
            save_dir: Directory to save visualizations.
            timestamp: Timestamp string for file naming.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from utils.heatmap import patches_to_heatmap, overlay_heatmap
        
        # Visualize first 4 images
        num_viz = min(4, images.shape[0])
        
        fig, axes = plt.subplots(num_viz, 3, figsize=(15, 5 * num_viz))
        if num_viz == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_viz):
            # Original image
            img = images[i].permute(1, 2, 0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # Normalize to [0, 1]
            img = (img * 255).astype(np.uint8)
            
            # Raw attention heatmap
            raw_attn = raw_attention_scores[i].detach().cpu()  # (N,)
            attn_heatmap = patches_to_heatmap(raw_attn, normalize=True)
            attn_overlay = overlay_heatmap(img.copy(), attn_heatmap, alpha=0.5)
            
            # Patch labels heatmap
            labels = patch_labels[i].detach().cpu().float()  # (N,)
            label_heatmap = patches_to_heatmap(labels, normalize=False)
            label_overlay = overlay_heatmap(img.copy(), label_heatmap, alpha=0.5, colormap_name="RdYlGn")
            
            # Plot
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Image {i+1}")
            axes[i, 0].axis("off")
            
            axes[i, 1].imshow(attn_overlay)
            axes[i, 1].set_title(f"Raw Attention (mean={raw_attn.mean().item():.4f})")
            axes[i, 1].axis("off")
            
            axes[i, 2].imshow(label_overlay)
            axes[i, 2].set_title(f"Ground Truth (anomaly={labels.sum().item()}/{len(labels)} patches)")
            axes[i, 2].axis("off")
        
        plt.tight_layout()
        import os
        viz_file = os.path.join(save_dir, f"attention_visualization_{timestamp}.png")
        plt.savefig(viz_file, dpi=150, bbox_inches="tight")
        plt.close()
        
        _debug_logger.info(f"💾 Saved attention visualization to: {viz_file}")
    
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
        # Update learnable parameter (don't replace it)
        with torch.no_grad():
            self.score_gamma.data.fill_(score_gamma)
        
        self.freq_extractor = FourierPatchFeatureExtractor(
            num_freq_bands=freq_num_bands,
            use_phase=freq_use_phase,
            feature_dim=freq_feature_dim,
        )
        
        # Separate frequency density scorer (GMM or Mahalanobis, matching spatial setting)
        if self.use_gmm and self.gmm_scorer is not None:
            self.freq_gmm_scorer = GMMScorer(
                feature_dim=freq_feature_dim,
                n_components=self._gmm_freq_n,
                covariance_type=self._gmm_covariance_type,
                reg_covar=self._gmm_reg_covar,
                gamma=float(self.gmm_scorer.gamma),
                fit_max_samples=self._gmm_fit_max_samples,
                fit_subsample_seed=self._gmm_fit_subsample_seed,
            )
            self.freq_mahalanobis_scorer = None
        else:
            reg = self.mahalanobis_scorer.regularization if self.mahalanobis_scorer is not None else 1e-4
            gam = self.mahalanobis_scorer.gamma if self.mahalanobis_scorer is not None else 1.0
            self.freq_mahalanobis_scorer = MahalanobisScoring(
                feature_dim=freq_feature_dim,
                regularization=reg,
                gamma=gam,
            )
            self.freq_gmm_scorer = None
        
        # Frequency statistics tracker
        self.freq_normal_stats_tracker = NormalStatisticsTracker(
            feature_dim=freq_feature_dim,
        )
