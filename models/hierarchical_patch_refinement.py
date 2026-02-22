"""Hierarchical Patch Refinement with Hybrid Patch Annealing (HPA).

Properly re-computes query attention at each refinement step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from utils.debug_logger import get_debug_logger

# Initialize debug logger (shared with SPADE logger)
_debug_logger = get_debug_logger("spade_debug")


def _qformer_checkpoint_wrapper(qformer, query_embeds, encoder_hidden_states, encoder_attention_mask):
    """Wrapper for Q-Former forward pass to use with gradient checkpointing."""
    return qformer(
        query_embeds=query_embeds,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
    )


class HybridPatchAnnealing(nn.Module):
    """Hybrid Patch Annealing for gradual patch pruning.
    
    At each step, re-computes query attention on the current patch subset.
    """
    
    def __init__(
        self,
        n_max: int = 256,
        n_min: int = 32,
        t_steps: int = 15,
        w: float = 0.4,
        p1: float = 0.5,
        p2: float = 2.0,
    ):
        super().__init__()
        self.n_max = n_max
        self.n_min = n_min
        self.t_steps = t_steps
        self.w = w
        self.p1 = p1
        self.p2 = p2
        
    def compute_n_t(self, t: int) -> int:
        """Compute number of patches to keep at step t.
        
        Args:
            t: Current step (0 to T).
            
        Returns:
            Number of patches to keep.
        """
        if t >= self.t_steps:
            return self.n_min
        
        ratio = t / self.t_steps
        term1 = self.w * ((1 - ratio) ** self.p1)
        term2 = (1 - self.w) * ((1 - ratio) ** self.p2)
        n_t = self.n_min + (self.n_max - self.n_min) * (term1 + term2)
        return int(n_t)
    
    def forward(
        self,
        patch_embeds: torch.Tensor,
        qformer: nn.Module,
        query_tokens: torch.Tensor,
        query_patch_attn: nn.Module,
        mahalanobis_scorer: nn.Module,
        cls_token: torch.Tensor,
        alpha: float = 0.5,
        beta: float = 0.5,
        # NEW: Frequency scoring
        freq_scores: torch.Tensor | None = None,  # (B, N) - precomputed for ALL patches
        gamma: float = 0.25,  # Weight for frequency component
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform hierarchical patch refinement with attention re-computation.
        
        Queries are accumulated across refinement steps: each iteration uses the
        refined queries from the previous step, allowing hierarchical refinement.
        
        Now supports dual-stream scoring: spatial + frequency.
        
        Args:
            patch_embeds: (B, N, D) patch embeddings (without CLS).
            qformer: Q-Former module for re-computing queries.
            query_tokens: (B, Q, D_q) learnable query tokens.
            query_patch_attn: Query-patch attention module.
            mahalanobis_scorer: Mahalanobis scoring module.
            cls_token: (B, 1, D) CLS token.
            alpha: Weight for attention importance.
            beta: Weight for spatial deviation.
            freq_scores: (B, N) precomputed frequency Mahalanobis scores for ALL patches.
            gamma: Weight for frequency deviation.
            
        Returns:
            (refined_patches, selected_indices, final_refined_queries)
            - refined_patches: (B, N_min, D) final refined patches.
            - selected_indices: (B, N_min) indices of selected patches.
            - final_refined_queries: (B, Q, D_q) final refined queries after all steps.
        """
        B, N, D = patch_embeds.shape
        current_patches = patch_embeds.clone()
        current_indices = torch.arange(N, device=patch_embeds.device).unsqueeze(0).expand(B, -1)
        
        # ── CRITICAL: Initialize query tokens and accumulate refinement ──
        current_query_tokens = query_tokens.clone()  # Start with original learnable tokens
        
        # DEBUG: Track query statistics at each step
        initial_mean = current_query_tokens.mean().item()
        initial_std = current_query_tokens.std().item()
        initial_norm = current_query_tokens.norm(dim=-1).mean().item()
        _debug_logger.debug(f"[HPA DEBUG] Step 0 (initial): N={N}, query_mean={initial_mean:.4f}, query_std={initial_std:.4f}, query_norm={initial_norm:.4f}")
        
        # Hierarchical refinement steps
        for t in range(self.t_steps):
            n_t = self.compute_n_t(t)
            
            # Only break if we've already reached n_min (can't prune further)
            # Don't break if n_t >= current_patches (we still want to refine queries)
            if current_patches.shape[1] <= self.n_min:
                break
            
            # ── Re-compute attention on current patch subset ──
            # 1. Re-compute Q-Former queries on current patches
            current_image_embeds = torch.cat([cls_token, current_patches], dim=1)  # (B, N_t+1, D)
            image_atts = torch.ones(
                current_image_embeds.size()[:-1],
                dtype=torch.long,
                device=current_image_embeds.device,
            )
            
            # Re-run Q-Former with accumulated refined queries
            # Use gradient checkpointing to save memory during training
            if self.training and t > 0:  # Skip checkpointing on first step (less memory pressure)
                # Gradient checkpointing: trade compute for memory
                # Only checkpoint intermediate steps, not the first one
                qformer_outputs = checkpoint(
                    _qformer_checkpoint_wrapper,
                    qformer,
                    current_query_tokens,
                    current_image_embeds,
                    image_atts,
                    use_reentrant=False,
                )
            else:
                # No checkpointing during eval or first step (faster)
                qformer_outputs = qformer(
                    query_embeds=current_query_tokens,  # ✅ Use accumulated refined queries
                    encoder_hidden_states=current_image_embeds,
                    encoder_attention_mask=image_atts,
                )
            refined_query_embeds = qformer_outputs.last_hidden_state  # (B, Q, D_q)
            
            # DEBUG: Track query change
            query_change = (refined_query_embeds - current_query_tokens).abs().mean().item()
            refined_mean = refined_query_embeds.mean().item()
            refined_std = refined_query_embeds.std().item()
            refined_norm = refined_query_embeds.norm(dim=-1).mean().item()
            
            _debug_logger.debug(f"[HPA DEBUG] Step {t+1}: N={current_patches.shape[1]}→{n_t}, "
                  f"query_change={query_change:.6f}, "
                  f"mean={refined_mean:.4f}, std={refined_std:.4f}, norm={refined_norm:.4f}")
            
            # ── CRITICAL: Update query tokens for next iteration ──
            current_query_tokens = refined_query_embeds  # ✅ Accumulate refinement
            
            # 2. Re-compute attention importance on current patches
            _, attention_importance = query_patch_attn(
                refined_query_embeds, current_patches
            )  # (B, N_t)
            
            # 3. Re-compute Mahalanobis deviation on current patches (RAW - no normalization)
            spatial_mahal = mahalanobis_scorer(current_patches)  # (B, N_t)
            
            # 4. Get frequency scores for current patches (RAW - no normalization)
            if freq_scores is not None:
                # Gather frequency scores using current_indices
                batch_indices = torch.arange(B, device=patch_embeds.device).unsqueeze(1)
                current_freq_mahal = freq_scores[batch_indices, current_indices]  # (B, N_t)
            else:
                current_freq_mahal = torch.zeros_like(spatial_mahal)
            
            # 5. ⭐ COMBINED SCORING (spatial + frequency) - Using RAW Mahalanobis scores
            patch_scores = (
                alpha * attention_importance +
                beta * spatial_mahal +
                gamma * current_freq_mahal
            )  # (B, N_t)
            
            # 6. Select top-N_t patches
            _, top_indices_local = torch.topk(patch_scores, n_t, dim=1)  # (B, n_t)
            
            # Map local indices back to global indices
            batch_indices = torch.arange(B, device=patch_embeds.device).unsqueeze(1)
            current_patches = current_patches[batch_indices, top_indices_local]
            current_indices = current_indices[batch_indices, top_indices_local]
            
            # Update CLS token
            cls_token = current_patches.mean(dim=1, keepdim=True)
            
            # Clear intermediate tensors (but keep current_query_tokens!)
            del qformer_outputs, attention_importance, spatial_mahal, patch_scores
            if freq_scores is not None:
                del current_freq_mahal
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # ── FINAL STEP: One more attention pass on n_min patches with accumulated queries ──
        # This is the final refinement before using queries for scoring all 256 patches
        final_image_embeds = torch.cat([cls_token, current_patches], dim=1)  # (B, n_min+1, D)
        final_image_atts = torch.ones(
            final_image_embeds.size()[:-1],
            dtype=torch.long,
            device=final_image_embeds.device,
        )
        
        # Final step: Use gradient checkpointing during training
        if self.training:
            final_qformer_outputs = checkpoint(
                _qformer_checkpoint_wrapper,
                qformer,
                current_query_tokens,
                final_image_embeds,
                final_image_atts,
                use_reentrant=False,
            )
        else:
            final_qformer_outputs = qformer(
                query_embeds=current_query_tokens,  # Use accumulated refined queries
                encoder_hidden_states=final_image_embeds,
                encoder_attention_mask=final_image_atts,
            )
        final_refined_queries = final_qformer_outputs.last_hidden_state  # (B, Q, D_q)
        
        # DEBUG: Final step stats
        final_mean = final_refined_queries.mean().item()
        final_std = final_refined_queries.std().item()
        final_norm = final_refined_queries.norm(dim=-1).mean().item()
        total_change = (final_refined_queries - query_tokens).abs().mean().item()
        
        _debug_logger.debug(f"[HPA DEBUG] Final step: query_mean={final_mean:.4f}, query_std={final_std:.4f}, "
              f"query_norm={final_norm:.4f}, total_change_from_initial={total_change:.6f}")
        
        return current_patches, current_indices, final_refined_queries


class QueryPatchAttention(nn.Module):
    """Query-to-patch attention for computing attention importance."""
    
    def __init__(self, query_dim: int, patch_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, patch_dim)
        self.scale = patch_dim ** -0.5
        
    def forward(
        self,
        query_embeds: torch.Tensor,
        patch_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute query-patch attention.
        
        Args:
            query_embeds: (B, Q, D_q) query embeddings.
            patch_embeds: (B, N, D_p) patch embeddings.
            
        Returns:
            (attention_weights, attention_importance)
            - attention_weights: (B, Q, N) attention weights.
            - attention_importance: (B, N) aggregated importance per patch.
        """
        # Project queries to patch dimension
        queries = self.query_proj(query_embeds)  # (B, Q, D_p)
        
        # Compute attention
        attn_scores = torch.bmm(queries, patch_embeds.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, Q, N)
        
        # Aggregate importance: sum over queries
        attention_importance = attn_weights.sum(dim=1)  # (B, N)
        
        return attn_weights, attention_importance

