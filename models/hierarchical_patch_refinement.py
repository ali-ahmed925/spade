"""Hierarchical Patch Refinement with Hybrid Patch Annealing (HPA).

Properly re-computes query attention at each refinement step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform hierarchical patch refinement with attention re-computation.
        
        Args:
            patch_embeds: (B, N, D) patch embeddings (without CLS).
            qformer: Q-Former module for re-computing queries.
            query_tokens: (B, Q, D_q) learnable query tokens.
            query_patch_attn: Query-patch attention module.
            mahalanobis_scorer: Mahalanobis scoring module.
            cls_token: (B, 1, D) CLS token.
            alpha: Weight for attention importance.
            beta: Weight for deviation.
            
        Returns:
            (refined_patches, selected_indices)
            - refined_patches: (B, N_min, D) final refined patches.
            - selected_indices: (B, N_min) indices of selected patches.
        """
        B, N, D = patch_embeds.shape
        current_patches = patch_embeds.clone()
        current_indices = torch.arange(N, device=patch_embeds.device).unsqueeze(0).expand(B, -1)
        
        # Hierarchical refinement steps
        for t in range(self.t_steps):
            n_t = self.compute_n_t(t)
            
            if current_patches.shape[1] <= n_t:
                break
            
            # ── CRITICAL: Re-compute attention on current patch subset ──
            # 1. Re-compute Q-Former queries on current patches
            current_image_embeds = torch.cat([cls_token, current_patches], dim=1)  # (B, N_t+1, D)
            image_atts = torch.ones(
                current_image_embeds.size()[:-1],
                dtype=torch.long,
                device=current_image_embeds.device,
            )
            
            # Re-run Q-Former on refined patch set
            qformer_outputs = qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=current_image_embeds,
                encoder_attention_mask=image_atts,
            )
            refined_query_embeds = qformer_outputs.last_hidden_state  # (B, Q, D_q)
            
            # 2. Re-compute attention importance on current patches
            _, attention_importance = query_patch_attn(
                refined_query_embeds, current_patches
            )  # (B, N_t)
            
            # 3. Re-compute Mahalanobis deviation on current patches
            deviation_scores = mahalanobis_scorer(current_patches)  # (B, N_t)
            
            # 4. Compute patch scores for selection
            patch_scores = alpha * attention_importance + beta * deviation_scores  # (B, N_t)
            
            # 5. Select top-N_t patches
            _, top_indices_local = torch.topk(patch_scores, n_t, dim=1)  # (B, n_t)
            
            # Map local indices back to global indices
            batch_indices = torch.arange(B, device=patch_embeds.device).unsqueeze(1)
            current_patches = current_patches[batch_indices, top_indices_local]
            current_indices = current_indices[batch_indices, top_indices_local]
            
            # Update CLS token (optional, could keep original)
            # For now, we'll recompute it from current patches
            cls_token = current_patches.mean(dim=1, keepdim=True)
            
            # Clear intermediate tensors to save memory
            del qformer_outputs, refined_query_embeds, attention_importance, deviation_scores, patch_scores
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return current_patches, current_indices


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

