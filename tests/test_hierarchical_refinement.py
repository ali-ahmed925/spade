"""Tests for hierarchical patch refinement."""

import torch
import pytest

from models.hierarchical_patch_refinement import HybridPatchAnnealing, QueryPatchAttention


def test_hpa_compute_n_t():
    """Test HPA patch count computation."""
    hpa = HybridPatchAnnealing(n_max=256, n_min=32, t_steps=15)
    
    # At t=0, should be close to n_max
    n_0 = hpa.compute_n_t(0)
    assert n_0 == 256
    
    # At t=T, should be n_min
    n_T = hpa.compute_n_t(15)
    assert n_T == 32
    
    # Should be monotonic decreasing
    n_prev = 256
    for t in range(1, 16):
        n_t = hpa.compute_n_t(t)
        assert n_t <= n_prev
        n_prev = n_t


def test_query_patch_attention():
    """Test query-patch attention computation."""
    B, Q, N, D = 2, 32, 256, 1408
    query_dim = 768
    
    attn = QueryPatchAttention(query_dim=query_dim, patch_dim=D)
    
    query_embeds = torch.randn(B, Q, query_dim)
    patch_embeds = torch.randn(B, N, D)
    
    attn_weights, attn_importance = attn(query_embeds, patch_embeds)
    
    assert attn_weights.shape == (B, Q, N)
    assert attn_importance.shape == (B, N)
    # Check attention weights sum to 1 over patches
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(B, Q), atol=1e-5)
    assert torch.all(attn_importance >= 0)



