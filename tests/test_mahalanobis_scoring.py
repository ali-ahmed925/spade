"""Tests for Mahalanobis scoring."""

import torch
import pytest

from models.mahalanobis_scoring import MahalanobisScoring


def test_mahalanobis_update_statistics():
    """Test statistics update."""
    D = 1408
    scorer = MahalanobisScoring(feature_dim=D)
    
    # Generate normal patches
    normal_patches = torch.randn(1000, D)
    
    scorer.update_statistics(normal_patches)
    
    assert scorer.is_initialized
    assert scorer.mu.shape == (D,)
    assert scorer.sigma_inv.shape == (D, D)


def test_mahalanobis_forward():
    """Test Mahalanobis score computation."""
    D = 1408
    scorer = MahalanobisScoring(feature_dim=D, gamma=1.3)
    
    # Initialize with normal patches
    normal_patches = torch.randn(1000, D)
    scorer.update_statistics(normal_patches)
    
    # Test on new patches
    B, N = 2, 256
    test_patches = torch.randn(B, N, D)
    
    scores = scorer(test_patches)
    
    assert scores.shape == (B, N)
    assert torch.all(scores >= 0)  # Mahalanobis distance is non-negative


def test_mahalanobis_amplification():
    """Test non-linear amplification."""
    D = 1408
    scorer_no_amp = MahalanobisScoring(feature_dim=D, gamma=1.0)
    scorer_amp = MahalanobisScoring(feature_dim=D, gamma=1.5)
    
    normal_patches = torch.randn(1000, D)
    scorer_no_amp.update_statistics(normal_patches)
    scorer_amp.update_statistics(normal_patches)
    
    test_patches = torch.randn(1, 10, D)
    scores_no_amp = scorer_no_amp(test_patches, apply_amplification=False)
    scores_amp = scorer_amp(test_patches, apply_amplification=True)
    
    # Amplified scores should be different (and typically higher for large values)
    assert not torch.allclose(scores_no_amp, scores_amp)



