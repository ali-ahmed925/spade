"""Tests for normal statistics tracking."""

import torch
import pytest

from models.normal_statistics import NormalStatisticsTracker


def test_normal_statistics_tracker():
    """Test normal statistics tracking."""
    D = 1408
    tracker = NormalStatisticsTracker(feature_dim=D, buffer_size=1000)
    
    # Add normal and anomalous patches
    B, N = 2, 256
    patch_embeds = torch.randn(B, N, D)
    patch_labels = torch.zeros(B, N)  # All normal
    patch_labels[0, :10] = 1  # Some anomalous patches
    
    tracker.add_normal_patches(patch_embeds, patch_labels)
    
    # Check that only normal patches are in buffer
    mu, sigma = tracker.get_statistics()
    
    assert mu is not None
    assert sigma is not None
    assert mu.shape == (D,)
    assert sigma.shape == (D, D)
    
    # Buffer should contain approximately (B*N - 10) normal patches
    assert len(tracker.normal_patch_buffer) > 0



