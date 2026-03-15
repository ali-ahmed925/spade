"""Track and update normal patch statistics during training."""

import torch
import torch.nn as nn


class NormalStatisticsTracker(nn.Module):
    """Tracks normal patch embeddings for statistical modeling.
    
    Collects ALL normal patches during training (no buffer limit).
    Statistics are computed once from all collected patches.
    """
    
    def __init__(
        self,
        feature_dim: int,
        buffer_size: int = None,  # Deprecated, kept for backward compatibility
        update_frequency: int = None,  # Deprecated, kept for backward compatibility
    ):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Buffer for normal patches - use list to store ALL patches (no limit)
        self.normal_patch_buffer = []
        self.statistics_computed = False
        
    def add_normal_patches(
        self,
        patch_embeds: torch.Tensor,
        patch_labels: torch.Tensor,
    ):
        """Add normal patches to buffer.
        
        Args:
            patch_embeds: (B, N, D) patch embeddings.
            patch_labels: (B, N) binary patch labels (0=normal, 1=anomaly).
        """
        if self.statistics_computed:
            # Statistics already computed, don't add more patches
            return
        
        # Extract normal patches (label == 0)
        normal_mask = (patch_labels == 0)  # (B, N)
        normal_patches = patch_embeds[normal_mask]  # (N_normal, D)
        
        if normal_patches.numel() > 0:
            # Detach and move to CPU to save memory
            normal_patches_cpu = normal_patches.detach().cpu()
            # Extend buffer with individual patches
            # normal_patches_cpu is (N_normal, D), iterate over first dimension
            for i in range(normal_patches_cpu.shape[0]):
                self.normal_patch_buffer.append(normal_patches_cpu[i])
    
    def get_statistics(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Get current μ and Σ from buffer.
        
        Returns:
            (mu, sigma) tensors of shape (D,) and (D, D), or (None, None) if buffer empty.
        """
        if len(self.normal_patch_buffer) == 0:
            return None, None
        
        # Convert buffer to tensor
        normal_patches = torch.stack(list(self.normal_patch_buffer))  # (N, D)
        
        # Compute statistics
        mu = normal_patches.mean(dim=0)  # (D,)
        centered = normal_patches - mu.unsqueeze(0)
        sigma = (centered.T @ centered) / max(normal_patches.shape[0] - 1, 1)  # (D, D)
        
        return mu, sigma
    
    def compute_statistics_once(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Compute statistics once from all collected patches and mark as computed.
        
        Returns:
            (mu, sigma) tensors of shape (D,) and (D, D), or (None, None) if buffer empty.
        """
        mu, sigma = self.get_statistics()
        self.statistics_computed = True
        return mu, sigma

