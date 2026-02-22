"""Utility for extracting image patches from tensors."""

import torch
import numpy as np


def extract_image_patches_from_tensor(
    images: torch.Tensor,
    patch_size: int = 14,
) -> np.ndarray:
    """
    Extract 14×14 patches from CLIP-normalized images.
    
    Args:
        images: (B, 3, 224, 224) normalized tensor
        patch_size: patch size (14 for ViT-B/14)
    
    Returns:
        patches: (B*N, 14, 14, 3) uint8 numpy array [0, 255]
    """
    B, C, H, W = images.shape
    device = images.device
    
    # Denormalize CLIP normalization
    # BLIP-2 uses ImageNet normalization (same as CLIP)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    
    images_denorm = images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)
    
    # Scale to [0, 255]
    images_denorm = (images_denorm * 255).to(torch.uint8)
    
    # Convert to numpy (B, H, W, C)
    images_np = images_denorm.cpu().numpy().transpose(0, 2, 3, 1)
    
    # Extract patches
    grid_size = H // patch_size  # 16
    all_patches = []
    
    for img in images_np:
        patches = []
        for i in range(grid_size):
            for j in range(grid_size):
                y = i * patch_size
                x = j * patch_size
                patch = img[y:y+patch_size, x:x+patch_size, :]
                patches.append(patch)
        all_patches.extend(patches)
    
    return np.array(all_patches, dtype=np.uint8)  # (B*256, 14, 14, 3)

