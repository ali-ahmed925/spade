"""Image transforms for training and evaluation.

For anomaly detection, we use minimal augmentation to preserve the normal manifold.
Aggressive augmentation (flips, color jitter) can make normal samples look anomalous
or hide real defects, which hurts anomaly detection performance.
"""

from torchvision import transforms


def get_train_transforms(
    image_size: int = 224,
    use_flip: bool = False,
    use_color_jitter: bool = False,
) -> transforms.Compose:
    """Training-time transforms (minimal augmentation for anomaly detection).
    
    Args:
        image_size: Target image size (ViT requirement).
        use_flip: Whether to use random horizontal flip (default: False).
                  ⚠️ Can be problematic for oriented objects (screws, bottles).
        use_color_jitter: Whether to use color jitter (default: False).
                          ⚠️ Can hide real anomalies or make normal samples look anomalous.
    
    Returns:
        Compose transform pipeline.
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),  # Required: ViT needs fixed size
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],   # CLIP stats (required for BLIP-2)
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ]
    
    # Optional augmentations (insert before ToTensor)
    if use_flip:
        transform_list.insert(1, transforms.RandomHorizontalFlip(p=0.5))
    if use_color_jitter:
        transform_list.insert(1, transforms.ColorJitter(brightness=0.05, contrast=0.05))
    
    return transforms.Compose(transform_list)


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Evaluation-time transforms (deterministic)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])



