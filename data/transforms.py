"""Image transforms for training and evaluation."""

from torchvision import transforms


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Training-time transforms (with augmentation)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],   # CLIP stats
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])


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



