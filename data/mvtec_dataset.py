"""MVTec Anomaly Detection dataset loader.

Supports per-category loading with train (normal-only) and test splits.
Integrates synthetic anomaly generation for training.
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.synthetic import (
    cutpaste,
    crack_anomaly,
    mask_to_patch_labels,
    get_methods_for_category,
)
from data.transforms import get_train_transforms, get_eval_transforms


class MVTecDataset(Dataset):
    """MVTec AD dataset.

    Directory layout expected::

        root/
          category/
            train/
              good/  *.png
            test/
              good/  *.png
              defect_type/  *.png
            ground_truth/
              defect_type/  *_mask.png
    """

    def __init__(
        self,
        root: str,
        category: str,
        split: str = "train",
        image_size: int = 224,
        patch_size: int = 14,
        synthetic_method: str | None = None,  # None = auto-select based on category
        synthetic_prob: float = 0.2,
        subset_indices: list[int] | None = None,
        deterministic: bool = False,
        base_seed: int = 42,
        use_flip: bool = False,
        use_color_jitter: bool = False,
        synthetic_cfg: dict | None = None,  # Config dict for synthetic parameters
    ) -> None:
        super().__init__()
        assert split in ("train", "test"), f"Unknown split: {split}"

        self.root = Path(root) / category
        self.category = category
        self.split = split
        self.image_size = image_size
        self.patch_size = patch_size
        self.synthetic_prob = float(synthetic_prob)
        self.subset_indices = subset_indices
        self.deterministic = bool(deterministic)
        self.base_seed = int(base_seed)
        self._use_flip = bool(use_flip)
        self._use_color_jitter = bool(use_color_jitter)
        self.synthetic_cfg = synthetic_cfg or {}
        
        # Determine synthetic method(s) for this category
        if split == "train":
            if not (0.0 <= self.synthetic_prob <= 1.0):
                raise ValueError(f"synthetic_prob must be in [0, 1], got {self.synthetic_prob}")
            
            if synthetic_method is None:
                # Auto-select methods based on category
                available_methods = get_methods_for_category(category)
                self.available_methods = available_methods
                self.synthetic_method = None  # Will randomly select from available_methods
            else:
                # Use specified method (must be in available methods for category)
                available_methods = get_methods_for_category(category)
                if synthetic_method not in available_methods:
                    raise ValueError(
                        f"Method '{synthetic_method}' not available for category '{category}'. "
                        f"Available: {available_methods}"
                    )
                self.available_methods = [synthetic_method]
                self.synthetic_method = synthetic_method
        else:
            self.synthetic_method = None
            self.available_methods = []

        self.transform = (
            get_train_transforms(image_size, use_flip=self._use_flip, use_color_jitter=self._use_color_jitter)
            if split == "train"
            else get_eval_transforms(image_size)
        )

        self.image_paths: list[str] = []
        self.mask_paths: list[str | None] = []
        self.labels: list[int] = []  # 0 = normal, 1 = anomaly

        self._load_paths()
        self._indices = self.subset_indices if self.subset_indices is not None else list(range(len(self.image_paths)))

    # ── path loading ──────────────────────────
    def _load_paths(self) -> None:
        split_dir = self.root / self.split
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            is_normal = class_dir.name == "good"
            for img_path in sorted(class_dir.glob("*.png")):
                self.image_paths.append(str(img_path))
                self.labels.append(0 if is_normal else 1)
                if is_normal or self.split == "train":
                    self.mask_paths.append(None)
                else:
                    # Locate ground-truth mask
                    mask_name = img_path.stem + "_mask.png"
                    mask_path = self.root / "ground_truth" / class_dir.name / mask_name
                    self.mask_paths.append(str(mask_path) if mask_path.exists() else None)

    # ── __len__ / __getitem__ ─────────────────
    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        """Return a sample dict.

        Keys:
            image:        (3, H, W) normalised tensor.
            label:        int, 0 or 1.
            patch_labels: (N_patches,) float tensor (train only, else zeros).
            mask:         (H, W) numpy uint8 ground-truth mask (test only, else zeros).
            path:         str, image file path.
        """
        real_idx = self._indices[idx]

        # Load image as RGB numpy array
        image_np = cv2.imread(self.image_paths[real_idx])
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_np = cv2.resize(image_np, (self.image_size, self.image_size))

        n_patches = (self.image_size // self.patch_size) ** 2

        # ── Train: apply synthetic anomaly ──
        if self.split == "train" and self.synthetic_method is not None:
            # Mix clean and synthetic-anomaly samples from the normal training set.
            # - clean sample: label=0, patch_labels=zeros
            # - synthetic:     label=1, patch_labels from synthetic mask
            if self.deterministic:
                import random as _random
                local_rng = _random.Random(self.base_seed + real_idx)
                local_np_rng = np.random.default_rng(self.base_seed + real_idx)
                apply_synth = float(local_np_rng.random()) < self.synthetic_prob
            else:
                local_rng = None
                local_np_rng = None
                apply_synth = np.random.rand() < self.synthetic_prob
            if apply_synth:
                # Randomly select from available methods for this category
                if self.synthetic_method is None:
                    # Random selection from available methods
                    if self.deterministic:
                        method_idx = local_rng.randint(0, len(self.available_methods) - 1)
                    else:
                        method_idx = random.randint(0, len(self.available_methods) - 1)
                    selected_method = self.available_methods[method_idx]
                else:
                    selected_method = self.synthetic_method
                
                # Get method-specific config
                method_cfg = self.synthetic_cfg.get(selected_method, {})
                
                # Apply selected synthetic method
                if selected_method == "cutpaste":
                    aug_np, mask_np = cutpaste(
                        image_np,
                        area_ratio=tuple(method_cfg.get("area_ratio", [0.02, 0.15])),
                        blend_alpha=method_cfg.get("blend_alpha", 0.7),
                        color_jitter=method_cfg.get("color_jitter", 0.1),
                        rng=local_rng,
                        np_rng=local_np_rng,
                    )
                elif selected_method == "crack":
                    aug_np, mask_np = crack_anomaly(
                        image_np,
                        num_cracks=method_cfg.get("num_cracks", 3),
                        min_length=method_cfg.get("min_length", 0.1),
                        max_length=method_cfg.get("max_length", 0.4),
                        min_width=method_cfg.get("min_width", 1),
                        max_width=method_cfg.get("max_width", 3),
                        darkness=method_cfg.get("darkness", 0.3),
                        branching_prob=method_cfg.get("branching_prob", 0.3),
                        rng=local_rng,
                        np_rng=local_np_rng,
                    )
                else:
                    raise ValueError(f"Unknown synthetic method: {selected_method}")

                patch_labels = mask_to_patch_labels(mask_np, self.patch_size)  # (N_patches,)
                image_pil = Image.fromarray(aug_np)
                image_tensor = self.transform(image_pil)  # (3, H, W)
                return {
                    "image": image_tensor,
                    "label": 1,  # synthetic anomaly
                    "patch_labels": patch_labels,  # (N_patches,)
                    "mask": mask_np,  # (H, W)
                    "path": self.image_paths[real_idx],
                }

            # Clean sample (no anomaly)
            clean_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            patch_labels = torch.zeros((n_patches,), dtype=torch.float32)
            image_pil = Image.fromarray(image_np)
            image_tensor = self.transform(image_pil)
            return {
                "image": image_tensor,
                "label": 0,
                "patch_labels": patch_labels,
                "mask": clean_mask,
                "path": self.image_paths[real_idx],
            }

        # ── Test / normal train (no synth) ──
        image_pil = Image.fromarray(image_np)
        image_tensor = self.transform(image_pil)

        # Ground truth mask for test set
        if self.mask_paths[real_idx] is not None:
            gt_mask = cv2.imread(self.mask_paths[real_idx], cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.resize(gt_mask, (self.image_size, self.image_size))
            gt_mask = (gt_mask > 127).astype(np.uint8)
        else:
            gt_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        patch_labels = mask_to_patch_labels(gt_mask, self.patch_size)

        return {
            "image": image_tensor,
            "label": self.labels[real_idx],
            "patch_labels": patch_labels,
            "mask": gt_mask,
            "path": self.image_paths[real_idx],
        }

