"""D5 — does adding spatial context to patch features raise defect contrast?

MECHANISM UNDER TEST
--------------------
D4 measured that transistor/cable defects sit only 1.5-3x above the normal
median patch score, while classes that work sit at 8-75x. The proposed
explanation: those defects are geometric rearrangements of locally-normal
material (a bent lead is still lead), and ViT-G patch embeddings are computed
per patch and encode local appearance, so such a defect barely moves a patch off
the normal manifold. If that is right, the missing information is RELATIONAL —
how a patch relates to its surroundings — and no scorer, aggregation or
resolution change can recover it.

THE TEST
--------
Augment each patch feature with context, refit the same global Gaussian, and
re-measure. Nothing is trained.

  raw          x                          the current representation (control)
  neigh        [x, mean(3x3 neighbours)]  local context appended
  global       [x, mean(all patches)]     image-level context appended
  residual     x - mean(3x3 neighbours)   how the patch DIFFERS from its
                                          surroundings; a bent lead differs from
                                          neighbouring leads even though it
                                          looks like lead
  neigh_resid  [x, x - mean(3x3)]         appearance and deviation together

READING THE RESULT
------------------
Defect elevation and WITHIN-image patch AUROC are the numbers that matter;
within-image cancels every whole-image confound. If a context variant lifts
transistor from ~1.5x / 0.68 toward the 8x / 0.93 band, relational information
is the missing ingredient and the Q-Former (the only component that aggregates
across patches) is the principled place to learn it. If nothing moves, the
mechanism is wrong.

All variants get identical treatment — same shrinkage, same regularization,
same gamma — so dimensionality is the only thing that differs, and that is
reported alongside so conditioning effects stay visible.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mvtec_dataset import MVTecDataset  # noqa: E402
from data.transforms import get_eval_transforms  # noqa: E402
from eval import load_config  # noqa: E402
from scripts.diagnose_failure import mask_to_patch_grid, split_indices  # noqa: E402
from models.builder import checkpoint_path  # noqa: E402
from scripts.fit_fine_statistics import build_model  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from utils.logging import get_logger  # noqa: E402

VARIANTS = ["raw", "neigh", "global", "residual", "neigh_resid"]


def neighbourhood_mean(x: torch.Tensor, grid: int) -> torch.Tensor:
    """Mean of the 3x3 neighbourhood of every patch. (B, N, D) -> (B, N, D)."""
    B, N, D = x.shape
    spatial = x.transpose(1, 2).reshape(B, D, grid, grid)
    pooled = F.avg_pool2d(spatial, kernel_size=3, stride=1, padding=1, count_include_pad=False)
    return pooled.reshape(B, D, N).transpose(1, 2)


def augment(x: torch.Tensor, variant: str, grid: int) -> torch.Tensor:
    """Build a context-augmented patch representation."""
    if variant == "raw":
        return x
    if variant == "global":
        g = x.mean(dim=1, keepdim=True).expand_as(x)
        return torch.cat([x, g], dim=-1)
    nb = neighbourhood_mean(x, grid)
    if variant == "neigh":
        return torch.cat([x, nb], dim=-1)
    if variant == "residual":
        return x - nb
    if variant == "neigh_resid":
        return torch.cat([x, x - nb], dim=-1)
    raise ValueError(f"unknown variant {variant!r}")


class GaussianFitter:
    """Single global Gaussian, fitted in one streaming pass (float64)."""

    def __init__(self, dim: int, device):
        self.dim = dim
        self.device = device
        self.sum_x = torch.zeros(dim, dtype=torch.float64, device=device)
        self.sum_xx = torch.zeros(dim, dim, dtype=torch.float64, device=device)
        self.n = 0

    @torch.no_grad()
    def update(self, feats: torch.Tensor) -> None:
        f = feats.reshape(-1, feats.shape[-1]).to(self.device, torch.float64)
        self.sum_x += f.sum(0)
        self.sum_xx += f.T @ f
        self.n += f.shape[0]

    @torch.no_grad()
    def finalize(self, regularization: float = 1e-4, shrinkage: float = 0.1):
        mu = self.sum_x / self.n
        sigma = (self.sum_xx - self.n * torch.outer(mu, mu)) / max(self.n - 1, 1)
        sigma = 0.5 * (sigma + sigma.T)
        eye = torch.eye(self.dim, dtype=sigma.dtype, device=sigma.device)
        if shrinkage > 0:
            sigma = (1 - shrinkage) * sigma + shrinkage * (torch.trace(sigma) / self.dim) * eye
        sigma_inv = torch.linalg.inv(sigma + regularization * eye)
        return mu.float(), sigma_inv.float(), self.n


@torch.no_grad()
def mahalanobis(feats: torch.Tensor, mu: torch.Tensor, sigma_inv: torch.Tensor) -> torch.Tensor:
    B, N, D = feats.shape
    flat = (feats - mu).reshape(B * N, D)
    return torch.clamp((flat @ sigma_inv * flat).sum(1), min=0).view(B, N)


@torch.no_grad()
def extract_train(model, paths, image_size, device, batch_size=4):
    """Yield (B, N, D) raw patch features for train/good images."""
    from PIL import Image as PILImage

    transform = get_eval_transforms(image_size)
    batch = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        batch.append(transform(PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
        if len(batch) == batch_size:
            yield model.vision_encoder(torch.stack(batch).to(device))[:, 1:, :].float()
            batch = []
    if batch:
        yield model.vision_encoder(torch.stack(batch).to(device))[:, 1:, :].float()


def evaluate_variant(scores_by_image, labels, defect_cov, defect_thresh=0.5) -> dict:
    """Elevation, within/across patch AUROC and image AUROC for one variant."""
    normal_patches, defect_patches, within = [], [], []
    image_scores = []

    for s, lab, cov in zip(scores_by_image, labels, defect_cov):
        k = min(3, s.size)
        image_scores.append(float(np.sort(s)[-k:].mean()))
        if lab == 0:
            normal_patches.append(s)
            continue
        d, c = s[cov >= defect_thresh], s[cov == 0.0]
        if d.size:
            defect_patches.append(d)
        if d.size and c.size:
            within.append(roc_auc_score(np.r_[np.zeros(len(c)), np.ones(len(d))], np.r_[c, d]))

    normal_all = np.concatenate(normal_patches)
    defect_all = np.concatenate(defect_patches)
    normal_median, defect_median = float(np.median(normal_all)), float(np.median(defect_all))

    return {
        "image_auroc": roc_auc_score(labels, image_scores),
        "across": roc_auc_score(
            np.r_[np.zeros(len(normal_all)), np.ones(len(defect_all))],
            np.r_[normal_all, defect_all]),
        "within": float(np.mean(within)) if within else float("nan"),
        "normal_median": normal_median,
        "defect_median": defect_median,
        "elevation_ratio": defect_median / normal_median if normal_median > 0 else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="D5: context-augmented features")
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["val", "heldout", "all"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--variants", type=str, default=",".join(VARIANTS))
    args = parser.parse_args()

    cfg = load_config()
    cfg["dataset"]["category"] = args.category
    logger = get_logger("D5")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or checkpoint_path(cfg, args.category)
    image_size, patch_size = cfg["vit"]["image_size"], cfg["vit"]["patch_size"]
    grid = image_size // patch_size

    train_paths = sorted(
        str(p) for p in (Path(cfg["dataset"]["root"]) / args.category / "train" / "good").glob("*.png")
    )
    probe = MVTecDataset(root=cfg["dataset"]["root"], category=args.category, split="test",
                         image_size=image_size, patch_size=patch_size, synthetic_method=None)
    dataset = MVTecDataset(root=cfg["dataset"]["root"], category=args.category, split="test",
                           image_size=image_size, patch_size=patch_size, synthetic_method=None,
                           subset_indices=split_indices(len(probe), args.split))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    logger.info(f"category={args.category} split={args.split} train={len(train_paths)} test={len(dataset)}")

    model = build_model(cfg, checkpoint, device)

    # Cache raw test features once; augmentation is cheap and variant-specific.
    logger.info("caching test features")
    test_feats, labels, defect_cov = [], [], []
    with torch.no_grad():
        for batch in loader:
            feats = model.vision_encoder(batch["image"].to(device))[:, 1:, :].float()
            test_feats.append(feats.cpu())
            for i in range(feats.shape[0]):
                labels.append(int(batch["label"][i]))
                m = batch["mask"][i]
                m = m.numpy() if isinstance(m, torch.Tensor) else m
                defect_cov.append(mask_to_patch_grid(m, grid))
    labels = np.array(labels)

    results = {}
    for variant in args.variants.split(","):
        logger.info(f"--- variant: {variant} ---")
        fitter = None
        for raw in extract_train(model, train_paths, image_size, device, args.batch_size):
            feats = augment(raw, variant, grid)
            if fitter is None:
                fitter = GaussianFitter(feats.shape[-1], device)
            fitter.update(feats)
        mu, sigma_inv, n_train = fitter.finalize(shrinkage=args.shrinkage)
        logger.info(f"    dim={mu.numel()}  train patches={n_train} "
                    f"({n_train / mu.numel():.1f}/dim)")

        scores_by_image = []
        with torch.no_grad():
            for chunk in test_feats:
                feats = augment(chunk.to(device), variant, grid)
                s = mahalanobis(feats, mu, sigma_inv).cpu().numpy()
                scores_by_image.extend(list(s))

        r = evaluate_variant(scores_by_image, labels, defect_cov)
        r["dim"] = int(mu.numel())
        r["samples_per_dim"] = n_train / mu.numel()
        results[variant] = r

    print(f"\n{'=' * 92}")
    print(f"D5  CONTEXT-AUGMENTED FEATURES  —  {args.category}  (split={args.split})")
    print(f"{'=' * 92}")
    print(f"{'variant':<14}{'dim':>7}{'smp/dim':>9}{'image AUROC':>13}"
          f"{'within':>9}{'across':>9}{'elevation':>11}")
    print("-" * 92)
    base = results.get("raw")
    for v, r in results.items():
        mark = ""
        if base and v != "raw":
            d_within = r["within"] - base["within"]
            mark = f"   within {d_within:+.4f}"
        print(f"{v:<14}{r['dim']:>7}{r['samples_per_dim']:>9.1f}{r['image_auroc']:>13.4f}"
              f"{r['within']:>9.4f}{r['across']:>9.4f}{r['elevation_ratio']:>10.2f}x{mark}")
    print("-" * 92)
    print("target band for a healthy class: elevation >= 8x, within >= 0.93")
    print(f"{'=' * 92}\n")


if __name__ == "__main__":
    main()
