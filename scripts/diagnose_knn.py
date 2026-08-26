"""D2 — how much of the gap is the single-Gaussian assumption?

The measured failure on hard classes is false positives on NORMAL images
(transistor good/037 outscoring eight real defects). One Gaussian, fitted over
all patches, assumes the normal distribution is unimodal. An object is not:
board, lead and plastic body are genuinely different modes, and a single mean
sits in empty space between them, so legitimately-normal patches land far from
it.

This measures the size of that effect WITHOUT changing the architecture: score
each test patch by its distance to the nearest normal patch in a memory bank
(the PatchCore mechanism) instead of by Mahalanobis distance to one mean, and
compare.

    kNN ~= Mahalanobis  -> unimodality is not the problem; look elsewhere
    kNN >> Mahalanobis  -> the normal model is the binding constraint, and the
                           fix that keeps Mahalanobis is a MIXTURE (per-mode
                           means and covariances), not a different scorer

This is a diagnostic, not a proposed replacement.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mvtec_dataset import MVTecDataset  # noqa: E402
from data.transforms import get_eval_transforms  # noqa: E402
from eval import load_config  # noqa: E402
from scripts.fit_fine_statistics import build_model  # noqa: E402
from utils.heatmap import patches_to_heatmap  # noqa: E402
from utils.logging import get_logger  # noqa: E402
from utils.metrics import compute_image_auroc, compute_pixel_auroc, compute_pro  # noqa: E402


def heldout_indices(n: int, seed: int = 42) -> list[int]:
    return np.random.RandomState(seed).permutation(n).tolist()[n // 2 :]


@torch.no_grad()
def build_memory_bank(
    model, image_paths, image_size, device, logger, max_patches=25000, seed=42, batch_size=4
) -> torch.Tensor:
    """Collect normal patch features from train/good, randomly subsampled."""
    from PIL import Image as PILImage

    transform = get_eval_transforms(image_size)
    rng = np.random.default_rng(seed)
    chunks, batch = [], []
    per_image_cap = max(1, max_patches // max(len(image_paths), 1))

    def flush(bt):
        if not bt:
            return
        images = torch.stack(bt).to(device)
        embeds = model.vision_encoder(images)[:, 1:, :].float()
        flat = embeds.reshape(-1, embeds.shape[-1])
        take = min(per_image_cap * len(bt), flat.shape[0])
        idx = torch.from_numpy(rng.choice(flat.shape[0], size=take, replace=False)).to(device)
        chunks.append(flat[idx].cpu())

    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        batch.append(transform(PILImage.fromarray(img)))
        if len(batch) == batch_size:
            flush(batch)
            batch = []
    flush(batch)

    bank = torch.cat(chunks)
    logger.info(f"memory bank: {bank.shape[0]} patches x {bank.shape[1]} dims")
    return bank


@torch.no_grad()
def knn_scores(patch_embeds: torch.Tensor, bank: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Mean squared distance to the k nearest normal patches. (B, N) out."""
    B, N, D = patch_embeds.shape
    flat = patch_embeds.reshape(B * N, D)
    # ||x - b||^2 = ||x||^2 - 2 x.b + ||b||^2
    x2 = (flat * flat).sum(1, keepdim=True)
    b2 = (bank * bank).sum(1).unsqueeze(0)
    out = torch.empty(B * N, device=flat.device)
    step = 4096
    for i in range(0, flat.shape[0], step):
        chunk = flat[i : i + step]
        d = x2[i : i + step] - 2.0 * (chunk @ bank.T) + b2
        d = torch.clamp(d, min=0.0)
        out[i : i + step] = d.topk(k, dim=1, largest=False).values.mean(dim=1)
    return out.view(B, N)


@torch.no_grad()
def evaluate(model, loader, device, bank, k, image_size, patch_size, smooth_sigma):
    """Score the held-out set with both the model's own scorer and kNN."""
    res = {"mahalanobis": {"img": [], "maps": []}, "knn": {"img": [], "maps": []}}
    labels, masks = [], []

    for batch in loader:
        images = batch["image"].to(device)
        out = model(images)
        embeds = model.vision_encoder(images)[:, 1:, :].float()
        knn = knn_scores(embeds, bank, k=k)

        for i in range(images.shape[0]):
            labels.append(int(batch["label"][i]))
            m = batch["mask"][i]
            masks.append(m.numpy() if isinstance(m, torch.Tensor) else m)
            for name, tensor in (("mahalanobis", out["patch_scores"]), ("knn", knn)):
                res[name]["img"].append(float(model.get_image_score(tensor[i : i + 1]).cpu()))
                res[name]["maps"].append(
                    patches_to_heatmap(
                        tensor[i].detach().cpu(), image_size=image_size,
                        patch_size=patch_size, normalize=False, smooth_sigma=smooth_sigma,
                    )
                )

    labels_arr, masks_arr = np.array(labels), np.stack(masks)
    summary = {}
    for name, slot in res.items():
        maps = np.stack(slot["maps"])
        entry = {"image_auroc": compute_image_auroc(labels_arr, np.array(slot["img"]))}
        if masks_arr.max() > 0:
            entry["pixel_auroc"] = compute_pixel_auroc(masks_arr, maps)
            try:
                entry["pro"] = compute_pro(masks_arr, maps)
            except Exception:
                entry["pro"] = float("nan")
        summary[name] = entry
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="D2: kNN vs single-Gaussian normal model")
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--max-patches", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--smooth-sigma", type=float, default=4.0)
    args = parser.parse_args()

    cfg = load_config()
    cfg["dataset"]["category"] = args.category
    logger = get_logger("D2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or f"checkpoints/{args.category}/spade_best.pt"
    image_size, patch_size = cfg["vit"]["image_size"], cfg["vit"]["patch_size"]

    train_dir = Path(cfg["dataset"]["root"]) / args.category / "train" / "good"
    train_paths = sorted(str(p) for p in train_dir.glob("*.png"))

    probe = MVTecDataset(root=cfg["dataset"]["root"], category=args.category, split="test",
                         image_size=image_size, patch_size=patch_size, synthetic_method=None)
    dataset = MVTecDataset(root=cfg["dataset"]["root"], category=args.category, split="test",
                           image_size=image_size, patch_size=patch_size, synthetic_method=None,
                           subset_indices=heldout_indices(len(probe)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    logger.info(f"category={args.category}  train={len(train_paths)}  held-out={len(dataset)}  k={args.k}")

    model = build_model(cfg, checkpoint, device)
    bank = build_memory_bank(
        model, train_paths, image_size, device, logger,
        max_patches=args.max_patches, batch_size=args.batch_size,
    ).to(device)

    summary = evaluate(model, loader, device, bank, args.k, image_size, patch_size, args.smooth_sigma)

    print(f"\n{'=' * 78}")
    print(f"D2  NORMAL MODEL: single Gaussian vs kNN memory bank  —  {args.category}")
    print(f"{'=' * 78}")
    print(f"{'normal model':<28}{'image AUROC':>14}{'pixel AUROC':>14}{'PRO':>10}")
    print("-" * 78)
    for name, label in (("mahalanobis", "Mahalanobis (current)"), ("knn", f"kNN memory bank (k={args.k})")):
        r = summary[name]
        print(f"{label:<28}{r['image_auroc']:>14.4f}"
              f"{r.get('pixel_auroc', float('nan')):>14.4f}{r.get('pro', float('nan')):>10.4f}")
    print("-" * 78)
    d_img = summary["knn"]["image_auroc"] - summary["mahalanobis"]["image_auroc"]
    d_pix = summary["knn"].get("pixel_auroc", 0) - summary["mahalanobis"].get("pixel_auroc", 0)
    print(f"delta (kNN - Mahalanobis):  image {d_img:+.4f}   pixel {d_pix:+.4f}")
    verdict = ("unimodality IS the binding constraint -> multi-modal Mahalanobis (GMM)"
               if d_img > 0.02 else
               "unimodality is NOT the main problem -> look elsewhere")
    print(f"verdict: {verdict}")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
