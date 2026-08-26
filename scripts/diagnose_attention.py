"""D1 — do the Q-Former queries encode WHERE the defect is?

Goal 3 (visual tokens that can drive language generation) rests on an untested
premise: that the learned queries carry spatial defect information. This measures
it directly, by scoring each component of the anomaly score ON ITS OWN against
the ground-truth masks.

    attention-only pixel AUROC ~= 0.5  -> queries know nothing about location
    attention-only pixel AUROC >> 0.5  -> queries already carry usable signal

The same decomposition also shows how much each stream contributes to detection,
which tells us where the headroom is before we spend a training run on it.

No training, no gradients. Uses SPADE's own `score_components` output so the
numbers are exactly the terms the model sums.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mvtec_dataset import MVTecDataset  # noqa: E402
from eval import load_config  # noqa: E402
from scripts.fit_fine_statistics import build_model  # noqa: E402
from utils.heatmap import patches_to_heatmap  # noqa: E402
from utils.logging import get_logger  # noqa: E402
from utils.metrics import compute_image_auroc, compute_pixel_auroc, compute_pro  # noqa: E402


def heldout_indices(n: int, seed: int = 42) -> list[int]:
    perm = np.random.RandomState(seed).permutation(n).tolist()
    return perm[n // 2 :]


@torch.no_grad()
def diagnose(model, loader, device, image_size, patch_size, smooth_sigma, logger) -> dict:
    """Score each component separately and report its standalone performance."""
    per_component: dict[str, dict[str, list]] = {}
    labels: list[int] = []
    masks: list[np.ndarray] = []

    for batch in loader:
        images = batch["image"].to(device)
        out = model(images)
        components = dict(out["score_components"])
        components["TOTAL"] = out["patch_scores"]

        for i in range(images.shape[0]):
            labels.append(int(batch["label"][i]))
            m = batch["mask"][i]
            masks.append(m.numpy() if isinstance(m, torch.Tensor) else m)

            for name, tensor in components.items():
                slot = per_component.setdefault(name, {"image": [], "maps": []})
                patch_scores = tensor[i].detach().cpu()
                slot["image"].append(float(model.get_image_score(tensor[i : i + 1]).cpu()))
                slot["maps"].append(
                    patches_to_heatmap(
                        patch_scores, image_size=image_size, patch_size=patch_size,
                        normalize=False, smooth_sigma=smooth_sigma,
                    )
                )

    labels_arr = np.array(labels)
    masks_arr = np.stack(masks)
    results = {}
    for name, slot in per_component.items():
        scores = np.array(slot["image"])
        maps = np.stack(slot["maps"])
        entry = {"image_auroc": compute_image_auroc(labels_arr, scores)}
        if masks_arr.max() > 0:
            entry["pixel_auroc"] = compute_pixel_auroc(masks_arr, maps)
            try:
                entry["pro"] = compute_pro(masks_arr, maps)
            except Exception:
                entry["pro"] = float("nan")
        results[name] = entry
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="D1: what does each score stream know?")
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--smooth-sigma", type=float, default=4.0)
    args = parser.parse_args()

    cfg = load_config()
    cfg["dataset"]["category"] = args.category
    logger = get_logger("D1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or f"checkpoints/{args.category}/spade_best.pt"

    probe = MVTecDataset(
        root=cfg["dataset"]["root"], category=args.category, split="test",
        image_size=cfg["vit"]["image_size"], patch_size=cfg["vit"]["patch_size"],
        synthetic_method=None,
    )
    dataset = MVTecDataset(
        root=cfg["dataset"]["root"], category=args.category, split="test",
        image_size=cfg["vit"]["image_size"], patch_size=cfg["vit"]["patch_size"],
        synthetic_method=None, subset_indices=heldout_indices(len(probe)),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    logger.info(f"category={args.category}  held-out images={len(dataset)}")

    model = build_model(cfg, checkpoint, device)
    results = diagnose(
        model, loader, device,
        image_size=cfg["vit"]["image_size"], patch_size=cfg["vit"]["patch_size"],
        smooth_sigma=args.smooth_sigma, logger=logger,
    )

    print(f"\n{'=' * 78}")
    print(f"D1  STANDALONE POWER OF EACH SCORE STREAM  —  {args.category}")
    print(f"{'=' * 78}")
    print(f"{'stream':<24}{'image AUROC':>14}{'pixel AUROC':>14}{'PRO':>10}")
    print("-" * 78)
    order = ["attention", "spatial_mahalanobis", "frequency", "cross", "TOTAL"]
    for name in order:
        if name not in results:
            continue
        r = results[name]
        print(f"{name:<24}{r['image_auroc']:>14.4f}"
              f"{r.get('pixel_auroc', float('nan')):>14.4f}{r.get('pro', float('nan')):>10.4f}")
    print("-" * 78)
    attn = results.get("attention", {}).get("pixel_auroc", float("nan"))
    print(f"attention-only pixel AUROC = {attn:.4f}  "
          f"({'NO spatial defect signal in the queries' if abs(attn - 0.5) < 0.05 else 'queries carry spatial signal'})")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
