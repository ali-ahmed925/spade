"""D4 — mechanism-level failure diagnosis for weak classes.

Aggregate AUROC says a class is weak. It does not say WHY. This separates the
candidate causes so an intervention can target the actual one.

The central decomposition is three patch-level separabilities, computed from the
same per-patch scores:

  1. ACROSS-IMAGE patch AUROC
       defect patches  vs  all patches of normal images.
       This is what the image-level score effectively sees.

  2. WITHIN-IMAGE patch AUROC
       defect patches  vs  non-defect patches OF THE SAME ANOMALOUS IMAGE.
       Any whole-image score offset cancels, because both groups share it.

  3. per-image OFFSET spread
       std of per-image median patch score across normal images, expressed in
       units of the defect elevation.

Reading them together identifies the mechanism:

  within HIGH, across LOW, offset LARGE
      -> the representation separates defects fine; whole-image score offsets
         (pose, lighting, exposure) move normal images into the anomalous range.
         The failure is a normalization/conditioning problem, not a detection one.

  within LOW
      -> defect patches are not separable from their own neighbours. The
         representation or the normal model is inadequate. No aggregation or
         normalization can fix this.

  within HIGH, across HIGH, image AUROC LOW
      -> patch scoring is fine and the failure is in how patches become an
         image score (aggregation, area effects, small-defect dilution).

It also reports, per class: which defect types are missed, the actual false
positives and false negatives, defect area, and where high scores land on normal
images (on the object vs the frame).

Diagnostics run on the VALIDATION half by default; the held-out half stays
untouched for final numbers.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mvtec_dataset import MVTecDataset  # noqa: E402
from eval import load_config  # noqa: E402
from models.builder import checkpoint_path  # noqa: E402
from scripts.fit_fine_statistics import build_model  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from utils.logging import get_logger  # noqa: E402


def split_indices(n: int, which: str, seed: int = 42) -> list[int] | None:
    if which == "all":
        return None
    perm = np.random.RandomState(seed).permutation(n).tolist()
    return perm[: n // 2] if which == "val" else perm[n // 2 :]


def mask_to_patch_grid(mask: np.ndarray, grid: int) -> np.ndarray:
    """Fraction of each patch cell covered by the ground-truth mask."""
    h, w = mask.shape
    cell_h, cell_w = h // grid, w // grid
    m = (mask > 0).astype(np.float32)
    return m[: cell_h * grid, : cell_w * grid].reshape(
        grid, cell_h, grid, cell_w
    ).mean(axis=(1, 3)).reshape(-1)


@torch.no_grad()
def collect(model, loader, device, grid: int) -> dict:
    """Per-image patch scores, patch defect coverage and metadata."""
    rows = []
    for batch in loader:
        images = batch["image"].to(device)
        out = model(images)
        patch_scores = out["patch_scores"].detach().cpu().numpy()
        image_scores = model.get_image_score(out["patch_scores"]).detach().cpu().numpy()

        for i in range(images.shape[0]):
            mask = batch["mask"][i]
            mask = mask.numpy() if isinstance(mask, torch.Tensor) else mask
            path = batch["path"][i] if isinstance(batch["path"], (list, tuple)) else batch["path"]
            rows.append({
                "path": str(path),
                "defect_type": Path(str(path)).parent.name,
                "name": Path(str(path)).stem,
                "label": int(batch["label"][i]),
                "patch_scores": patch_scores[i],
                "patch_defect": mask_to_patch_grid(mask, grid),
                "defect_area": float((mask > 0).mean()),
                "image_score": float(image_scores[i]),
            })
    return rows


def analyse(rows: list[dict], grid: int, logger, defect_thresh: float = 0.5) -> dict:
    normals = [r for r in rows if r["label"] == 0]
    anomalies = [r for r in rows if r["label"] == 1]
    if not normals or not anomalies:
        raise RuntimeError("need both normal and anomalous images")

    # ── 1. image level ───────────────────────────────────────────────────────
    y = np.array([r["label"] for r in rows])
    s = np.array([r["image_score"] for r in rows])
    image_auroc = roc_auc_score(y, s)

    # ── 2. patch level, ACROSS images ────────────────────────────────────────
    defect_patches, normal_image_patches, clean_patches_of_anomalous = [], [], []
    for r in normals:
        normal_image_patches.append(r["patch_scores"])
    for r in anomalies:
        is_defect = r["patch_defect"] >= defect_thresh
        is_clean = r["patch_defect"] == 0.0
        defect_patches.append(r["patch_scores"][is_defect])
        clean_patches_of_anomalous.append(r["patch_scores"][is_clean])

    defect_all = np.concatenate([d for d in defect_patches if d.size])
    normal_all = np.concatenate(normal_image_patches)
    across = roc_auc_score(
        np.r_[np.zeros(len(normal_all)), np.ones(len(defect_all))],
        np.r_[normal_all, defect_all],
    )

    # ── 3. patch level, WITHIN each anomalous image ──────────────────────────
    within_scores = []
    for d, c in zip(defect_patches, clean_patches_of_anomalous):
        if d.size == 0 or c.size == 0:
            continue
        within_scores.append(
            roc_auc_score(np.r_[np.zeros(len(c)), np.ones(len(d))], np.r_[c, d])
        )
    within = float(np.mean(within_scores)) if within_scores else float("nan")

    # ── 4. whole-image offsets ───────────────────────────────────────────────
    normal_medians = np.array([np.median(r["patch_scores"]) for r in normals])
    defect_median = float(np.median(defect_all))
    normal_median = float(np.median(normal_all))
    elevation = defect_median - normal_median
    offset_std = float(normal_medians.std())
    offset_ratio = offset_std / elevation if elevation > 0 else float("inf")

    # ── 5. who is misranked ──────────────────────────────────────────────────
    ranked = sorted(rows, key=lambda r: -r["image_score"])
    worst_normal = max(normals, key=lambda r: r["image_score"])
    n_normals_above = sum(
        1 for a in anomalies if a["image_score"] < worst_normal["image_score"]
    )
    false_positives = [
        r for r in ranked if r["label"] == 0
        and r["image_score"] > np.median([a["image_score"] for a in anomalies])
    ]
    false_negatives = [
        r for r in anomalies
        if r["image_score"] < np.median([n["image_score"] for n in normals])
    ]

    # ── 6. per defect type ───────────────────────────────────────────────────
    per_type = {}
    normal_img_scores = np.array([r["image_score"] for r in normals])
    for t in sorted({r["defect_type"] for r in anomalies}):
        sub = [r for r in anomalies if r["defect_type"] == t]
        sc = np.array([r["image_score"] for r in sub])
        per_type[t] = {
            "n": len(sub),
            "auroc": roc_auc_score(
                np.r_[np.zeros(len(normal_img_scores)), np.ones(len(sc))],
                np.r_[normal_img_scores, sc],
            ),
            "median_defect_area": float(np.median([r["defect_area"] for r in sub])),
            "within_auroc": float(np.mean([
                w for w, r in zip(within_scores, [a for a in anomalies
                                                  if a["patch_defect"].max() >= defect_thresh])
                if r["defect_type"] == t
            ])) if within_scores else float("nan"),
        }

    # ── 7. where do high scores land on NORMAL images ────────────────────────
    hot = np.zeros(grid * grid)
    for r in normals:
        top = np.argsort(r["patch_scores"])[-3:]
        hot[top] += 1
    hot_concentration = float(hot.max() / max(hot.sum(), 1))

    return {
        "image_auroc": image_auroc,
        "patch_across": across,
        "patch_within": within,
        "normal_median": normal_median,
        "defect_median": defect_median,
        "elevation": elevation,
        "offset_std": offset_std,
        "offset_ratio": offset_ratio,
        "worst_normal": worst_normal,
        "n_anomalies_below_worst_normal": n_normals_above,
        "n_anomalies": len(anomalies),
        "false_positives": false_positives[:5],
        "false_negatives": sorted(false_negatives, key=lambda r: r["image_score"])[:5],
        "per_type": per_type,
        "hot_concentration": hot_concentration,
    }


def report(category: str, res: dict) -> None:
    print(f"\n{'=' * 82}")
    print(f"D4  FAILURE DIAGNOSIS  —  {category}")
    print(f"{'=' * 82}")

    print(f"image AUROC                         {res['image_auroc']:.4f}")
    print(f"patch AUROC, ACROSS images          {res['patch_across']:.4f}   "
          f"(what the image score sees)")
    print(f"patch AUROC, WITHIN anomalous image {res['patch_within']:.4f}   "
          f"(offset-free separability)")
    print()
    print(f"median patch score, normal images   {res['normal_median']:.1f}")
    print(f"median patch score, defect patches  {res['defect_median']:.1f}")
    print(f"defect elevation                    {res['elevation']:.1f}")
    print(f"per-image offset spread (std)       {res['offset_std']:.1f}")
    print(f"offset / elevation                  {res['offset_ratio']:.2f}   "
          f"({'OFFSETS DOMINATE' if res['offset_ratio'] > 0.5 else 'offsets minor'})")
    print()

    wn = res["worst_normal"]
    print(f"worst normal image: {wn['name']}  score {wn['image_score']:.1f}")
    print(f"  {res['n_anomalies_below_worst_normal']} of {res['n_anomalies']} "
          f"anomalies score BELOW it")
    print(f"top-3 hotspot concentration on normals {res['hot_concentration']:.3f}  "
          f"({'same locations every image' if res['hot_concentration'] > 0.3 else 'scattered'})")

    print(f"\n{'defect type':<24}{'n':>4}{'image AUROC':>13}{'within AUROC':>14}{'median area':>13}")
    print("-" * 82)
    for t, d in sorted(res["per_type"].items(), key=lambda kv: kv[1]["auroc"]):
        print(f"{t:<24}{d['n']:>4}{d['auroc']:>13.4f}{d['within_auroc']:>14.4f}"
              f"{d['median_defect_area'] * 100:>12.2f}%")

    if res["false_negatives"]:
        print("\nworst false negatives (anomalies scoring below the normal median):")
        for r in res["false_negatives"]:
            print(f"   {r['defect_type']:<22} {r['name']:<6} score={r['image_score']:>10.1f}"
                  f"  defect area={r['defect_area'] * 100:.2f}%")
    print(f"{'=' * 82}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="D4: mechanism-level failure diagnosis")
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["val", "heldout", "all"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config()
    cfg["dataset"]["category"] = args.category
    logger = get_logger("D4")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or checkpoint_path(cfg, args.category)
    image_size, patch_size = cfg["vit"]["image_size"], cfg["vit"]["patch_size"]
    grid = image_size // patch_size

    probe = MVTecDataset(root=cfg["dataset"]["root"], category=args.category, split="test",
                         image_size=image_size, patch_size=patch_size, synthetic_method=None)
    dataset = MVTecDataset(root=cfg["dataset"]["root"], category=args.category, split="test",
                           image_size=image_size, patch_size=patch_size, synthetic_method=None,
                           subset_indices=split_indices(len(probe), args.split))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    logger.info(f"category={args.category}  split={args.split}  images={len(dataset)}")

    model = build_model(cfg, checkpoint, device)
    rows = collect(model, loader, device, grid)
    report(args.category, analyse(rows, grid, logger))


if __name__ == "__main__":
    main()
