"""D6 — anatomy of the failure: which patches, which defects, and why.

Answers three questions with distributions rather than aggregates:

  Q1  Is the failure normal patches scoring too HIGH, or defect patches scoring
      too LOW? Compared by percentile, not by mean, because the image score is a
      top-k statistic and therefore reads the upper tail of normals against the
      body of the defect distribution.

  Q2  Which defect subtypes fail, and do they share a property — small area,
      geometric/structural change, or something else?

  Q3  For each subtype, is the defect signal absent (not separable even from its
      own image's clean patches) or merely drowned (separable locally, but below
      the normal tail globally)? These need completely different fixes:
        absent  -> representation cannot encode the defect
        drowned -> representation is fine, calibration/aggregation loses it

Run on a healthy class alongside the weak ones; the contrast is the evidence.
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
from scripts.diagnose_failure import collect, split_indices  # noqa: E402
from models.builder import checkpoint_path  # noqa: E402
from scripts.fit_fine_statistics import build_model  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from utils.logging import get_logger  # noqa: E402


def anatomy(rows: list[dict], defect_thresh: float = 0.5) -> dict:
    normals = [r for r in rows if r["label"] == 0]
    anomalies = [r for r in rows if r["label"] == 1]

    normal_patches = np.concatenate([r["patch_scores"] for r in normals])
    defect_patches, clean_of_anom = [], []
    for r in anomalies:
        d = r["patch_scores"][r["patch_defect"] >= defect_thresh]
        c = r["patch_scores"][r["patch_defect"] == 0.0]
        if d.size:
            defect_patches.append(d)
        if d.size and c.size:
            clean_of_anom.append(c)
    defect_all = np.concatenate(defect_patches)

    n_pct = {p: float(np.percentile(normal_patches, p)) for p in (50, 90, 99, 99.9)}
    n_pct["max"] = float(normal_patches.max())
    d_pct = {p: float(np.percentile(defect_all, p)) for p in (10, 25, 50, 75, 90)}

    # How much of the defect distribution is buried under the normal tail?
    drowned = float((defect_all < n_pct[99]).mean())
    # How much of the normal distribution reaches into defect territory?
    normals_in_defect_range = float((normal_patches > d_pct[50]).mean())

    # per-image top-3 (what the image score actually reads)
    normal_top3 = np.array([np.sort(r["patch_scores"])[-3:].mean() for r in normals])
    anom_top3 = np.array([np.sort(r["patch_scores"])[-3:].mean() for r in anomalies])

    per_type = {}
    for t in sorted({r["defect_type"] for r in anomalies}):
        sub = [r for r in anomalies if r["defect_type"] == t]
        d_sub, w_sub = [], []
        for r in sub:
            d = r["patch_scores"][r["patch_defect"] >= defect_thresh]
            c = r["patch_scores"][r["patch_defect"] == 0.0]
            if d.size:
                d_sub.append(d)
            if d.size and c.size:
                w_sub.append(roc_auc_score(
                    np.r_[np.zeros(len(c)), np.ones(len(d))], np.r_[c, d]))
        d_cat = np.concatenate(d_sub) if d_sub else np.array([0.0])
        top3 = np.array([np.sort(r["patch_scores"])[-3:].mean() for r in sub])
        per_type[t] = {
            "n": len(sub),
            "area": float(np.median([r["defect_area"] for r in sub])) * 100,
            "image_auroc": roc_auc_score(
                np.r_[np.zeros(len(normal_top3)), np.ones(len(top3))],
                np.r_[normal_top3, top3]),
            "within": float(np.mean(w_sub)) if w_sub else float("nan"),
            "defect_median": float(np.median(d_cat)),
            "drowned": float((d_cat < n_pct[99]).mean()),
        }

    return {
        "n_pct": n_pct, "d_pct": d_pct,
        "drowned": drowned,
        "normals_in_defect_range": normals_in_defect_range,
        "normal_top3": normal_top3, "anom_top3": anom_top3,
        "per_type": per_type,
        "n_normal_patches": len(normal_patches),
        "n_defect_patches": len(defect_all),
    }


def report(category: str, a: dict) -> None:
    n, d = a["n_pct"], a["d_pct"]
    print(f"\n{'=' * 88}")
    print(f"D6  FAILURE ANATOMY  —  {category}")
    print(f"{'=' * 88}")

    print("Q1  where do the two distributions sit?")
    print(f"    normal patches   p50 {n[50]:>9.0f}   p90 {n[90]:>9.0f}   "
          f"p99 {n[99]:>9.0f}   p99.9 {n[99.9]:>9.0f}   max {n['max']:>9.0f}")
    print(f"    defect patches   p10 {d[10]:>9.0f}   p25 {d[25]:>9.0f}   "
          f"p50 {d[50]:>9.0f}   p75 {d[75]:>9.0f}   p90 {d[90]:>9.0f}")
    print()
    print(f"    defect p50 / normal p50   {d[50] / n[50]:>6.2f}x   (elevation over the body)")
    print(f"    defect p50 / normal p99   {d[50] / n[99]:>6.2f}x   "
          f"(elevation over the TAIL — this is what top-k reads)")
    print(f"    defect patches below normal p99      {a['drowned'] * 100:>5.1f}%  <- drowned")
    print(f"    normal patches above defect p50      {a['normals_in_defect_range'] * 100:>5.1f}%")
    print()
    nt, at = a["normal_top3"], a["anom_top3"]
    print(f"    per-image top-3   normals  median {np.median(nt):>9.0f}  max {nt.max():>9.0f}")
    print(f"                      anomalies median {np.median(at):>9.0f}  min {at.min():>9.0f}")
    print(f"    anomalies scoring below the worst normal: "
          f"{int((at < nt.max()).sum())}/{len(at)}")

    print(f"\nQ2/Q3  per defect subtype")
    print(f"{'defect type':<24}{'n':>3}{'area%':>8}{'img AUROC':>11}{'within':>9}"
          f"{'defect p50':>12}{'drowned':>9}")
    print("-" * 88)
    for t, v in sorted(a["per_type"].items(), key=lambda kv: kv[1]["image_auroc"]):
        print(f"{t:<24}{v['n']:>3}{v['area']:>8.2f}{v['image_auroc']:>11.4f}"
              f"{v['within']:>9.4f}{v['defect_median']:>12.0f}{v['drowned'] * 100:>8.1f}%")
    print("-" * 88)
    print("within LOW  -> signal ABSENT   (representation cannot encode it)")
    print("within HIGH + drowned HIGH -> signal DROWNED (present locally, lost globally)")
    print(f"{'=' * 88}\n")


def main() -> None:
    p = argparse.ArgumentParser(description="D6: failure anatomy")
    p.add_argument("--category", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--split", type=str, default="val", choices=["val", "heldout", "all"])
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()

    cfg = load_config()
    cfg["dataset"]["category"] = args.category
    logger = get_logger("D6")
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
    logger.info(f"category={args.category} split={args.split} images={len(dataset)}")

    model = build_model(cfg, checkpoint, device)
    report(args.category, anatomy(collect(model, loader, device, grid)))


if __name__ == "__main__":
    main()
