"""D7 — which defect subclasses actually cost us the AUROC?

Per-subclass AUROC says how well a subclass is detected. It does NOT say how
much that subclass *cost* the overall number, because a subclass with 2 images
and a subclass with 20 damage the aggregate very differently.

This attributes the loss exactly. Image AUROC is the fraction of
(normal, anomaly) pairs ranked correctly:

    AUROC = 1 - (misordered pairs / total pairs)

Every misordered pair belongs to exactly one anomalous image, so the deficit
decomposes with no remainder:

    1 - AUROC  =  sum over subclasses of  (misordered pairs of that subclass)
                                          ----------------------------------
                                            n_normal * n_anomalous

The "AUROC cost" column is therefore how many points of image AUROC would be
recovered if that subclass were ranked perfectly, holding everything else fixed.
The column sums to the total deficit.

Also reports, per anomalous image, its rank among all test images and how many
normal images outscore it — so individual failures can be inspected directly.
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
from scripts.fit_fine_statistics import build_model  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from utils.logging import get_logger  # noqa: E402


def attribute(rows: list[dict]) -> dict:
    normals = [r for r in rows if r["label"] == 0]
    anomalies = [r for r in rows if r["label"] == 1]
    normal_scores = np.array([r["image_score"] for r in normals])
    n_norm, n_anom = len(normals), len(anomalies)
    total_pairs = n_norm * n_anom

    overall = roc_auc_score(
        np.array([r["label"] for r in rows]),
        np.array([r["image_score"] for r in rows]),
    )

    # rank of every image (1 = highest score)
    order = sorted(rows, key=lambda r: -r["image_score"])
    rank = {id(r): i + 1 for i, r in enumerate(order)}

    per_image = []
    for r in anomalies:
        above = int((normal_scores > r["image_score"]).sum())
        ties = int((normal_scores == r["image_score"]).sum())
        per_image.append({
            "defect_type": r["defect_type"],
            "name": r["name"],
            "score": r["image_score"],
            "rank": rank[id(r)],
            "normals_above": above + 0.5 * ties,
            "defect_area": r["defect_area"] * 100,
        })

    per_type = {}
    for t in sorted({r["defect_type"] for r in anomalies}):
        sub = [p for p in per_image if p["defect_type"] == t]
        sub_scores = np.array([p["score"] for p in sub])
        misordered = float(sum(p["normals_above"] for p in sub))
        per_type[t] = {
            "n": len(sub),
            "auroc": roc_auc_score(
                np.r_[np.zeros(n_norm), np.ones(len(sub))],
                np.r_[normal_scores, sub_scores]),
            "median_rank": float(np.median([p["rank"] for p in sub])),
            "n_below_worst_normal": int(sum(1 for p in sub if p["score"] < normal_scores.max())),
            "misordered": misordered,
            "auroc_cost": misordered / total_pairs,
            "median_area": float(np.median([p["defect_area"] for p in sub])),
        }

    return {
        "overall": overall, "n_norm": n_norm, "n_anom": n_anom,
        "total_pairs": total_pairs, "per_type": per_type, "per_image": per_image,
        "normal_max": float(normal_scores.max()),
        "normal_median": float(np.median(normal_scores)),
    }


def report(category: str, split: str, a: dict, show_images: int = 12) -> None:
    print(f"\n{'=' * 94}")
    print(f"D7  SUBCLASS ATTRIBUTION  —  {category}  (split={split})")
    print(f"{'=' * 94}")
    print(f"image AUROC {a['overall']:.4f}   "
          f"{a['n_norm']} normal / {a['n_anom']} anomalous   "
          f"deficit {1 - a['overall']:.4f}")
    print(f"worst normal score {a['normal_max']:.0f}   median normal {a['normal_median']:.0f}\n")

    print(f"{'defect type':<24}{'n':>4}{'AUROC':>9}{'med rank':>10}"
          f"{'<worst nrm':>12}{'AUROC cost':>12}{'share':>9}{'area%':>8}")
    print("-" * 94)
    deficit = max(1 - a["overall"], 1e-12)
    for t, v in sorted(a["per_type"].items(), key=lambda kv: -kv[1]["auroc_cost"]):
        print(f"{t:<24}{v['n']:>4}{v['auroc']:>9.4f}{v['median_rank']:>10.0f}"
              f"{v['n_below_worst_normal']:>7}/{v['n']:<4}{v['auroc_cost']:>12.4f}"
              f"{v['auroc_cost'] / deficit * 100:>8.1f}%{v['median_area']:>8.2f}")
    print("-" * 94)
    total_cost = sum(v["auroc_cost"] for v in a["per_type"].values())
    print(f"{'TOTAL':<24}{a['n_anom']:>4}{'':>9}{'':>10}{'':>12}{total_cost:>12.4f}"
          f"{'100.0%':>9}")
    print(f"(check: 1 - AUROC = {1 - a['overall']:.4f})")

    worst = sorted(a["per_image"], key=lambda p: -p["normals_above"])[:show_images]
    if worst and worst[0]["normals_above"] > 0:
        print(f"\nworst individual failures ({show_images} shown):")
        print(f"{'defect type':<24}{'image':<8}{'score':>12}{'rank':>7}"
              f"{'normals above':>15}{'area%':>8}")
        print("-" * 94)
        for p in worst:
            if p["normals_above"] == 0:
                break
            print(f"{p['defect_type']:<24}{p['name']:<8}{p['score']:>12.0f}"
                  f"{p['rank']:>7}{p['normals_above']:>15.1f}{p['defect_area']:>8.2f}")
    print(f"{'=' * 94}\n")


def main() -> None:
    p = argparse.ArgumentParser(description="D7: subclass attribution of the AUROC deficit")
    p.add_argument("--category", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--split", type=str, default="heldout", choices=["val", "heldout", "all"])
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()

    cfg = load_config()
    cfg["dataset"]["category"] = args.category
    logger = get_logger("D7")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or f"checkpoints/{args.category}/spade_best.pt"
    image_size, patch_size = cfg["vit"]["image_size"], cfg["vit"]["patch_size"]

    probe = MVTecDataset(root=cfg["dataset"]["root"], category=args.category, split="test",
                         image_size=image_size, patch_size=patch_size, synthetic_method=None)
    dataset = MVTecDataset(root=cfg["dataset"]["root"], category=args.category, split="test",
                           image_size=image_size, patch_size=patch_size, synthetic_method=None,
                           subset_indices=split_indices(len(probe), args.split))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    logger.info(f"category={args.category} split={args.split} images={len(dataset)}")

    model = build_model(cfg, checkpoint, device)
    report(args.category, args.split, attribute(collect(model, loader, device, image_size // patch_size)))


if __name__ == "__main__":
    main()
