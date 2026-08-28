"""D1 — do the Q-Former queries know WHERE the defect is?

Goal 3 (visual tokens driving language generation) rests on a premise that must
be measured rather than assumed: that the 32 query tokens carry spatial defect
information. If they do not, any sentence the LLM produces about *where* the
defect is is unfounded, however fluent it sounds.

After the feature-space redesign the queries reach the score through the
contextualiser — every patch attends to all 32 queries — so the thing to measure
is that attention map, (B, N_patches, 32), scored against the ground-truth masks.

Three readings, each answering a different question:

  1. SCORE STREAMS, standalone
        what each additive term of the score achieves on its own.

  2. QUERY SALIENCY (max over queries) as an anomaly map
        does patch-to-query attention localise defects at all?
        ~0.5 -> the queries carry no spatial signal.

  3. PER-QUERY localisation
        does any INDIVIDUAL query specialise on defects? A query with high
        pixel AUROC is a defect detector, and its token is the one worth handing
        to the language model. A flat profile across all 32 means the queries
        share the work and no single token is groundable.

No training, no gradients.
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
from models.builder import checkpoint_path  # noqa: E402
from scripts.diagnose_failure import split_indices  # noqa: E402
from scripts.fit_fine_statistics import build_model  # noqa: E402
from utils.heatmap import patches_to_heatmap  # noqa: E402
from utils.logging import get_logger  # noqa: E402
from utils.metrics import compute_image_auroc, compute_pixel_auroc, compute_pro  # noqa: E402


def _score_maps(patch_scores, image_size, patch_size, smooth_sigma):
    return patches_to_heatmap(
        patch_scores, image_size=image_size, patch_size=patch_size,
        normalize=False, smooth_sigma=smooth_sigma,
    )


@torch.no_grad()
def diagnose(model, loader, device, image_size, patch_size, smooth_sigma, logger) -> dict:
    streams: dict[str, dict[str, list]] = {}
    query_maps: list[np.ndarray] = []      # per image: (n_queries, H, W)
    saliency_maps: list[np.ndarray] = []
    labels: list[int] = []
    masks: list[np.ndarray] = []
    n_queries = None

    for batch in loader:
        images = batch["image"].to(device)
        out = model(images, return_attention=True)

        # Component names are read from the model, never hardcoded — the old
        # version of this script assumed "attention"/"spatial_mahalanobis" and
        # silently reported nothing once the redesign renamed them.
        components = dict(out["score_components"])
        components["TOTAL"] = out["patch_scores"]
        attention = out.get("patch_query_attention")   # (B, N, Q)

        for i in range(images.shape[0]):
            labels.append(int(batch["label"][i]))
            m = batch["mask"][i]
            masks.append(m.numpy() if isinstance(m, torch.Tensor) else m)

            for name, tensor in components.items():
                slot = streams.setdefault(name, {"image": [], "maps": []})
                slot["image"].append(float(model.get_image_score(tensor[i : i + 1]).cpu()))
                slot["maps"].append(
                    _score_maps(tensor[i].detach().cpu(), image_size, patch_size, smooth_sigma)
                )

            if attention is not None:
                attn = attention[i].detach().cpu()            # (N, Q)
                n_queries = attn.shape[1]
                saliency_maps.append(
                    _score_maps(attn.max(dim=1).values, image_size, patch_size, smooth_sigma)
                )
                query_maps.append(
                    np.stack([
                        _score_maps(attn[:, q], image_size, patch_size, smooth_sigma)
                        for q in range(n_queries)
                    ])
                )

    labels_arr = np.array(labels)
    masks_arr = np.stack(masks)
    has_masks = masks_arr.max() > 0

    results = {"streams": {}, "n_queries": n_queries}
    for name, slot in streams.items():
        entry = {"image_auroc": compute_image_auroc(labels_arr, np.array(slot["image"]))}
        if has_masks:
            maps = np.stack(slot["maps"])
            entry["pixel_auroc"] = compute_pixel_auroc(masks_arr, maps)
            try:
                entry["pro"] = compute_pro(masks_arr, maps)
            except Exception:
                entry["pro"] = float("nan")
        results["streams"][name] = entry

    if saliency_maps and has_masks:
        sal = np.stack(saliency_maps)
        results["saliency_pixel_auroc"] = compute_pixel_auroc(masks_arr, sal)
        qm = np.stack(query_maps)                                  # (images, Q, H, W)
        results["per_query_pixel_auroc"] = [
            compute_pixel_auroc(masks_arr, qm[:, q]) for q in range(qm.shape[1])
        ]
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="D1: what do the query tokens know?")
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["val", "heldout", "all"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--smooth-sigma", type=float, default=None)
    parser.add_argument("--json", type=str, default=None,
                        help="write results here, for before/after comparison")
    parser.add_argument("--compare", type=str, default=None,
                        help="a previous --json file to diff against")
    args = parser.parse_args()

    cfg = load_config()
    cfg["dataset"]["category"] = args.category
    logger = get_logger("D1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or checkpoint_path(cfg, args.category)
    image_size, patch_size = cfg["vit"]["image_size"], cfg["vit"]["patch_size"]
    smooth = (
        args.smooth_sigma if args.smooth_sigma is not None
        else float(cfg.get("scoring", {}).get("smooth_sigma", 0.0))
    )

    probe = MVTecDataset(root=cfg["dataset"]["root"], category=args.category, split="test",
                         image_size=image_size, patch_size=patch_size, synthetic_method=None)
    dataset = MVTecDataset(root=cfg["dataset"]["root"], category=args.category, split="test",
                           image_size=image_size, patch_size=patch_size, synthetic_method=None,
                           subset_indices=split_indices(len(probe), args.split))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    logger.info(f"category={args.category} split={args.split} images={len(dataset)}")

    model = build_model(cfg, checkpoint, device)
    r = diagnose(model, loader, device, image_size, patch_size, smooth, logger)

    print(f"\n{'=' * 84}")
    print(f"D1  WHAT THE QUERY TOKENS KNOW  —  {args.category}  (split={args.split})")
    print(f"{'=' * 84}")
    print(f"{'score stream':<28}{'image AUROC':>14}{'pixel AUROC':>14}{'PRO':>10}")
    print("-" * 84)
    for name, e in r["streams"].items():
        print(f"{name:<28}{e['image_auroc']:>14.4f}"
              f"{e.get('pixel_auroc', float('nan')):>14.4f}{e.get('pro', float('nan')):>10.4f}")

    sal = r.get("saliency_pixel_auroc")
    per_q = r.get("per_query_pixel_auroc")
    if sal is None or per_q is None:
        print("\n(no attention returned — model did not provide patch_query_attention)")
        print(f"{'=' * 84}\n")
        return

    print(f"\nQUERY ATTENTION AS A LOCALISER  ({r['n_queries']} queries)")
    print("-" * 84)
    print(f"  max-over-queries saliency, pixel AUROC : {sal:.4f}")
    arr = np.array(per_q)
    order = np.argsort(-arr)
    print(f"  best query  #{order[0]:<3d} {arr[order[0]]:.4f}      "
          f"worst query #{order[-1]:<3d} {arr[order[-1]]:.4f}")
    print(f"  mean {arr.mean():.4f}   std {arr.std():.4f}   "
          f"(std is how SPECIALISED the queries are)")
    print(f"  top 5: " + "  ".join(f"#{q}={arr[q]:.3f}" for q in order[:5]))
    print("-" * 84)
    best = arr[order[0]]
    if best > 0.85:
        verdict = "a query localises defects well — groundable for language"
    elif best > 0.65:
        verdict = "partial spatial signal — usable but weak grounding"
    else:
        verdict = "NO usable spatial signal in any single query"
    print(f"  verdict: {verdict}")

    payload = {
        "category": args.category,
        "split": args.split,
        "checkpoint": checkpoint,
        "streams": r["streams"],
        "saliency_pixel_auroc": sal,
        "per_query_pixel_auroc": per_q,
        "best_query": int(order[0]),
        "best_query_auroc": float(best),
        "query_mean": float(arr.mean()),
        "query_std": float(arr.std()),
    }

    if args.compare:
        import json as _json

        with open(args.compare) as fh:
            before = _json.load(fh)
        print(f"\nBEFORE vs AFTER  (before: {args.compare})")
        print("-" * 84)
        print(f"{'metric':<34}{'before':>12}{'after':>12}{'delta':>12}")
        rows = [
            ("image AUROC (TOTAL)", before["streams"]["TOTAL"]["image_auroc"],
             r["streams"]["TOTAL"]["image_auroc"]),
            ("pixel AUROC (TOTAL)", before["streams"]["TOTAL"].get("pixel_auroc"),
             r["streams"]["TOTAL"].get("pixel_auroc")),
            ("PRO (TOTAL)", before["streams"]["TOTAL"].get("pro"),
             r["streams"]["TOTAL"].get("pro")),
            ("query saliency pixel AUROC", before["saliency_pixel_auroc"], sal),
            ("best-query pixel AUROC", before["best_query_auroc"], float(best)),
            ("query specialisation (std)", before["query_std"], float(arr.std())),
        ]
        for name, b, a in rows:
            if b is None or a is None:
                continue
            flag = ""
            if "AUROC (TOTAL)" in name or "PRO" in name:
                flag = "  <-- DETECTION DEGRADED" if a < b - 0.005 else ""
            print(f"{name:<34}{b:>12.4f}{a:>12.4f}{a - b:>+12.4f}{flag}")
        print("-" * 84)

    if args.json:
        import json as _json

        with open(args.json, "w") as fh:
            _json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json}")

    print(f"{'=' * 84}\n")


if __name__ == "__main__":
    main()
