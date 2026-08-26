"""Fit position-conditioned Mahalanobis statistics from train/good.

Closed-form, single pass, no gradients. Produces per-position means and one
pooled within-position covariance (see models/positional_mahalanobis.py for why
that split is the sample-efficient choice at ~250 images/category).

Writes a sidecar next to the checkpoint so released checkpoints stay untouched:

    checkpoints/<category>/positional_stats.pt

Usage:
    python scripts/fit_positional_statistics.py --category transistor
    python scripts/fit_positional_statistics.py --category cable --shrinkage 0.1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.transforms import get_eval_transforms  # noqa: E402
from models.positional_mahalanobis import PositionalStatsAccumulator  # noqa: E402
from scripts.fit_fine_statistics import build_model, load_config  # noqa: E402
from utils.logging import get_logger  # noqa: E402


@torch.no_grad()
def fit(
    model,
    image_paths: list[str],
    image_size: int,
    device: torch.device,
    logger,
    batch_size: int = 8,
    regularization: float = 1e-4,
    shrinkage: float = 0.0,
) -> dict:
    from PIL import Image as PILImage

    transform = get_eval_transforms(image_size)
    accumulator = None

    batch: list[torch.Tensor] = []
    processed = 0

    def flush(batch_tensors):
        nonlocal accumulator, processed
        if not batch_tensors:
            return
        images = torch.stack(batch_tensors).to(device)
        embeds = model.vision_encoder(images)[:, 1:, :].float()  # (B, N, D)
        if accumulator is None:
            _, n_positions, feature_dim = embeds.shape
            logger.info(f"feature dim {feature_dim}, {n_positions} positions")
            accumulator = PositionalStatsAccumulator(
                feature_dim=feature_dim, num_positions=n_positions, device=device
            )
        accumulator.update(embeds)
        processed += len(batch_tensors)

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            logger.warning(f"unreadable, skipping: {path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        batch.append(transform(PILImage.fromarray(img)))
        if len(batch) == batch_size:
            flush(batch)
            batch = []
            if processed % 40 == 0:
                logger.info(f"  {processed}/{len(image_paths)} images")
    flush(batch)

    if accumulator is None:
        raise RuntimeError("no usable training images")

    logger.info(f"fitting statistics from {processed} images")
    stats = accumulator.finalize(regularization=regularization, shrinkage=shrinkage)

    ratio = stats["samples_per_dim"]
    logger.info(
        f"pooled covariance: {stats['n_images'] * stats['n_positions']} samples "
        f"for {stats['feature_dim']} dims ({ratio:.1f} per dim)"
    )
    if ratio < 10:
        logger.warning(
            f"only {ratio:.1f} samples per dimension — the covariance is poorly "
            "conditioned. Consider --shrinkage 0.1 or higher."
        )
    logger.info(
        f"per-position means: {stats['n_images']} samples each "
        f"({'sufficient' if stats['n_images'] >= 50 else 'FEW — means will be noisy'})"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit per-position Mahalanobis statistics")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument(
        "--shrinkage", type=float, default=0.0,
        help="Ledoit-Wolf style shrinkage toward a scaled identity, in [0, 1]",
    )
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    logger = get_logger("fit_positional")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    category = args.category or cfg["dataset"]["category"]
    checkpoint = args.checkpoint or f"checkpoints/{category}/spade_best.pt"
    image_size = cfg["vit"]["image_size"]

    train_dir = Path(cfg["dataset"]["root"]) / category / "train" / "good"
    image_paths = sorted(str(p) for p in train_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"no training images under {train_dir}")
    if args.limit_images:
        image_paths = image_paths[: args.limit_images]

    logger.info(f"category={category}  images={len(image_paths)}  device={device}")
    logger.info(f"checkpoint={checkpoint}  shrinkage={args.shrinkage}")

    model = build_model(cfg, checkpoint, device)
    stats = fit(
        model=model,
        image_paths=image_paths,
        image_size=image_size,
        device=device,
        logger=logger,
        batch_size=args.batch_size,
        regularization=args.regularization,
        shrinkage=args.shrinkage,
    )
    stats["meta"] = {
        "category": category,
        "image_size": image_size,
        "checkpoint": checkpoint,
        "regularization": args.regularization,
        "shrinkage": args.shrinkage,
        "n_train_images": len(image_paths),
    }

    out = args.output or f"checkpoints/{category}/positional_stats.pt"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, out)
    logger.info(f"saved -> {out}")
    logger.info("the released checkpoint was NOT modified")


if __name__ == "__main__":
    main()
