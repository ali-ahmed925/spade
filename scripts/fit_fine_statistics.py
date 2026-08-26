"""Fit fine-scale Mahalanobis statistics for dense tiling / region refinement.

WHY THIS EXISTS
---------------
Patch statistics do not transfer across scale. mu and Sigma^-1 in the released
checkpoints were estimated from patches of *whole images resized to 224*. A
512x512 crop resized to 224 is a 2x magnification: different texture frequency,
different blur, different feature distribution. Scoring zoomed crops against
whole-image statistics produces distances that look plausible and mean nothing.

So the fine scale gets its own statistics, estimated the same closed-form way
(mean + covariance over normal patches) from train/good only, in the SAME
contextual-descriptor space the model scores. No gradients, no optimizer, no
labels.

OUTPUT
------
A sidecar file next to the checkpoint:

    checkpoints/<category>/fine_stats_g<grid>_o<overlap>.pt

Sidecar rather than mutating the checkpoint: the baseline checkpoints must stay
byte-identical so the reported baseline numbers remain reproducible.

USAGE
-----
    python scripts/fit_fine_statistics.py --category wood \
        --checkpoint checkpoints/wood/spade_best.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.transforms import get_eval_transforms  # noqa: E402
from models.builder import checkpoint_path, load_spade  # noqa: E402
from models.spade import SPADE  # noqa: E402
from models.tiling import crops_from_image, tile_boxes  # noqa: E402
from utils.logging import get_logger  # noqa: E402


def load_config() -> dict:
    cfg = {}
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
    for name in ("model", "data", "train"):
        with open(os.path.join(config_dir, f"{name}.yaml")) as f:
            cfg.update(yaml.safe_load(f))
    return cfg


def build_model(cfg: dict, checkpoint: str, device: torch.device) -> SPADE:
    """Build SPADE exactly as eval.py does, so the scoring path is identical.

    Thin wrapper over models.builder.load_spade, kept because several diagnostic
    scripts import this name.
    """
    model, _ = load_spade(cfg, checkpoint, device=device)
    return model


@torch.no_grad()
def fit(
    model: SPADE,
    image_paths: list[str],
    grid: int,
    overlap: float,
    image_size: int,
    max_patches: int,
    device: torch.device,
    logger,
    seed: int = 42,
) -> dict:
    """Collect normal-patch features at the tile scale and fit mu / Sigma^-1."""
    transform = get_eval_transforms(image_size)
    rng = np.random.default_rng(seed)

    spatial_chunks: list[torch.Tensor] = []
    freq_chunks: list[torch.Tensor] = []
    n_spatial = 0
    use_freq = getattr(model, "use_frequency", False) and model.freq_extractor is not None

    for idx, path in enumerate(image_paths):
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            logger.warning(f"unreadable, skipping: {path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        native = image_rgb.shape[0]

        boxes = tile_boxes(native, grid=grid, overlap=overlap)
        crops = crops_from_image(image_rgb, boxes, out_size=image_size)

        from PIL import Image as PILImage

        batch = torch.stack([transform(PILImage.fromarray(c)) for c in crops]).to(device)

        # The scorer consumes CONTEXTUAL DESCRIPTORS, not raw ViT embeddings, so
        # the statistics must be fitted in that same space or the swapped-in
        # sidecar would have the wrong width entirely.
        embeds = model.build_descriptors(batch)["descriptors"].float()  # (T, N, C)
        flat = embeds.reshape(-1, embeds.shape[-1])

        # Subsample per image so one image cannot dominate the covariance.
        per_image_cap = max(1, max_patches // max(len(image_paths), 1))
        if flat.shape[0] > per_image_cap:
            sel = rng.choice(flat.shape[0], size=per_image_cap, replace=False)
            flat = flat[torch.from_numpy(sel).to(flat.device)]
        spatial_chunks.append(flat.cpu())
        n_spatial += flat.shape[0]

        if use_freq:
            from utils.patch_extraction import extract_image_patches_from_tensor

            patches = extract_image_patches_from_tensor(batch, patch_size=model_patch_size(model))
            feats = model.freq_extractor(torch.from_numpy(patches).to(device))
            feats = feats.reshape(-1, feats.shape[-1]).float()
            if feats.shape[0] > per_image_cap:
                sel = rng.choice(feats.shape[0], size=per_image_cap, replace=False)
                feats = feats[torch.from_numpy(sel).to(feats.device)]
            freq_chunks.append(feats.cpu())

        if (idx + 1) % 20 == 0:
            logger.info(f"  {idx + 1}/{len(image_paths)} images, {n_spatial} patches")

    if not spatial_chunks:
        raise RuntimeError("no usable training images found")

    stats: dict[str, dict] = {}

    spatial = torch.cat(spatial_chunks).to(device)
    logger.info(f"fitting spatial statistics on {spatial.shape[0]} patches, dim {spatial.shape[1]}")
    _warn_if_ill_conditioned(spatial, logger, "spatial")
    model.mahalanobis_scorer.update_statistics(spatial)
    stats["spatial"] = {
        "mu": model.mahalanobis_scorer.mu.detach().cpu().clone(),
        "sigma_inv": model.mahalanobis_scorer.sigma_inv.detach().cpu().clone(),
        "n_patches": int(spatial.shape[0]),
    }

    if use_freq and freq_chunks:
        freq = torch.cat(freq_chunks).to(device)
        logger.info(f"fitting frequency statistics on {freq.shape[0]} patches, dim {freq.shape[1]}")
        model.freq_mahalanobis_scorer.update_statistics(freq)
        stats["frequency"] = {
            "mu": model.freq_mahalanobis_scorer.mu.detach().cpu().clone(),
            "sigma_inv": model.freq_mahalanobis_scorer.sigma_inv.detach().cpu().clone(),
            "n_patches": int(freq.shape[0]),
        }

    return stats


def model_patch_size(model: SPADE) -> int:
    return int(model.vision_encoder.vision_model.config.patch_size)


def _warn_if_ill_conditioned(features: torch.Tensor, logger, name: str) -> None:
    """A covariance fitted from too few samples inverts into noise amplification."""
    n, d = features.shape
    ratio = n / d
    if ratio < 10:
        logger.warning(
            f"{name}: {n} samples for {d} dimensions ({ratio:.1f} per dim). The "
            "covariance estimate is poorly conditioned and Sigma^-1 will amplify "
            "noise directions. Raise --max-patches, or apply shrinkage/PCA before "
            "drawing conclusions from these distances."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit fine-scale Mahalanobis statistics")
    parser.add_argument("--category", type=str, default=None, help="defaults to config/data.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="defaults to checkpoints/<cat>/spade_best.pt")
    parser.add_argument("--grid", type=int, default=3, help="tiles per axis")
    parser.add_argument("--overlap", type=float, default=0.5, help="tile overlap fraction")
    parser.add_argument("--max-patches", type=int, default=50000, help="total patches used for fitting")
    parser.add_argument("--limit-images", type=int, default=None, help="cap training images (debug)")
    parser.add_argument("--output", type=str, default=None, help="override output path")
    args = parser.parse_args()

    cfg = load_config()
    logger = get_logger("fit_fine_stats")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    category = args.category or cfg["dataset"]["category"]
    checkpoint = args.checkpoint or checkpoint_path(cfg, category)
    image_size = cfg["vit"]["image_size"]

    train_dir = Path(cfg["dataset"]["root"]) / category / "train" / "good"
    image_paths = sorted(str(p) for p in train_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"no training images under {train_dir}")
    if args.limit_images:
        image_paths = image_paths[: args.limit_images]

    logger.info(f"category={category}  images={len(image_paths)}  grid={args.grid}  overlap={args.overlap}")
    logger.info(f"checkpoint={checkpoint}  device={device}")

    model = build_model(cfg, checkpoint, device)
    stats = fit(
        model=model,
        image_paths=image_paths,
        grid=args.grid,
        overlap=args.overlap,
        image_size=image_size,
        max_patches=args.max_patches,
        device=device,
        logger=logger,
    )

    stats["meta"] = {
        "category": category,
        "grid": args.grid,
        "overlap": args.overlap,
        "image_size": image_size,
        "checkpoint": checkpoint,
        "n_train_images": len(image_paths),
    }

    out = args.output or checkpoint_path(cfg, category, f"fine_stats_g{args.grid}_o{args.overlap}.pt")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, out)
    logger.info(f"saved fine statistics -> {out}")
    logger.info("the released checkpoint was NOT modified")


if __name__ == "__main__":
    main()
