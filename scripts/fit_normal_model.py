"""Fit the memory bank and Mahalanobis statistics into an existing checkpoint.

train.py now does this automatically after the last epoch. This exists for
checkpoints trained before that, and for re-fitting after a config change
(a different coreset ratio or neighbourhood size) without retraining.

Reads train/good only. No labels, no synthetic anomalies, no test data.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mvtec_dataset import MVTecDataset  # noqa: E402
from eval import load_config  # noqa: E402
from models.builder import build_spade, checkpoint_path  # noqa: E402
from models.normal_fit import fit_normal_model  # noqa: E402
from utils.checkpoint import load_checkpoint_into  # noqa: E402
from utils.logging import get_logger, run_log_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the normal model into a checkpoint")
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="where to write; defaults to overwriting --checkpoint")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-patches", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    cfg["dataset"]["category"] = args.category
    logger = get_logger("fit_normal", log_file=run_log_path("fit_normal", args.category))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source = args.checkpoint or checkpoint_path(cfg, args.category)
    destination = args.output or source

    dataset = MVTecDataset(
        root=cfg["dataset"]["root"], category=args.category, split="train",
        image_size=cfg["vit"]["image_size"], patch_size=cfg["vit"]["patch_size"],
        synthetic_method=None, synthetic_prob=0.0,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    logger.info(f"category={args.category} train/good images={len(dataset)}")

    model = build_spade(cfg, device=device)
    payload = torch.load(source, map_location=device, weights_only=False)
    load_checkpoint_into(model, payload["model_state_dict"], logger=logger, context=source)

    report = fit_normal_model(
        model, loader, device,
        max_patches=args.max_patches, seed=args.seed, logger=logger,
    )

    payload["model_state_dict"] = {
        k: v for k, v in model.state_dict().items() if not k.startswith("vision_encoder.")
    }
    payload["normal_fit"] = report
    torch.save(payload, destination)
    logger.info(f"wrote {destination}")
    for key, value in report.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()
