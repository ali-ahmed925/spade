"""SPADE training script.

Trains the BLIP-2 Q-Former (fine-tuned), patch anomaly head, and LLM
projection on normal images with synthetic anomalies.

Usage:
    python train.py
"""

import os

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import numpy as np

from data.mvtec_dataset import MVTecDataset
from losses.total_loss import TotalLoss
from models.spade import SPADE
from optim.optimizer import build_optimizer
from optim.scheduler import build_scheduler
from optim.regularizer import clip_gradients
from utils.logging import get_logger
from utils.seed import set_seed


def load_config() -> dict:
    """Load and merge all config files."""
    cfg = {}
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    for name in ("model", "data", "train"):
        with open(os.path.join(config_dir, f"{name}.yaml")) as f:
            cfg.update(yaml.safe_load(f))
    return cfg


def train_one_epoch(
    model: SPADE,
    loader: DataLoader,
    criterion: TotalLoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    cfg: dict,
    logger,
    use_wandb: bool = False,
) -> dict[str, float]:
    """Run one training epoch."""
    model.train()
    # Keep ViT in eval mode (frozen + BatchNorm/Dropout behaviour)
    model.vision_encoder.eval()

    running = {"total": 0.0, "patch": 0.0}
    num_batches = 0
    global_step = (epoch - 1) * len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for step, batch in enumerate(pbar):
        images = batch["image"].to(device)               # (B, 3, H, W)
        patch_labels = batch["patch_labels"].to(device)   # (B, N)
        labels = batch["label"].to(device).long()        # (B,) 0=clean, 1=synthetic anomaly

        # Forward
        outputs = model(images)

        # Loss
        losses = criterion(
            patch_logits=outputs["patch_logits"],
            patch_targets=patch_labels,
            query_embeds=outputs["query_embeds"],
            labels=labels,
        )

        # Backward
        optimizer.zero_grad()
        losses["total"].backward()
        grad_norm = clip_gradients(model, cfg["training"]["gradient_clip"])
        optimizer.step()
        scheduler.step()

        # Accumulate
        for k in running:
            running[k] += losses[k].item()
        num_batches += 1
        current_step = global_step + step + 1

        if (step + 1) % cfg["training"]["log_interval"] == 0:
            pbar.set_postfix(
                loss=f"{running['total']/num_batches:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                gnorm=f"{grad_norm:.2f}",
            )

            # Log to wandb at step level
            if use_wandb:
                wandb.log({
                    "train/loss_total": running["total"] / num_batches,
                    "train/loss_patch": running["patch"] / num_batches,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/gradient_norm": grad_norm,
                    "train/epoch": epoch,
                    "train/step": current_step,
                }, step=current_step)

    return {k: v / max(num_batches, 1) for k, v in running.items()}

@torch.no_grad()
def validate(
    model: SPADE,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Validation on real MVTec test split (image AUROC + patch AUROC)."""
    model.eval()

    all_image_labels: list[int] = []
    all_image_scores: list[float] = []
    all_patch_labels: list[np.ndarray] = []
    all_patch_scores: list[np.ndarray] = []

    for batch in tqdm(loader, desc="Validating", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].cpu().numpy().astype(np.int64)  # (B,)
        patch_labels = batch["patch_labels"].cpu().numpy().astype(np.float32)  # (B, N)

        outputs = model(images)
        patch_probs = torch.sigmoid(outputs["patch_logits"]).detach().cpu().numpy().astype(np.float32)  # (B, N)
        image_scores = model.get_image_score(outputs["patch_logits"]).detach().cpu().numpy().astype(np.float32)  # (B,)

        all_image_labels.extend(labels.tolist())
        all_image_scores.extend(image_scores.tolist())
        all_patch_labels.append(patch_labels.reshape(-1))
        all_patch_scores.append(patch_probs.reshape(-1))

    from sklearn.metrics import roc_auc_score

    y_img = np.array(all_image_labels)
    s_img = np.array(all_image_scores)
    y_patch = np.concatenate(all_patch_labels, axis=0)
    s_patch = np.concatenate(all_patch_scores, axis=0)

    # AUROC requires both classes present
    metrics: dict[str, float] = {}
    metrics["val/image_auroc"] = float(roc_auc_score(y_img, s_img)) if len(np.unique(y_img)) > 1 else float("nan")
    metrics["val/patch_auroc"] = float(roc_auc_score(y_patch, s_patch)) if len(np.unique(y_patch)) > 1 else float("nan")
    return metrics


def main() -> None:
    cfg = load_config()
    tcfg = cfg["training"]
    set_seed(tcfg["seed"])

    logger = get_logger("train", log_file="logs/train.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Initialize wandb ──
    use_wandb = cfg.get("wandb", {}).get("enabled", False)
    if use_wandb:
        wandb_cfg = cfg["wandb"]
        wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("name"),
            tags=wandb_cfg.get("tags", []),
            notes=wandb_cfg.get("notes"),
            config={
                "model": cfg.get("blip2", {}),
                "vit": cfg.get("vit", {}),
                "qformer": cfg.get("qformer", {}),
                "patch_head": cfg.get("patch_head", {}),
                "projection": cfg.get("projection", {}),
                "dataset": cfg.get("dataset", {}),
                "synthetic": cfg.get("synthetic", {}),
                "training": tcfg,
                "loss": cfg.get("loss", {}),
            },
        )
        logger.info("Initialized Weights & Biases logging")

    # ── Train/Val split from train/good (normal-only) ──
    # synthetic_method=None means auto-select based on category
    base_train = MVTecDataset(
        root=cfg["dataset"]["root"],
        category=cfg["dataset"]["category"],
        split="train",
        image_size=cfg["vit"]["image_size"],
        patch_size=cfg["vit"]["patch_size"],
        synthetic_method=None,  # Auto-select based on category
        synthetic_prob=cfg["synthetic"].get("synthetic_prob", 0.2),
        synthetic_cfg=cfg.get("synthetic", {}),
    )

    val_cfg = cfg.get("validation", {})
    use_synth_val = bool(val_cfg.get("enabled", True)) and val_cfg.get("source", "train_good") == "train_good"
    if use_synth_val:
        n = len(base_train)
        val_size = int(val_cfg.get("size", 40))
        val_size = max(1, min(val_size, n - 1))
        rng = np.random.RandomState(int(val_cfg.get("seed", 42)))
        perm = rng.permutation(n).tolist()
        val_indices = perm[:val_size]
        train_indices = perm[val_size:]
        logger.info(f"Train/Val split from train/good → train={len(train_indices)} val={len(val_indices)}")

        train_dataset = MVTecDataset(
            root=cfg["dataset"]["root"],
            category=cfg["dataset"]["category"],
            split="train",
            image_size=cfg["vit"]["image_size"],
            patch_size=cfg["vit"]["patch_size"],
            synthetic_method=None,  # Auto-select based on category
            synthetic_prob=cfg["synthetic"].get("synthetic_prob", 0.2),
            subset_indices=train_indices,
            deterministic=False,
            synthetic_cfg=cfg.get("synthetic", {}),
        )

        val_dataset = MVTecDataset(
            root=cfg["dataset"]["root"],
            category=cfg["dataset"]["category"],
            split="train",
            image_size=cfg["vit"]["image_size"],
            patch_size=cfg["vit"]["patch_size"],
            synthetic_method=val_cfg.get("synthetic_method", None),  # None = auto-select
            synthetic_prob=float(val_cfg.get("synthetic_prob", 0.2)),
            subset_indices=val_indices,
            deterministic=True,
            synthetic_cfg=cfg.get("synthetic", {}),
            base_seed=int(val_cfg.get("seed", 42)),
        )
    else:
        train_dataset = base_train
        val_dataset = None

    loader = DataLoader(
        train_dataset,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=cfg["dataset"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Training samples: {len(train_dataset)}")

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=tcfg["batch_size"],
            shuffle=False,
            num_workers=cfg["dataset"]["num_workers"],
            pin_memory=True,
        )
        logger.info(f"Validation samples (synthetic from train/good): {len(val_dataset)}")
    else:
        val_loader = None

    # ── Model ──
    model = SPADE(
        blip2_model_name=cfg["blip2"]["model_name"],
        patch_head_hidden=cfg["patch_head"]["hidden_dim"],
        patch_head_dropout=cfg["patch_head"]["dropout"],
        llm_embed_dim=cfg["projection"]["output_dim"],
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,}")

    if use_wandb:
        wandb.config.update({
            "model/trainable_params": trainable,
            "model/total_params": total,
            "model/trainable_ratio": trainable / total,
        })

    # ── Optimizer / Scheduler / Loss ──
    optimizer = build_optimizer(model, lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])
    scheduler = build_scheduler(
        optimizer,
        warmup_epochs=tcfg["warmup_epochs"],
        total_epochs=tcfg["epochs"],
        steps_per_epoch=len(loader),
    )
    criterion = TotalLoss(
        patch_weight=cfg["loss"]["patch_weight"],
        use_focal=cfg["loss"]["use_focal"],
        focal_alpha=cfg["loss"]["focal_alpha"],
        focal_gamma=cfg["loss"]["focal_gamma"],
    )

    # ── Training loop ──
    os.makedirs(tcfg["checkpoint_dir"], exist_ok=True)
    best_val_image_auroc = -float("inf")
    best_epoch = -1

    for epoch in range(1, tcfg["epochs"] + 1):
        metrics = train_one_epoch(
            model, loader, criterion, optimizer, scheduler, device, epoch, cfg, logger, use_wandb,
        )
        logger.info(
            f"Epoch {epoch}/{tcfg['epochs']} — "
            f"loss: {metrics['total']:.4f} | "
            f"patch: {metrics['patch']:.4f}"
        )

        # ── Validation (synthetic from train/good to avoid test leakage) ──
        if val_loader is not None:
            val_metrics = validate(model, val_loader, device)
            logger.info(
                f"Val — image_auroc: {val_metrics['val/image_auroc']:.4f} | "
                f"patch_auroc: {val_metrics['val/patch_auroc']:.4f}"
            )
        else:
            val_metrics = {"val/image_auroc": float("nan"), "val/patch_auroc": float("nan")}

        # Log epoch-level metrics to wandb
        if use_wandb:
            wandb.log({
                "epoch/loss_total": metrics["total"],
                "epoch/loss_patch": metrics["patch"],
                "epoch/epoch": epoch,
                "val/image_auroc": val_metrics["val/image_auroc"],
                "val/patch_auroc": val_metrics["val/patch_auroc"],
            }, step=epoch * len(loader))

        # Save checkpoint ONLY when validation improves (max score wins)
        current = val_metrics["val/image_auroc"]
        if val_loader is not None and (not np.isnan(current)) and current > best_val_image_auroc:
            best_val_image_auroc = current
            best_epoch = epoch

            trainable_state = {
                k: v for k, v in model.state_dict().items()
                if any(k.startswith(prefix) for prefix in ["qformer.", "patch_head.", "projection."])
            }

            best_path = os.path.join(tcfg["checkpoint_dir"], "spade_best.pt")
            torch.save({
                "epoch": epoch,
                "val_image_auroc": float(best_val_image_auroc),
                "model_state_dict": trainable_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "blip2_model_name": cfg["blip2"]["model_name"],
                    "patch_head_hidden": cfg["patch_head"]["hidden_dim"],
                    "patch_head_dropout": cfg["patch_head"]["dropout"],
                    "llm_embed_dim": cfg["projection"]["output_dim"],
                },
            }, best_path)
            logger.info(f"New best val/image_auroc={best_val_image_auroc:.4f} @ epoch {epoch} → saved {best_path}")
            if use_wandb:
                wandb.run.summary["best_val_image_auroc"] = float(best_val_image_auroc)
                wandb.run.summary["best_epoch"] = int(best_epoch)

    if use_wandb:
        wandb.finish()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
