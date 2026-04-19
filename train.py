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
from utils.early_stopping import EarlyStopping


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
    # Add pseudo key if using normal-only training with pseudo-anomaly loss
    if not cfg["synthetic"].get("enabled", False) and cfg.get("loss", {}).get("use_pseudo", False):
        running["pseudo"] = 0.0
    num_batches = 0
    global_step = (epoch - 1) * len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for step, batch in enumerate(pbar):
        # Clear cache periodically
        if step > 0 and step % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        images = batch["image"].to(device)               # (B, 3, H, W)
        patch_labels = batch["patch_labels"].to(device)   # (B, N)
        labels = batch["label"].to(device).long()        # (B,) 0=clean, 1=synthetic anomaly

        # Forward with patch_labels for statistics update
        outputs = model(
            images,
            patch_labels=patch_labels,
            update_stats=True,  # Update normal statistics during training
        )
        
        # ⚠️ CRITICAL: Verify HPA is actually affecting patch scores during training
        patch_scores = outputs["patch_scores"]
        if step == 0 and epoch == 1:
            import hashlib
            patch_scores_bytes = patch_scores.detach().cpu().numpy().tobytes()
            patch_scores_hash = hashlib.md5(patch_scores_bytes).hexdigest()[:8]
            score_mean = patch_scores.mean().item()
            score_max = patch_scores.max().item()
            score_min = patch_scores.min().item()
            logger.info(f"Epoch {epoch} Step {step}: HPA enabled={model.use_hpa}, "
                  f"scores mean={score_mean:.4f}, min={score_min:.4f}, max={score_max:.4f}, "
                  f"hash={patch_scores_hash}")

        # Loss: With normal-only training, all patch_labels = 0 (normal patches)
        # Loss pushes patch_scores → 0 for normal patches
        # Q-Former learns to represent normal features
        # Mahalanobis learns normal distribution
        # At test time, anomalies have high Mahalanobis distance → high scores → detected
        losses = criterion(
            patch_scores=patch_scores,
            patch_targets=patch_labels,  # All zeros for normal-only training
            query_embeds=outputs["query_embeds"],
            labels=labels,  # All zeros for normal-only training
        )

        # Backward
        optimizer.zero_grad()
        losses["total"].backward()
        grad_norm = clip_gradients(model, cfg["training"]["gradient_clip"])
        optimizer.step()
        scheduler.step()

        # Accumulate (handle optional keys like "pseudo")
        for k in running:
            if k in losses:
                running[k] += losses[k].item()
        num_batches += 1
        
        # Clear intermediate tensors
        del outputs, losses
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        current_step = global_step + step + 1

        if (step + 1) % cfg["training"]["log_interval"] == 0:
            pbar.set_postfix(
                loss=f"{running['total']/num_batches:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                gnorm=f"{grad_norm:.2f}",
            )

            # Log to wandb at step level
            if use_wandb:
                log_dict = {
                    "train/loss_total": running["total"] / num_batches,
                    "train/loss_patch": running["patch"] / num_batches,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/gradient_norm": grad_norm,
                    "train/epoch": epoch,
                    "train/step": current_step,
                }
                # Add pseudo loss if available
                if "pseudo" in running:
                    log_dict["train/loss_pseudo"] = running["pseudo"] / num_batches
                wandb.log(log_dict, step=current_step)

    return {k: v / max(num_batches, 1) for k, v in running.items()}

@torch.no_grad()
def validate(
    model: SPADE,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Validation on real MVTec test split (image AUROC + patch AUROC)."""
    model.eval()
    # Reset verification flag for this validation run
    if hasattr(model, '_verification_run_this_epoch'):
        model._verification_run_this_epoch = False

    all_image_labels: list[int] = []
    all_image_scores: list[float] = []
    all_patch_labels: list[np.ndarray] = []
    all_patch_scores: list[np.ndarray] = []

    for batch_idx, batch in enumerate(tqdm(loader, desc="Validating", leave=False)):
        images = batch["image"].to(device)
        labels = batch["label"].cpu().numpy().astype(np.int64)  # (B,)
        patch_labels = batch["patch_labels"].cpu().numpy().astype(np.float32)  # (B, N)
        patch_labels_tensor = torch.from_numpy(patch_labels).to(device)  # For verification

        # Pass patch_labels for attention verification (only on first batch to avoid too many files)
        if batch_idx == 0 and model.verify_attention:
            outputs = model(images, patch_labels=patch_labels_tensor, update_stats=False)
        else:
            outputs = model(images)  # No update_stats in validation
        # Normalize patch_scores for evaluation
        patch_scores = outputs["patch_scores"]
        normalized_scores = torch.sigmoid(torch.log1p(patch_scores))
        patch_probs = normalized_scores.detach().cpu().numpy().astype(np.float32)  # (B, N)
        image_scores = model.get_image_score(patch_scores).detach().cpu().numpy().astype(np.float32)  # (B,)

        all_image_labels.extend(labels.tolist())
        all_image_scores.extend(image_scores.tolist())
        all_patch_labels.append(patch_labels.reshape(-1))
        all_patch_scores.append(patch_probs.reshape(-1))

    from sklearn.metrics import roc_auc_score

    y_img = np.array(all_image_labels)
    s_img = np.array(all_image_scores)
    y_patch = np.concatenate(all_patch_labels, axis=0)
    s_patch = np.concatenate(all_patch_scores, axis=0)

    # Check for NaN/Inf values and replace them
    if np.any(~np.isfinite(s_img)):
        print(f"WARNING: Found {np.sum(~np.isfinite(s_img))} non-finite image scores, replacing with 0")
        s_img = np.where(np.isfinite(s_img), s_img, 0.0)
    if np.any(~np.isfinite(s_patch)):
        print(f"WARNING: Found {np.sum(~np.isfinite(s_patch))} non-finite patch scores, replacing with 0")
        s_patch = np.where(np.isfinite(s_patch), s_patch, 0.0)

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
                "hpa": cfg.get("hpa", {}),
                "scoring": cfg.get("scoring", {}),
                "normal_stats": cfg.get("normal_stats", {}),
                "projection": cfg.get("projection", {}),
                "dataset": cfg.get("dataset", {}),
                "synthetic": cfg.get("synthetic", {}),
                "training": tcfg,
                "loss": cfg.get("loss", {}),
            },
        )
        logger.info("Initialized Weights & Biases logging")

    # ── Train/Val split from train/good (normal-only) ──
    # Unsupervised training: only normal samples, no synthetic anomalies
    use_synthetic = cfg["synthetic"].get("enabled", False)
    if use_synthetic:
        synthetic_method = None  # None = auto-select based on category
        synthetic_prob = cfg["synthetic"].get("synthetic_prob", 0.2)
    else:
        synthetic_method = None  # None = no synthetic anomalies
        synthetic_prob = 0.0  # No synthetic anomalies
    
    base_train = MVTecDataset(
        root=cfg["dataset"]["root"],
        category=cfg["dataset"]["category"],
        split="train",
        image_size=cfg["vit"]["image_size"],
        patch_size=cfg["vit"]["patch_size"],
        synthetic_method=synthetic_method,  # None = no synthetic anomalies
        synthetic_prob=synthetic_prob,  # 0.0 = no synthetic anomalies
        synthetic_cfg=cfg.get("synthetic", {}),
    )

    val_cfg = cfg.get("validation", {})
    use_real_val = val_cfg.get("use_real_test", True)  # Use real test anomalies for validation
    
    if val_cfg.get("enabled", True):
        if use_real_val:
            # Use real test set anomalies for validation (split test 50/50)
            test_dataset = MVTecDataset(
                root=cfg["dataset"]["root"],
                category=cfg["dataset"]["category"],
                split="test",
                image_size=cfg["vit"]["image_size"],
                patch_size=cfg["vit"]["patch_size"],
                synthetic_method=None,  # No synthetic for test set
                synthetic_prob=0.0,
            )
            # Split test set 50/50 for val/test
            n_test = len(test_dataset)
            val_size = n_test // 2
            rng = np.random.RandomState(int(val_cfg.get("seed", 42)))
            perm = rng.permutation(n_test).tolist()
            val_indices_test = perm[:val_size]
            val_dataset = MVTecDataset(
                root=cfg["dataset"]["root"],
                category=cfg["dataset"]["category"],
                split="test",
                image_size=cfg["vit"]["image_size"],
                patch_size=cfg["vit"]["patch_size"],
                synthetic_method=None,
                synthetic_prob=0.0,
                subset_indices=val_indices_test,
            )
            logger.info(f"Using real test anomalies for validation: {len(val_dataset)} samples (from {n_test} test samples)")
            train_dataset = base_train  # Use all training data
        else:
            # Fallback: synthetic validation from train/good (not recommended for unsupervised)
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
                synthetic_method=synthetic_method,  # None = no synthetic anomalies
                synthetic_prob=synthetic_prob,  # 0.0 = no synthetic anomalies
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
                synthetic_method=val_cfg.get("synthetic_method", None),
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
        logger.info(f"Validation samples: {len(val_dataset)}")
    else:
        val_loader = None

    # ── Clear GPU memory before loading model ──
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        logger.info("Cleared GPU cache before model loading")
    
    # ── Model ──
    model = SPADE(
        blip2_model_name=cfg["blip2"]["model_name"],
        llm_embed_dim=cfg["projection"]["output_dim"],
        # HPA parameters
        hpa_n_max=cfg["hpa"]["n_max"],
        hpa_n_min=cfg["hpa"]["n_min"],
        hpa_t_steps=cfg["hpa"]["t_steps"],
        hpa_w=cfg["hpa"]["w"],
        hpa_p1=cfg["hpa"]["p1"],
        hpa_p2=cfg["hpa"]["p2"],
        # Scoring parameters
        score_alpha=cfg["scoring"]["alpha"],
        score_beta=cfg["scoring"]["beta"],
        score_gamma=cfg["scoring"].get("gamma", 0.25),  # Frequency weight
        score_lambda=cfg["scoring"]["lambda"],
        mahalanobis_gamma=cfg["scoring"]["mahalanobis_gamma"],
        mahalanobis_reg=cfg["scoring"]["mahalanobis_reg"],
        use_mahalanobis=cfg["scoring"].get("use_mahalanobis", True),
        # Normal statistics parameters
        normal_stats_buffer_size=cfg["normal_stats"]["buffer_size"],
        normal_stats_update_frequency=cfg["normal_stats"]["update_frequency"],
        # Attention verification
        verify_attention=cfg["scoring"].get("verify_attention", False),
        # Raw attention mode
        use_raw_attention=cfg["scoring"].get("use_raw_attention", True),
        # Image-level aggregation (quantile / max / topk_mean)
        image_score_mode=(cfg["scoring"].get("image_score") or {}).get("mode", "quantile"),
        image_score_quantile=float((cfg["scoring"].get("image_score") or {}).get("quantile", 0.99)),
        image_score_top_k=int((cfg["scoring"].get("image_score") or {}).get("top_k", 3)),
    ).to(device)

    # Enable/disable HPA based on config
    use_hpa = cfg.get("hpa", {}).get("enabled", True)
    
    # ⚠️ CRITICAL: Force set and verify
    model.use_hpa = use_hpa
    logger.info(f"🔧 SET model.use_hpa = {use_hpa} (type: {type(use_hpa)})")
    
    # Log Mahalanobis status
    if model.use_mahalanobis:
        logger.info("✅ Mahalanobis scoring enabled")
    else:
        logger.info("❌ Mahalanobis scoring DISABLED - using only attention scores")
    logger.info(
        f"Image-level score: mode={model.image_score_mode}, "
        f"quantile={model.image_score_quantile}, top_k={model.image_score_top_k}"
    )
    
    # ⚠️ CRITICAL: Verify it's actually set correctly
    if model.use_hpa != use_hpa:
        logger.error(f"❌ CRITICAL BUG: model.use_hpa ({model.use_hpa}) != config ({use_hpa})! Fixing...")
        model.use_hpa = use_hpa
        logger.info(f"✅ Fixed: model.use_hpa = {model.use_hpa}")
    
    # ⚠️ CRITICAL: Double-check it's a boolean
    if not isinstance(model.use_hpa, bool):
        logger.error(f"❌ CRITICAL BUG: model.use_hpa is not a boolean! Value: {model.use_hpa}, type: {type(model.use_hpa)}")
        model.use_hpa = bool(use_hpa)
        logger.info(f"✅ Fixed: model.use_hpa = {model.use_hpa} (forced to boolean)")
    
    if model.use_hpa:
        logger.info("HPA enabled - queries will be refined through hierarchical patch annealing")
    else:
        logger.info("HPA disabled - queries will attend to all patches directly (no refinement)")
    
    logger.info(f"⚠️ FINAL VERIFICATION: model.use_hpa = {model.use_hpa} (type: {type(model.use_hpa)})")
    
    # Enable frequency features if configured
    if cfg.get("frequency", {}).get("enabled", False):
        model.enable_frequency_features(
            freq_num_bands=cfg["frequency"].get("num_bands", 6),
            freq_use_phase=cfg["frequency"].get("use_phase", True),
            freq_feature_dim=cfg["frequency"].get("feature_dim", 32),
            score_gamma=cfg["scoring"].get("gamma", 0.25),
        )
        logger.info("Frequency features enabled")
    else:
        logger.info("Frequency features disabled")

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
    
    # Select loss based on training mode
    use_synthetic = cfg["synthetic"].get("enabled", False)
    if use_synthetic:
        # Synthetic training: BCE/Focal loss
        criterion = TotalLoss(
            patch_weight=cfg["loss"]["patch_weight"],
            use_focal=cfg["loss"]["use_focal"],
            focal_alpha=cfg["loss"]["focal_alpha"],
            focal_gamma=cfg["loss"]["focal_gamma"],
            use_normal_only=False,
        )
        logger.info("Using BCE/Focal loss for synthetic training")
    else:
        # Normal-only training: Mahalanobis clustering loss
        loss_cfg = cfg.get("loss", {})
        criterion = TotalLoss(
            patch_weight=loss_cfg.get("patch_weight", 1.0),
            use_normal_only=True,
            var_weight=loss_cfg.get("var_weight", 0.1),
            use_pseudo=loss_cfg.get("use_pseudo", False),
            pseudo_epsilon=loss_cfg.get("pseudo_epsilon", 0.01),
            clamp_max=loss_cfg.get("clamp_max", 100.0),
        )
        logger.info(
            f"Using Mahalanobis clustering loss for normal-only training "
            f"(var_weight={loss_cfg.get('var_weight', 0.1)}, "
            f"use_pseudo={loss_cfg.get('use_pseudo', False)})"
        )

    # ── Training loop ──
    # Organize checkpoints by category
    category = cfg["dataset"]["category"]
    checkpoint_dir = os.path.join(tcfg["checkpoint_dir"], category)
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
    best_val_image_auroc = -float("inf")
    best_val_patch_auroc = -float("inf")
    best_epoch = -1
    
    # ── Early Stopping (patience counts epochs without image AUROC improvement) ──
    early_stop_cfg = tcfg.get("early_stopping", {})
    improve_delta = float(early_stop_cfg.get("min_delta", 0.0))
    early_stopper = None
    if early_stop_cfg.get("enabled", False) and val_loader is not None:
        early_stopper = EarlyStopping(
            patience=early_stop_cfg.get("patience", 10),
            mode=early_stop_cfg.get("mode", "max"),
            min_delta=early_stop_cfg.get("min_delta", 0.001),
            verbose=True,
        )
        logger.info(
            f"Early stopping enabled: stop after {early_stopper.patience} epoch(s) without image AUROC "
            f"improvement (> best + {improve_delta})"
        )

    for epoch in range(1, tcfg["epochs"] + 1):
        metrics = train_one_epoch(
            model, loader, criterion, optimizer, scheduler, device, epoch, cfg, logger, use_wandb,
        )
        logger.info(
            f"Epoch {epoch}/{tcfg['epochs']} — "
            f"loss: {metrics['total']:.4f} | "
            f"patch: {metrics['patch']:.4f}"
        )
        
        # Compute Mahalanobis statistics once after first epoch (using all collected normal patches)
        if epoch == 1 and model.use_mahalanobis:
            logger.info("Computing Mahalanobis statistics from all collected normal patches...")
            model.compute_statistics_once(device=device)
            num_spatial = len(model.normal_stats_tracker.normal_patch_buffer)
            logger.info(f"✅ Computed spatial Mahalanobis statistics from {num_spatial} normal patches")
            if model.use_frequency and model.freq_normal_stats_tracker is not None:
                num_freq = len(model.freq_normal_stats_tracker.normal_patch_buffer)
                logger.info(f"✅ Computed frequency Mahalanobis statistics from {num_freq} normal patches")

        # ── Validation (real test anomalies or synthetic) ──
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
            log_dict = {
                "epoch/loss_total": metrics["total"],
                "epoch/loss_patch": metrics["patch"],
                "epoch/epoch": epoch,
                "val/image_auroc": val_metrics["val/image_auroc"],
                "val/patch_auroc": val_metrics["val/patch_auroc"],
            }
            # Add pseudo loss if available
            if "pseudo" in metrics:
                log_dict["epoch/loss_pseudo"] = metrics["pseudo"]
            wandb.log(log_dict, step=epoch * len(loader))

        # ── Checkpoint saving & early stopping (image AUROC only) ──
        # Save when image AUROC improves by more than min_delta; otherwise increment patience (1/epoch toward limit).
        current_image = val_metrics["val/image_auroc"]
        current_patch = val_metrics["val/patch_auroc"]

        if val_loader is not None:
            image_valid = not np.isnan(current_image)
            patch_valid = not np.isnan(current_patch)

            if image_valid:
                image_improved = current_image > best_val_image_auroc + improve_delta

                if image_improved:
                    if early_stopper is not None:
                        early_stopper.counter = 0
                    best_val_image_auroc = current_image
                    if patch_valid:
                        best_val_patch_auroc = current_patch
                    best_epoch = epoch

                    trainable_state = {
                        k: v for k, v in model.state_dict().items()
                        if any(k.startswith(prefix) for prefix in ["qformer.", "projection.", "hpa.", "query_patch_attn.", "mahalanobis_scorer.", "freq_mahalanobis_scorer.", "normal_stats_tracker.", "freq_normal_stats_tracker.", "score_alpha", "score_beta", "score_gamma", "score_lambda"])
                        or k in ["attn_mean", "attn_std", "spatial_mean", "spatial_std", "freq_mean", "freq_std", "_stats_count"]  # Running statistics buffers
                    }

                    best_path = os.path.join(checkpoint_dir, "spade_best.pt")
                    saved_patch = float(current_patch) if patch_valid else float("nan")
                    patch_log = f"{saved_patch:.4f}" if patch_valid else "nan"
                    torch.save({
                        "epoch": epoch,
                        "val_image_auroc": float(best_val_image_auroc),
                        "val_patch_auroc": saved_patch,
                        "model_state_dict": trainable_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": {
                            "blip2_model_name": cfg["blip2"]["model_name"],
                            "llm_embed_dim": cfg["projection"]["output_dim"],
                            "hpa": cfg["hpa"],
                            "scoring": cfg["scoring"],
                            "normal_stats": cfg["normal_stats"],
                        },
                    }, best_path)
                    logger.info(
                        f"New best image AUROC @ epoch {epoch} → "
                        f"image_auroc={best_val_image_auroc:.4f}, "
                        f"patch_auroc={patch_log} → saved {best_path}"
                    )
                    if use_wandb:
                        wandb.run.summary["best_val_image_auroc"] = float(best_val_image_auroc)
                        wandb.run.summary["best_val_patch_auroc"] = saved_patch
                        wandb.run.summary["best_epoch"] = int(best_epoch)
                else:
                    if early_stopper is not None:
                        early_stopper.counter += 1
                        if early_stopper.verbose:
                            thresh = best_val_image_auroc + improve_delta
                            logger.info(
                                f"EarlyStopping: no image AUROC improvement "
                                f"({current_image:.4f} <= {thresh:.4f}) — "
                                f"patience {early_stopper.counter}/{early_stopper.patience}"
                            )
                        if early_stopper.counter >= early_stopper.patience:
                            early_stopper.early_stop = True
                            logger.info(
                                f"Early stopping triggered at epoch {epoch} "
                                f"after {early_stopper.patience} epoch(s) without image AUROC improvement"
                            )
                            if use_wandb:
                                wandb.log({"early_stopping/triggered": True, "early_stopping/epoch": epoch})
                            break
                    if patch_valid and current_patch > best_val_patch_auroc:
                        logger.info(
                            f"Patch AUROC improved ({current_patch:.4f} > {best_val_patch_auroc:.4f}) "
                            f"but image AUROC did not — checkpoint not saved"
                        )

    if use_wandb:
        wandb.finish()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
