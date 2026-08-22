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

        # Forward with patch_labels for statistics update.
        # perturb_epsilon makes the model score a perturbed copy of the patches
        # so the pseudo-anomaly loss has a real (non-cancelling) gradient.
        perturb_epsilon = None
        if criterion.pseudo_loss_fn is not None:
            perturb_epsilon = criterion.pseudo_loss_fn.epsilon
        outputs = model(
            images,
            patch_labels=patch_labels,
            update_stats=True,  # Update normal statistics during training
            perturb_epsilon=perturb_epsilon,
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
            patch_scores_perturbed=outputs.get("patch_scores_perturbed"),
        )

        # Backward
        optimizer.zero_grad()
        losses["total"].backward()

        # ── Gradient self-check (first step only) ──────────────────────────
        # Fails loudly if any parameter we call "trainable" received no signal.
        # A silent dead path here is how the projection head went a whole
        # project without ever being trained.
        if step == 0 and epoch == 1:
            from utils.grad_audit import classify_parameters, dead_parameters, format_report
            report = classify_parameters(model)
            dead = dead_parameters(report)
            logger.info("\n" + format_report(report))
            if dead:
                raise RuntimeError(
                    f"{len(dead)} trainable parameter tensor(s) received no gradient: "
                    f"{dead[:5]}{'...' if len(dead) > 5 else ''}. "
                    "Either connect them to the loss or freeze them "
                    "(see utils/grad_audit.py)."
                )

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

    all_image_labels: list[int] = []
    all_image_scores: list[float] = []
    all_patch_labels: list[np.ndarray] = []
    all_patch_scores: list[np.ndarray] = []

    for batch in tqdm(loader, desc="Validating", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].cpu().numpy().astype(np.int64)  # (B,)
        patch_labels = batch["patch_labels"].cpu().numpy().astype(np.float32)  # (B, N)

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
        score_lambda=cfg["scoring"]["lambda"],
        mahalanobis_gamma=cfg["scoring"]["mahalanobis_gamma"],
        mahalanobis_reg=cfg["scoring"]["mahalanobis_reg"],
        # Normal statistics parameters
        normal_stats_buffer_size=cfg["normal_stats"]["buffer_size"],
        normal_stats_update_frequency=cfg["normal_stats"]["update_frequency"],
        # Scoring-correctness knobs (default to the corrected behaviour)
        normalize_streams=cfg.get("scoring", {}).get("normalize_streams", True),
        attention_aggregation=cfg.get("scoring", {}).get("attention_aggregation", "logit_mean"),
    ).to(device)

    # Enable/disable HPA based on config
    use_hpa = cfg.get("hpa", {}).get("enabled", True)
    
    # ⚠️ CRITICAL: Force set and verify
    model.use_hpa = use_hpa
    logger.info(f"🔧 SET model.use_hpa = {use_hpa} (type: {type(use_hpa)})")
    
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
            pseudo_margin=loss_cfg.get("pseudo_margin", 0.1),
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
    
    # ── Early Stopping ──
    early_stop_cfg = tcfg.get("early_stopping", {})
    early_stopper = None
    if early_stop_cfg.get("enabled", False) and val_loader is not None:
        early_stopper = EarlyStopping(
            patience=early_stop_cfg.get("patience", 10),
            mode=early_stop_cfg.get("mode", "max"),
            min_delta=early_stop_cfg.get("min_delta", 0.001),
            verbose=True,
        )
        logger.info(f"Early stopping enabled: patience={early_stopper.patience}, mode={early_stopper.mode}")

    for epoch in range(1, tcfg["epochs"] + 1):
        metrics = train_one_epoch(
            model, loader, criterion, optimizer, scheduler, device, epoch, cfg, logger, use_wandb,
        )
        logger.info(
            f"Epoch {epoch}/{tcfg['epochs']} — "
            f"loss: {metrics['total']:.4f} | "
            f"patch: {metrics['patch']:.4f}"
        )

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

        # ── Early Stopping Check ──
        # Use combined metric (average of image and patch AUROC) for early stopping
        if early_stopper is not None and val_loader is not None:
            current_image = val_metrics["val/image_auroc"]
            current_patch = val_metrics["val/patch_auroc"]
            # Use average of both metrics, or image AUROC if patch is NaN
            if not np.isnan(current_image) and not np.isnan(current_patch):
                current_score = (current_image + current_patch) / 2.0
            elif not np.isnan(current_image):
                current_score = current_image
            else:
                current_score = float("nan")
            
            if not np.isnan(current_score):
                if early_stopper(current_score):
                    logger.info(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(combined score: {current_score:.4f} = "
                        f"image:{current_image:.4f} + patch:{current_patch:.4f} / 2)"
                    )
                    if use_wandb:
                        wandb.log({"early_stopping/triggered": True, "early_stopping/epoch": epoch})
                    break

        # ── Checkpoint Saving Logic ──
        # Save checkpoint when:
        # 1. BOTH metrics improve, OR
        # 2. One improves and the other stays constant, OR
        # 3. One improves and the other degrades only slightly (within tolerance)
        current_image = val_metrics["val/image_auroc"]
        current_patch = val_metrics["val/patch_auroc"]
        
        # Get degradation tolerance from config (default 0.02 = 2% acceptable drop)
        degradation_tolerance = tcfg.get("checkpoint", {}).get("degradation_tolerance", 0.02)
        
        if val_loader is not None:
            image_valid = not np.isnan(current_image)
            patch_valid = not np.isnan(current_patch)
            
            if image_valid and patch_valid:
                image_improved = current_image > best_val_image_auroc
                patch_improved = current_patch > best_val_patch_auroc
                image_degraded = current_image < best_val_image_auroc
                patch_degraded = current_patch < best_val_patch_auroc
                image_unchanged = current_image == best_val_image_auroc
                patch_unchanged = current_patch == best_val_patch_auroc
                
                # Check if degradation is within acceptable tolerance
                image_degradation = best_val_image_auroc - current_image if image_degraded else 0.0
                patch_degradation = best_val_patch_auroc - current_patch if patch_degraded else 0.0
                image_degradation_acceptable = image_degradation <= degradation_tolerance
                patch_degradation_acceptable = patch_degradation <= degradation_tolerance
                
                # Save checkpoint if:
                # 1. BOTH metrics improve, OR
                # 2. One improves and the other stays constant, OR
                # 3. One improves and the other degrades only slightly (within tolerance)
                should_save = (
                    (image_improved and patch_improved) or
                    (image_improved and patch_unchanged) or
                    (patch_improved and image_unchanged) or
                    (image_improved and patch_degradation_acceptable) or
                    (patch_improved and image_degradation_acceptable)
                )
                
                if should_save:
                    best_val_image_auroc = current_image
                    best_val_patch_auroc = current_patch
                    best_epoch = epoch

                    # Save everything except the frozen backbone, which is
                    # restored from the BLIP-2 download. The previous allowlist
                    # silently dropped any state added later (e.g. the stream
                    # scale buffers), producing checkpoints that could not
                    # reproduce their own scores.
                    trainable_state = {
                        k: v for k, v in model.state_dict().items()
                        if not k.startswith("vision_encoder.")
                    }

                    best_path = os.path.join(checkpoint_dir, "spade_best.pt")
                    torch.save({
                        "epoch": epoch,
                        "val_image_auroc": float(best_val_image_auroc),
                        "val_patch_auroc": float(best_val_patch_auroc),
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
                        f"New best metrics @ epoch {epoch} → "
                        f"image_auroc={best_val_image_auroc:.4f}, "
                        f"patch_auroc={best_val_patch_auroc:.4f} → saved {best_path}"
                    )
                    if use_wandb:
                        wandb.run.summary["best_val_image_auroc"] = float(best_val_image_auroc)
                        wandb.run.summary["best_val_patch_auroc"] = float(best_val_patch_auroc)
                        wandb.run.summary["best_epoch"] = int(best_epoch)
                elif image_improved or patch_improved:
                    # Log when only one metric improves but the other degrades beyond tolerance
                    if image_improved and patch_degraded and not patch_degradation_acceptable:
                        logger.info(
                            f"Image AUROC improved ({current_image:.4f} > {best_val_image_auroc:.4f}), "
                            f"but patch AUROC degraded too much ({current_patch:.4f} < {best_val_patch_auroc:.4f}, "
                            f"degradation: {patch_degradation:.4f} > tolerance: {degradation_tolerance:.4f}) - "
                            f"checkpoint not saved"
                        )
                    elif patch_improved and image_degraded and not image_degradation_acceptable:
                        logger.info(
                            f"Patch AUROC improved ({current_patch:.4f} > {best_val_patch_auroc:.4f}), "
                            f"but image AUROC degraded too much ({current_image:.4f} < {best_val_image_auroc:.4f}, "
                            f"degradation: {image_degradation:.4f} > tolerance: {degradation_tolerance:.4f}) - "
                            f"checkpoint not saved"
                        )

    if use_wandb:
        wandb.finish()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
