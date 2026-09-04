"""SPADE training script.

Trains the BLIP-2 Q-Former (fine-tuned), patch anomaly head, and LLM
projection on normal images with synthetic anomalies.

Usage:
    python train.py
"""

import os
import time

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import numpy as np

from data.mvtec_dataset import MVTecDataset
from losses.total_loss import TotalLoss
from models.builder import build_spade, describe
from models.spade import SPADE
from optim.optimizer import build_optimizer
from optim.scheduler import build_scheduler
from optim.regularizer import clip_gradients
from utils.logging import get_logger, run_log_path
from utils.run_logger import RunLogger
from utils.seed import set_seed
from utils.early_stopping import EarlyStopping
from utils.metrics import compute_image_auroc
from utils.selection import should_save_checkpoint
from models.normal_fit import fit_normal_model
from utils.checkpoint import load_checkpoint_into


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
    run_log=None,
) -> dict[str, float]:
    """Run one training epoch."""
    model.train()
    # Keep ViT in eval mode (frozen + BatchNorm/Dropout behaviour)
    model.vision_encoder.eval()

    running = {"total": 0.0, "patch": 0.0, "detection": 0.0}
    # Add pseudo key if using normal-only training with pseudo-anomaly loss
    if not cfg["synthetic"].get("enabled", False) and cfg.get("loss", {}).get("use_pseudo", False):
        running["pseudo"] = 0.0
    if criterion.grounding_loss_fn is not None:
        running["grounding"] = 0.0
    last_grounding_diagnostics: dict[str, float] = {}
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
        needs_attention = criterion.grounding_loss_fn is not None
        outputs = model(
            images,
            patch_labels=patch_labels,
            update_stats=True,  # Update normal statistics during training
            perturb_epsilon=perturb_epsilon,
            return_attention=needs_attention,
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
            logger.info(f"Epoch {epoch} Step {step}: patches={patch_scores.shape[1]}, "
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
            patch_query_attention=outputs.get("patch_query_attention"),
        )

        # Backward
        optimizer.zero_grad()

        # A loss with no graph means the scored path produced a constant. The
        # usual cause is that the Mahalanobis statistics were never installed,
        # so the scorer returned its uninitialised zeros. backward() reports
        # this as "element 0 of tensors does not require grad", which names
        # neither the component nor the reason -- so say it here instead.
        if not losses["total"].requires_grad:
            stats = model.normal_stats
            raise RuntimeError(
                "the loss has no gradient graph, so nothing can train.\n"
                f"  patch_scores: mean={float(patch_scores.mean()):.6g} "
                f"min={float(patch_scores.min()):.6g} max={float(patch_scores.max()):.6g}\n"
                f"  Mahalanobis initialised: {bool(model.mahalanobis_scorer.is_initialized)}\n"
                f"  statistics: count={int(stats.count)} "
                f"min_samples={stats.min_samples} ready={stats.ready} "
                f"ever_fitted={bool(stats.ever_fitted)} "
                f"update_frequency={stats.update_frequency}\n"
                "If the scorer is uninitialised, no refit has happened yet: check "
                "normal_stats.update_frequency in config/model.yaml and that the "
                "batch supplies more than min_samples normal patches."
            )

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
                running[k] += float(losses[k].detach())
        if "grounding_diagnostics" in losses:
            last_grounding_diagnostics = losses["grounding_diagnostics"]
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

            # Step-level record. Goes to disk unconditionally; wandb mirrors it.
            if run_log is not None:
                log_dict = {
                    "train/loss_total": running["total"] / num_batches,
                    "train/loss_patch": running["patch"] / num_batches,
                    "train/loss_detection": running["detection"] / num_batches,
                    "train/grounding_lambda": criterion.grounding_weight,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/gradient_norm": grad_norm,
                    "train/epoch": epoch,
                    "train/step": current_step,
                }
                # Add pseudo loss if available
                if "pseudo" in running:
                    log_dict["train/loss_pseudo"] = running["pseudo"] / num_batches
                if "grounding" in running:
                    log_dict["train/loss_grounding"] = running["grounding"] / num_batches
                    log_dict.update({f"train/{k}": v for k, v in last_grounding_diagnostics.items()})
                if torch.cuda.is_available():
                    log_dict["sys/gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
                run_log.log(log_dict, step=current_step, event="step")

    summary = {k: v / max(num_batches, 1) for k, v in running.items()}
    if criterion.grounding_loss_fn is not None:
        summary["grounding_lambda"] = criterion.grounding_weight
        summary.update(last_grounding_diagnostics)
    return summary

def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC, or nan when the split has only one class.

    `roc_auc_score` raises in that case. A validation split that happens to be
    all-normal must not abort a 20-epoch run.
    """
    labels = np.asarray(labels)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return compute_image_auroc(labels, np.asarray(scores))


@torch.no_grad()
def validate(
    model: SPADE,
    loader: DataLoader,
    device: torch.device,
    aggregation: str | None = None,
) -> dict[str, float]:
    """Validation on the real MVTec test half used for checkpoint selection.

    `aggregation` is passed explicitly rather than read from the model, so the
    reduction used to RANK epochs is a declared part of the selection protocol
    instead of an implicit consequence of `scoring.image_aggregation`.

    Reports the fused detector AND every stream on its own, plus how far apart
    normal and anomalous images score. The per-stream numbers are what make a
    bad epoch diagnosable: a falling TOTAL beside a healthy local_knn means the
    fusion weights are wrong, not the representation.
    """
    model.eval()

    all_image_labels: list[int] = []
    all_image_scores: list[float] = []
    all_patch_labels: list[np.ndarray] = []
    all_patch_scores: list[np.ndarray] = []
    stream_scores: dict[str, list[float]] = {}

    for batch in tqdm(loader, desc="Validating", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].cpu().numpy().astype(np.int64)
        patch_labels = batch["patch_labels"].cpu().numpy().astype(np.float32)

        outputs = model(images)
        patch_scores = outputs["patch_scores"]
        patch_probs = torch.sigmoid(torch.log1p(patch_scores)).detach().cpu().numpy().astype(np.float32)
        image_scores = model.get_image_score(
            patch_scores, aggregation=aggregation
        ).detach().cpu().numpy().astype(np.float32)

        for name, tensor in outputs.get("score_components", {}).items():
            per_image = model.get_image_score(tensor, aggregation=aggregation)
            stream_scores.setdefault(name, []).extend(
                per_image.detach().cpu().numpy().astype(np.float32).tolist()
            )

        all_image_labels.extend(labels.tolist())
        all_image_scores.extend(image_scores.tolist())
        all_patch_labels.append(patch_labels.reshape(-1))
        all_patch_scores.append(patch_probs.reshape(-1))

    labels_arr = np.array(all_image_labels)
    scores_arr = np.array(all_image_scores)
    patch_labels_arr = np.concatenate(all_patch_labels) if all_patch_labels else np.array([])
    patch_scores_arr = np.concatenate(all_patch_scores) if all_patch_scores else np.array([])

    has_patch_defects = patch_labels_arr.size > 0 and (patch_labels_arr > 0).any()
    metrics = {
        "val/image_auroc": _safe_auroc(labels_arr, scores_arr),
        "val/patch_auroc": (
            _safe_auroc((patch_labels_arr > 0).astype(np.int64), patch_scores_arr)
            if has_patch_defects else float("nan")
        ),
    }

    # Per-stream AUROC: which part of the detector is carrying the result.
    for name, values in stream_scores.items():
        metrics[f"val/stream_{name}"] = _safe_auroc(labels_arr, np.array(values))

    # Separability, the quantity every earlier failure analysis turned on:
    # normal images score X, anomalous score Y, the ratio is defect elevation,
    # and the fraction of anomalous images below the normal p99 is how many are
    # drowned by ordinary variation.
    normal = scores_arr[labels_arr == 0]
    anomalous = scores_arr[labels_arr == 1]
    if normal.size and anomalous.size:
        metrics["val/score_normal_mean"] = float(normal.mean())
        metrics["val/score_anomalous_mean"] = float(anomalous.mean())
        metrics["val/defect_elevation"] = float(anomalous.mean() / max(abs(normal.mean()), 1e-8))
        metrics["val/drowned_fraction"] = float((anomalous < np.percentile(normal, 99)).mean())

    metrics["_scores"] = scores_arr
    metrics["_labels"] = labels_arr
    return metrics


def main() -> None:
    cfg = load_config()
    tcfg = cfg["training"]
    set_seed(tcfg["seed"])

    logger = get_logger(
        "train", log_file=run_log_path("train", cfg["dataset"]["category"])
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Initialize wandb ──
    use_wandb = cfg.get("wandb", {}).get("enabled", False)
    if use_wandb:
        wandb_cfg = cfg["wandb"]
        try:
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
                    "context": cfg.get("context", {}),
                    "scoring": cfg.get("scoring", {}),
                    "normal_stats": cfg.get("normal_stats", {}),
                    "projection": cfg.get("projection", {}),
                    "dataset": cfg.get("dataset", {}),
                    "synthetic": cfg.get("synthetic", {}),
                    "training": tcfg,
                    "loss": cfg.get("loss", {}),
                },
            )
            logger.info(
                f"Weights & Biases: {wandb.run.url if wandb.run else 'initialised'}"
            )
        except Exception as exc:  # noqa: BLE001
            # Telemetry must never abort a training run. A missing API key used
            # to raise UsageError out of main() and kill the job before the
            # first batch.
            logger.warning(
                f"wandb disabled ({type(exc).__name__}: {exc}). "
                "Run `wandb login --relogin`, set WANDB_API_KEY, use "
                "WANDB_MODE=offline, or set wandb.enabled=false to silence this. "
                "Training continues and the full JSONL run record is unaffected; "
                "utils/run_logger.replay_to_wandb can push it up afterwards."
            )
            use_wandb = False

    # ── Structured run record ──
    # Disk is primary: a wandb quota failure at epoch 12 must not cost epochs
    # 1-20. Every metric below lands in JSONL first and is mirrored to wandb
    # inside a try/except that cannot raise into the training loop.
    run_log = RunLogger(
        run_dir=os.path.join("logs", "runs", cfg["dataset"]["category"]),
        wandb_run=wandb if use_wandb else None,
        logger=logger,
        name=f"train_{time.strftime('%Y%m%d-%H%M%S')}",
    )
    run_log.log_config({
        "category": cfg["dataset"]["category"],
        "vit": cfg.get("vit", {}), "context": cfg.get("context", {}),
        "scoring": cfg.get("scoring", {}), "local_pathway": cfg.get("local_pathway", {}),
        "memory_bank": cfg.get("memory_bank", {}), "normal_stats": cfg.get("normal_stats", {}),
        "normal_fit": cfg.get("normal_fit", {}), "loss": cfg.get("loss", {}),
        "training": tcfg, "synthetic": cfg.get("synthetic", {}),
        "frequency": cfg.get("frequency", {}), "projection": cfg.get("projection", {}),
        "device": str(device),
    })

    # ── Train/Val split from train/good (normal-only) ──
    # Unsupervised training: only normal samples, no synthetic anomalies
    use_synthetic = cfg["synthetic"].get("enabled", False)
    grounding_weight = float(cfg.get("loss", {}).get("grounding_weight", 0.0))
    grounding_enabled = grounding_weight > 0.0

    # Synthetic anomalies serve two DIFFERENT purposes and must not be confused:
    #   synthetic.enabled      -> supervised detection (replaces the objective)
    #   grounding_weight > 0   -> masks ONLY, for the auxiliary attention loss;
    #                             the detection objective stays unsupervised
    # At grounding_weight == 0 no synthetic images are generated at all, so the
    # run is bit-identical to a pre-grounding one.
    synthetic_method = None  # None = auto-select per category
    if use_synthetic:
        synthetic_prob = cfg["synthetic"].get("synthetic_prob", 0.2)
    elif grounding_enabled:
        synthetic_prob = float(cfg["loss"].get("grounding_synthetic_prob", 0.5))
    else:
        synthetic_prob = 0.0
    
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

    # A clean view of train/good for the end-of-training normal-model fit: no
    # synthetic anomalies, no shuffling, so the fit sees exactly the normal
    # distribution and is reproducible.
    fit_dataset = MVTecDataset(
        root=cfg["dataset"]["root"],
        category=cfg["dataset"]["category"],
        split="train",
        image_size=cfg["vit"]["image_size"],
        patch_size=cfg["vit"]["patch_size"],
        synthetic_method=None,
        synthetic_prob=0.0,
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
    model = build_spade(cfg, device=device)
    logger.info(f"model: {describe(model)}")
    logger.info(
        "frequency stream: "
        + ("enabled" if model.use_frequency else "disabled")
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,}")

    if run_log is not None:
        run_log.log_config({
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
            grounding_weight=grounding_weight,
            grounding_queries=loss_cfg.get("grounding_queries", 4),
            grounding_pos_weight=loss_cfg.get("grounding_pos_weight", "auto"),
        )
        logger.info(
            f"Using Mahalanobis clustering loss for normal-only training "
            f"(var_weight={loss_cfg.get('var_weight', 0.1)}, "
            f"use_pseudo={loss_cfg.get('use_pseudo', False)})"
        )

    # A cheaper fit used for per-epoch checkpoint selection. Selection needs the
    # epochs RANKED correctly, not measured exactly, so it runs on a subsample;
    # the reported numbers come from the full fit after training.
    fit_cfg_early = cfg.get("normal_fit", {})
    selection_fit_patches = int(fit_cfg_early.get("selection_max_patches", 100_000))
    selection_fit_loader = DataLoader(
        fit_dataset, batch_size=tcfg["batch_size"], shuffle=False, num_workers=0
    )

    # ── Training loop ──
    # Organize checkpoints by category
    category = cfg["dataset"]["category"]
    checkpoint_dir = os.path.join(tcfg["checkpoint_dir"], category)
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
    best_val_image_auroc = -float("inf")
    best_val_patch_auroc = -float("inf")
    # The secondary metric's best AT THE CURRENT PRIMARY LEVEL. Reset whenever
    # the primary strictly improves, because a higher primary level starts a
    # fresh race on the secondary.
    best_secondary_at_primary = -float("inf")
    best_epoch = -1

    def write_checkpoint(path, epoch, image_auroc, patch_auroc):
        """Save everything except the frozen backbone, restored from BLIP-2.

        The previous allowlist silently dropped state added later (e.g. the
        stream scale buffers), producing checkpoints that could not reproduce
        their own scores.
        """
        torch.save({
            "epoch": epoch,
            # the metrics OF THIS MODEL, not the running maxima — otherwise a
            # checkpoint advertises a score it does not have
            "val_image_auroc": float(image_auroc),
            "val_patch_auroc": float(patch_auroc),
            "selection_metric": selection_metric,
            "selection_config": selection_config,
            "model_state_dict": {
                k: v for k, v in model.state_dict().items()
                if not k.startswith("vision_encoder.")
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "blip2_model_name": cfg["blip2"]["model_name"],
                "llm_embed_dim": cfg["projection"]["output_dim"],
                "vit": cfg["vit"],
                "context": cfg.get("context", {}),
                "scoring": cfg["scoring"],
                "normal_stats": cfg["normal_stats"],
                "descriptor_dim": model.descriptor_dim,
                "num_patches": model.num_patches,
            },
        }, path)
    
    # ── Early Stopping ──
    ckpt_cfg = tcfg.get("checkpoint", {})
    selection_metric = ckpt_cfg.get("selection_metric", "image_auroc")
    selection_min_delta = float(ckpt_cfg.get("selection_min_delta", 0.0))
    selection_tie_tol = float(ckpt_cfg.get("selection_tie_tol", 1e-6))

    # Which detector decides "best". The scored model now has a local kNN stream
    # that does not exist until the normal model is fitted -- and the fit happens
    # after training. Selecting on the contextual stream alone would rank epochs
    # by a DIFFERENT detector from the one we report, so the saved checkpoint
    # need not be the best one for the detector that ships.
    # ── selection protocol, made explicit and recorded (P3) ──
    # The detector that ranks epochs must be stated, not inferred, and must
    # travel with the checkpoint so evaluation can detect a mismatch.
    selection_aggregation = ckpt_cfg.get("selection_aggregation", "max")
    if selection_aggregation not in ("topk_mean", "max"):
        raise ValueError(
            f"selection_aggregation must be 'topk_mean' or 'max', got "
            f"{selection_aggregation!r}"
        )
    selection_config = {
        "detector": ckpt_cfg.get("selection_detector", "full"),
        "metric": selection_metric,
        "aggregation": selection_aggregation,
        "stream_weights": {
            "local_knn": float(cfg["scoring"].get("w_local", 1.0)),
            "contextual_mahalanobis": float(cfg["scoring"].get("beta", 0.9)),
            "frequency": float(cfg["scoring"].get("gamma", 0.1)),
        },
        "fit_max_patches": int(cfg.get("normal_fit", {}).get("selection_max_patches", 100_000)),
        "final_fit_max_patches": int(cfg.get("normal_fit", {}).get("max_patches", 500_000)),
        "coreset_ratio": float(cfg.get("memory_bank", {}).get("coreset_ratio", 0.01)),
        "local_source": cfg.get("local_pathway", {}).get("source", "fused"),
        "seed": int(tcfg["seed"]),
    }
    selection_detector = selection_config["detector"]
    if selection_detector not in ("full", "contextual"):
        raise ValueError(
            f"training.checkpoint.selection_detector must be 'full' or "
            f"'contextual', got {selection_detector!r}"
        )
    if selection_metric not in ("image_auroc", "patch_auroc"):
        raise ValueError(
            f"training.checkpoint.selection_metric must be 'image_auroc' or "
            f"'patch_auroc', got {selection_metric!r}"
        )
    if grounding_enabled:
        logger.info(
            f"QUERY GROUNDING ON: lambda={grounding_weight}, "
            f"{cfg['loss'].get('grounding_queries', 4)} reserved queries, "
            f"synthetic_prob={synthetic_prob} (masks only — the detection "
            f"objective and the normal-only statistics are unchanged)"
        )
    else:
        logger.info("query grounding OFF (lambda=0) — exact pre-grounding baseline")

    logger.info(
        f"checkpoint selection: {selection_metric} (min_delta={selection_min_delta}), "
        f"ties within {selection_tie_tol:g} broken by the other metric; "
        f"spade_last.pt is always written for fixed-budget comparisons"
    )
    weights = selection_config["stream_weights"]
    logger.info(
        "SELECTION DETECTOR (recorded in every checkpoint):\n"
        f"    detector      : {selection_config['detector']}\n"
        f"    aggregation   : {selection_aggregation}\n"
        f"    stream weights: local={weights['local_knn']} "
        f"contextual={weights['contextual_mahalanobis']} "
        f"frequency={weights['frequency']}   <- PRIORS, not tuned\n"
        f"    local source  : {selection_config['local_source']}\n"
        f"    fit patches   : {selection_config['fit_max_patches']} for selection, "
        f"{selection_config['final_fit_max_patches']} for the final fit\n"
        "    APPROXIMATION: the per-epoch bank is built from the smaller sample, so it is\n"
        "    correspondingly smaller than the final one. Epoch RANKING is assumed to\n"
        "    transfer across bank sizes; that assumption is untested."
    )

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

    last_fit_report: dict[str, float] = {}
    for epoch in range(1, tcfg["epochs"] + 1):
        # The streaming statistics describe the CURRENT feature space. Carrying
        # them across epochs would mix descriptors produced by different weights
        # into one covariance -- the same staleness the estimator exists to
        # remove, just at epoch granularity.
        model.normal_stats.reset()
        if getattr(model, "freq_normal_stats", None) is not None:
            model.freq_normal_stats.reset()

        metrics = train_one_epoch(
            model, loader, criterion, optimizer, scheduler, device, epoch, cfg, logger, run_log,
        )
        epoch_line = (
            f"Epoch {epoch}/{tcfg['epochs']} — "
            f"loss: {metrics['total']:.4f} | "
            f"detection: {metrics['detection']:.4f} | "
            f"patch: {metrics['patch']:.4f}"
        )
        if "grounding" in metrics:
            epoch_line += (
                f" | grounding: {metrics['grounding']:.4f} (lambda={criterion.grounding_weight})"
                f" | attn mass normal/anomalous: "
                f"{metrics.get('grounding/mass_on_normal', float('nan')):.3f}/"
                f"{metrics.get('grounding/mass_on_anomalous', float('nan')):.3f}"
            )
        logger.info(epoch_line)

        # ── Validation (real test anomalies or synthetic) ──
        if val_loader is not None:
            # Fit the normal model BEFORE validating, so validate() scores with
            # the same detector we report. validate() calls model(images), which
            # picks up whatever streams are fitted -- so this one call is what
            # aligns selection with the shipped detector.
            #
            # It also fixes the statistics used for the epoch's val numbers: the
            # EMA-over-a-20k-deque path is replaced by a closed-form fit over the
            # training set every epoch, not just at the end.
            if selection_detector == "full":
                last_fit_report = fit_normal_model(
                    model, selection_fit_loader, device,
                    max_patches=selection_fit_patches,
                    seed=int(tcfg["seed"]), logger=None,
                )
                logger.info(
                    f"  geometry: local norm {last_fit_report.get('local_norm', float('nan')):.3f}, "
                    f"eff.rank {last_fit_report.get('local_effective_rank', float('nan')):.1f}"
                    f"/{model.local_dim}, "
                    f"fusion gain {last_fit_report.get('fusion_gain', float('nan')):.4f}, "
                    f"bank {int(last_fit_report.get('bank_size', 0))}"
                )
            val_metrics = validate(
                model, val_loader, device, aggregation=selection_aggregation
            )
            logger.info(
                f"Val — image_auroc: {val_metrics['val/image_auroc']:.4f} | "
                f"patch_auroc: {val_metrics['val/patch_auroc']:.4f}"
            )
        else:
            val_metrics = {"val/image_auroc": float("nan"), "val/patch_auroc": float("nan")}

        # ── epoch record: losses, every validation metric, the normal-model
        # fit report, and the score distributions ──
        epoch_step = epoch * len(loader)
        log_dict = {f"epoch/loss_{k}": v for k, v in metrics.items()
                    if isinstance(v, (int, float))}
        log_dict["epoch/epoch"] = epoch
        log_dict.update({
            k: v for k, v in val_metrics.items()
            if not k.startswith("_") and isinstance(v, (int, float))
        })
        log_dict.update({f"fit/{k}": v for k, v in last_fit_report.items()})
        run_log.log(log_dict, step=epoch_step, event="epoch")

        # Score distributions: normal vs anomalous, separately, so a collapse in
        # separability is visible before it shows up in AUROC.
        scores = val_metrics.get("_scores")
        labels_v = val_metrics.get("_labels")
        if scores is not None and labels_v is not None and len(scores):
            run_log.log_histogram("dist/image_scores_normal", scores[labels_v == 0], epoch_step)
            if (labels_v == 1).any():
                run_log.log_histogram("dist/image_scores_anomalous", scores[labels_v == 1], epoch_step)

        logger.info(
            "  val streams: "
            + "  ".join(
                f"{k.replace('val/stream_', '')}={v:.4f}"
                for k, v in sorted(val_metrics.items()) if k.startswith("val/stream_")
            )
            + (f"   elevation={val_metrics['val/defect_elevation']:.2f}x"
               if "val/defect_elevation" in val_metrics else "")
        )

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
                    run_log.log(
                        {"early_stopping/triggered": 1, "early_stopping/epoch": epoch},
                        event="early_stopping",
                    )
                    break

        # ── Checkpoint Saving Logic ──
        current_image = val_metrics["val/image_auroc"]
        current_patch = val_metrics["val/patch_auroc"]

        if val_loader is not None:
            primary_is_image = selection_metric == "image_auroc"
            primary = current_image if primary_is_image else current_patch
            secondary = current_patch if primary_is_image else current_image
            secondary_name = "patch_auroc" if primary_is_image else "image_auroc"

            # A NaN secondary must not be able to win a tie-break, but it must
            # also not block selection on a perfectly valid primary.
            if np.isnan(secondary):
                secondary = -float("inf")

            if np.isnan(primary):
                logger.info(
                    f"{selection_metric} is NaN this epoch; checkpoint not saved"
                )
            else:
                best_primary = (
                    best_val_image_auroc if primary_is_image else best_val_patch_auroc
                )

                # Selection is lexicographic on (primary, secondary).
                #
                # Selecting on the primary ALONE deadlocks the moment it
                # saturates: wood reaches val image AUROC 1.0000 at epoch 1, no
                # later epoch can exceed it, and every subsequent epoch of
                # training is discarded — the run keeps the epoch-1 weights
                # while appearing to train for 20. Ranking ties by the secondary
                # metric (0.7329 on that same epoch, nowhere near saturated)
                # keeps selection live without ever lowering the primary bar,
                # which is what the earlier ratchet bug did.
                save, improved = should_save_checkpoint(
                    primary, secondary, best_primary, best_secondary_at_primary,
                    min_delta=selection_min_delta, tie_tol=selection_tie_tol,
                )
                # Strictly below the bar, versus level with it — the two
                # non-saving cases have different explanations and must not
                # report the same message.
                tied = (not improved) and primary >= best_primary - selection_tie_tol

                if save:
                    # Neither all-time best is ever lowered.
                    best_val_image_auroc = max(best_val_image_auroc, current_image)
                    best_val_patch_auroc = max(best_val_patch_auroc, current_patch)
                    best_secondary_at_primary = (
                        secondary if improved
                        else max(best_secondary_at_primary, secondary)
                    )
                    best_epoch = epoch

                    best_path = os.path.join(checkpoint_dir, "spade_best.pt")
                    write_checkpoint(best_path, epoch, current_image, current_patch)

                    reason = (
                        f"new best {selection_metric}={primary:.4f}" if improved
                        else f"{selection_metric} tied at {primary:.4f}, "
                             f"{secondary_name} improved to {secondary:.4f}"
                    )
                    logger.info(
                        f"{reason} @ epoch {epoch} → "
                        f"image_auroc={best_val_image_auroc:.4f}, "
                        f"patch_auroc={best_val_patch_auroc:.4f} → saved {best_path}"
                    )
                    run_log.summary("best_val_image_auroc", float(best_val_image_auroc))
                    run_log.summary("best_val_patch_auroc", float(best_val_patch_auroc))
                    run_log.summary("best_epoch", int(best_epoch))
                    run_log.log(
                        {"selection/saved": 1, "selection/epoch": epoch,
                         "selection/primary": float(primary), "selection/secondary": float(secondary)},
                        event="selection",
                    )
                else:
                    detail = (
                        f"did not beat best {best_primary:.4f}" if not tied
                        else f"tied best {best_primary:.4f} but {secondary_name}="
                             f"{secondary:.4f} did not beat {best_secondary_at_primary:.4f}"
                    )
                    logger.info(
                        f"{selection_metric}={primary:.4f} {detail}; checkpoint not saved"
                    )

        # The final-epoch weights, saved unconditionally. Comparing two training
        # runs through `spade_best.pt` confounds the thing being compared with
        # which epoch each run happened to peak on; `spade_last.pt` gives every
        # arm of an ablation an identical update budget.
        write_checkpoint(
            os.path.join(checkpoint_dir, "spade_last.pt"),
            epoch, current_image, current_patch,
        )

    # ── Fit the normal model over the FULL training set ──────────────────
    # Until here the Mahalanobis statistics came from a 20k rolling deque --
    # roughly the last 20 images of the last epoch -- EMA'd across overlapping
    # windows, and the memory bank did not exist at all. Both are fitted once,
    # in closed form, from every normal patch.
    #
    # Done per checkpoint rather than once, because spade_best.pt holds
    # different weights from the final epoch: a bank fitted for one set of
    # weights does not describe the other.
    fit_loader = DataLoader(
        fit_dataset, batch_size=tcfg["batch_size"], shuffle=False, num_workers=0
    )
    fit_cfg = cfg.get("normal_fit", {})

    for name in ("spade_last.pt", "spade_best.pt"):
        path = os.path.join(checkpoint_dir, name)
        if not os.path.exists(path):
            continue
        logger.info(f"Fitting the normal model for {name} ...")

        payload = torch.load(path, map_location=device, weights_only=False)
        load_checkpoint_into(
            model, payload["model_state_dict"], logger=logger, context=name
        )
        report = fit_normal_model(
            model, fit_loader, device,
            max_patches=int(fit_cfg.get("max_patches", 500_000)),
            seed=int(tcfg["seed"]),
            logger=logger,
        )

        payload["model_state_dict"] = {
            k: v for k, v in model.state_dict().items()
            if not k.startswith("vision_encoder.")
        }
        payload["normal_fit"] = report
        torch.save(payload, path)
        logger.info(f"  {name}: {report}")
        run_log.log(
            {f"final_fit/{name.replace('.pt', '')}/{k}": v for k, v in report.items()},
            event="final_fit",
        )

    run_log.summary("final_bank_size", float(last_fit_report.get("bank_size", 0)))
    run_log.finish()
    logger.info(f"Training complete. Run record: {run_log.path}")


if __name__ == "__main__":
    main()
