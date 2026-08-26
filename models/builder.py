"""Single construction path for SPADE.

train.py, eval.py, infer.py, app.py and the scripts each used to build the model
with their own copy of a ~20-line argument list. They drifted: a key added to
config/model.yaml reached whichever entry points someone remembered to edit.
With the feature-space redesign introducing several new keys, that duplication
is a liability, so construction lives here and everything calls it.
"""

from __future__ import annotations

import torch

from models.spade import SPADE
from utils.checkpoint import load_checkpoint_into


def build_spade(cfg: dict, device: torch.device | str = "cpu", blip2_model=None) -> SPADE:
    """Construct SPADE from a merged config dict (model + data + train/llm)."""
    vit = cfg["vit"]
    ctx = cfg.get("context", {})
    scoring = cfg.get("scoring", {})
    stats = cfg.get("normal_stats", {})

    model = SPADE(
        blip2_model_name=cfg["blip2"]["model_name"],
        llm_embed_dim=cfg["projection"]["output_dim"],
        image_size=vit["image_size"],
        feature_layers=tuple(vit.get("feature_layers", (20, 30))),
        fusion_proj_dim=vit.get("fusion_proj_dim", 256),
        context_hidden_dim=ctx.get("hidden_dim", 256),
        context_heads=ctx.get("n_heads", 8),
        context_dropout=ctx.get("dropout", 0.0),
        score_beta=scoring.get("beta", 0.9),
        score_gamma=scoring.get("gamma", 0.1),
        mahalanobis_gamma=float(scoring.get("mahalanobis_gamma", 1.0)),
        mahalanobis_reg=float(scoring.get("mahalanobis_reg", 1e-4)),
        normalize_streams=scoring.get("normalize_streams", True),
        normal_stats_buffer_size=stats.get("buffer_size", 20000),
        normal_stats_update_frequency=stats.get("update_frequency", 100),
        projection_trainable=cfg.get("projection", {}).get("trainable", False),
        blip2_model=blip2_model,
    )

    if cfg.get("frequency", {}).get("enabled", False):
        freq = cfg["frequency"]
        model.enable_frequency_features(
            freq_num_bands=freq.get("num_bands", 6),
            freq_use_phase=freq.get("use_phase", True),
            freq_feature_dim=freq.get("feature_dim", 32),
            score_gamma=scoring.get("gamma", 0.1),
        )

    return model.to(device)


def load_spade(
    cfg: dict,
    checkpoint_path: str,
    device: torch.device | str = "cpu",
    logger=None,
) -> tuple[SPADE, dict]:
    """Build SPADE and load a checkpoint into it.

    `model.to(device)` runs AFTER load_state_dict because the frequency stream's
    submodules are created on CPU and their buffers would otherwise stay there.

    Returns (model, checkpoint_metadata).
    """
    model = build_spade(cfg, device=device)

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = state.get("model_state_dict", state)
    load_checkpoint_into(model, state_dict, logger=logger, context=checkpoint_path)

    model.to(device)
    model.eval()

    meta = {k: v for k, v in state.items() if k not in ("model_state_dict", "optimizer_state_dict")} \
        if isinstance(state, dict) else {}
    if logger is not None and meta:
        logger.info(f"checkpoint epoch={meta.get('epoch')} "
                    f"val_image_auroc={meta.get('val_image_auroc')} "
                    f"val_patch_auroc={meta.get('val_patch_auroc')}")
    return model, meta


def describe(model: SPADE) -> str:
    """One-line summary of the descriptor pipeline, for startup logs."""
    enc = model.vision_encoder
    return (
        f"ViT-G {enc.image_size}px "
        f"(pos-emb {'interpolated from ' + str(enc.native_image_size) if enc.needs_interpolation else 'native'}) "
        f"-> blocks {list(enc.feature_layers)} of {enc.num_blocks} "
        f"-> {enc.grid_size}x{enc.grid_size}={enc.num_patches} patches "
        f"-> fusion {model.fusion.output_dim}d "
        f"-> Q-Former {model.qformer.num_queries} queries "
        f"-> contextual descriptor {model.descriptor_dim}d "
        f"-> Mahalanobis"
    )
