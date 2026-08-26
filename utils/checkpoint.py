"""Checkpoint loading helpers.

`load_state_dict(strict=False)` tolerates missing and unexpected keys but still
RAISES on a shape mismatch. That matters here: the feature-space redesign changes
what Mahalanobis scores (a 512-d contextualised descriptor instead of a 1408-d
raw ViT embedding) and the patch grid (1024 instead of 256), so every statistics
buffer in a pre-redesign checkpoint has the wrong shape.

Rather than let that surface as an opaque RuntimeError deep inside model
loading, incompatible tensors are dropped explicitly and reported. The
statistics they held are refitted from train/good anyway; the trainable weights
that still match are kept.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def filter_compatible_state(
    state_dict: dict[str, torch.Tensor],
    model: nn.Module,
) -> tuple[dict[str, torch.Tensor], list[tuple[str, tuple, tuple]]]:
    """Split a state dict into shape-compatible entries and rejected ones.

    Args:
        state_dict: candidate tensors, e.g. checkpoint["model_state_dict"].
        model: the model they are destined for.

    Returns:
        (compatible, rejected) where rejected is a list of
        (name, checkpoint_shape, model_shape).
    """
    reference = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    rejected: list[tuple[str, tuple, tuple]] = []

    for name, tensor in state_dict.items():
        target = reference.get(name)
        if target is None:
            continue  # unexpected key; strict=False would ignore it anyway
        if tuple(target.shape) != tuple(tensor.shape):
            rejected.append((name, tuple(tensor.shape), tuple(target.shape)))
            continue
        compatible[name] = tensor
    return compatible, rejected


def load_checkpoint_into(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    logger=None,
    context: str = "checkpoint",
) -> dict[str, list]:
    """Load what fits, drop what does not, and say clearly which was which.

    The frozen ViT-G weights are never expected in the file — they come from the
    BLIP-2 download — so their absence is normal and not reported as a problem.

    Returns a report dict with "loaded", "rejected" and "missing" entries.
    """
    compatible, rejected = filter_compatible_state(state_dict, model)
    result = model.load_state_dict(compatible, strict=False)

    missing = [
        k for k in result.missing_keys
        if not k.startswith("vision_encoder.")
    ]

    if logger is not None:
        logger.info(f"{context}: loaded {len(compatible)}/{len(state_dict)} tensors")
        if rejected:
            logger.warning(
                f"{context}: {len(rejected)} tensor(s) dropped for shape mismatch — "
                "this is expected when loading a pre-redesign checkpoint, whose "
                "statistics were fitted in the old descriptor space and must be "
                "refitted:"
            )
            for name, ckpt_shape, model_shape in rejected[:8]:
                logger.warning(f"    {name}: checkpoint {ckpt_shape} vs model {model_shape}")
            if len(rejected) > 8:
                logger.warning(f"    ... and {len(rejected) - 8} more")
        if missing:
            logger.warning(
                f"{context}: {len(missing)} trainable tensor(s) not present in the "
                f"file and left at initialization, e.g. {missing[:5]}"
            )

    return {"loaded": list(compatible), "rejected": rejected, "missing": missing}
