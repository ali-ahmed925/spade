"""Gradient clipping utility."""

import torch.nn as nn


def clip_gradients(model: nn.Module, max_norm: float = 1.0) -> float:
    """Clip gradients by global norm.

    Args:
        model: model whose trainable gradients to clip.
        max_norm: maximum gradient norm.

    Returns:
        Total gradient norm before clipping.
    """
    params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not params:
        return 0.0
    total_norm = nn.utils.clip_grad_norm_(params, max_norm)
    return float(total_norm)



