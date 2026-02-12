"""Optimizer factory — only trainable parameters."""

import torch.nn as nn
from torch.optim import AdamW


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> AdamW:
    """Create AdamW optimizer for trainable parameters only.

    ViT and LLM are frozen, so their params are automatically excluded.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(params, lr=lr, weight_decay=weight_decay)



