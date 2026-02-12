"""Learning rate scheduler with linear warmup + cosine decay."""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
) -> LambdaLR:
    """Cosine annealing with linear warmup.

    Args:
        optimizer: the optimizer to schedule.
        warmup_epochs: number of warmup epochs.
        total_epochs: total training epochs.
        steps_per_epoch: number of steps (batches) per epoch.

    Returns:
        LambdaLR scheduler (step-level, not epoch-level).
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)



