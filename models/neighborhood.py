"""Local neighbourhood aggregation over the patch grid.

WHY
---
PatchCore does not score raw backbone features. Each descriptor is an average
over a 3x3 window of its neighbours on the feature grid, and the paper is
explicit that this matters: it makes a test patch robust to a one-cell
misalignment against the closest normal patch, which is the dominant nuisance
on classes where the object is not pose-aligned.

Our pipeline had no equivalent. Nothing anywhere in the descriptor path pooled
over neighbouring patches, so a screw rotated two degrees relative to every
training image produced descriptors that matched nothing, at every position.

This is deliberately parameter-free. An average pool adds no capacity and
cannot be blamed for a result; it only widens the support each descriptor
summarises. Any gain is attributable to the aggregation itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeighborhoodAggregator(nn.Module):
    """Average-pool each patch descriptor over its k x k grid neighbourhood.

    Input and output are both (B, N, D) with N = grid_size ** 2, so this drops
    into the descriptor path without changing any downstream shape.
    """

    def __init__(self, grid_size: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd so the window is centred, got {kernel_size}"
            )
        self.grid_size = grid_size
        self.kernel_size = kernel_size

    @property
    def enabled(self) -> bool:
        """kernel_size 1 is the identity, and is the ablation control."""
        return self.kernel_size > 1

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (B, N, D) with N == grid_size ** 2.

        Returns:
            (B, N, D) each descriptor averaged over its neighbourhood. Padding is
            'same' by replication of the border via zero-padding's count
            correction -- see the note below.
        """
        if not self.enabled:
            return patches

        b, n, d = patches.shape
        g = self.grid_size
        if n != g * g:
            raise ValueError(f"expected {g * g} patches for a {g}x{g} grid, got {n}")

        grid = patches.transpose(1, 2).reshape(b, d, g, g)
        pad = self.kernel_size // 2

        # count_include_pad=False divides each border cell by the number of REAL
        # neighbours rather than by k*k. With the default (True) every border
        # descriptor would be shrunk toward zero purely for being at the edge,
        # which then reads as anomalous under any distance-based score -- a
        # border artefact that would look exactly like a real detection.
        pooled = F.avg_pool2d(
            grid,
            kernel_size=self.kernel_size,
            stride=1,
            padding=pad,
            count_include_pad=False,
        )
        return pooled.reshape(b, d, n).transpose(1, 2)

    def extra_repr(self) -> str:
        return f"grid_size={self.grid_size}, kernel_size={self.kernel_size}"
