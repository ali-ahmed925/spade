"""Exact streaming Gaussian statistics, with no lag-exploitable degree of freedom.

THE PROBLEM THIS REPLACES
-------------------------
The detection loss minimises s(x) = (x-mu)' Sigma^-1 (x-mu) over normal patches.
When (mu, Sigma) are estimated from the same distribution the x's come from,

    E[s] = trace(Sigma^-1 Sigma) = D

exactly, for ANY distribution and ANY invertible reparameterisation x -> Ax.
The mean term is therefore a constant and carries no gradient at all.

It only becomes non-constant when the statistics LAG. The previous
implementation refit every 100 steps from `deque(maxlen=20000)` -- roughly the
last 20 images -- via an EMA of window-covariances that inverted `sigma_inv`
and re-inverted the result on every update. Between refits, scaling the
features by alpha scales s by alpha^2, so

    mean(s) = alpha^2 * D    ->    minimised by alpha -> 0

an unbounded degenerate descent direction that teaches the model nothing about
anomalies. Measured: halving the features between refits dropped the loss from
31.996 to 7.999 at D=32.

THE FIX HAS TWO PARTS
---------------------
(A) Remove the scale degree of freedom entirely, in SPADE's constructor: the
    scored descriptor is `out_norm(...)`, a LayerNorm, and with its affine
    parameters frozen at (weight=1, bias=0) every descriptor satisfies
    ||x||_2 = sqrt(D) EXACTLY. No parameter anywhere can rescale it, so
    alpha == 1 by construction and the degenerate direction does not exist,
    at zero compute cost and independent of refit frequency.

(B) Shrink the remaining lag, which is what this class is for. Exact sufficient
    statistics (n, sum x, sum x x') cost D + D^2 floats -- 1 MB at D=512 -- and
    O(n D^2) per batch, about 537 MFLOP for a 2048-patch batch. That is cheap
    enough to refit EVERY step, from every patch seen this epoch rather than
    the last twenty images, with a single Cholesky and no EMA.

What (A) cannot remove is a residual directional effect: the model may still
redistribute variance toward directions where a lagging Sigma^-1 is small. That
is bounded -- on a fixed-radius sphere s lies in [lambda_min D, lambda_max D] --
and (B) shrinks it further. It is not claimed to be zero.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StreamingGaussianEstimator(nn.Module):
    """Exact running (mu, Sigma^-1) from sufficient statistics.

    Accumulates count, first moment and second moment. Nothing is approximated:
    a refit at any point returns exactly the mean and covariance of every sample
    accumulated since the last `reset()`.
    """

    def __init__(
        self,
        feature_dim: int,
        regularization: float = 1e-4,
        update_frequency: int = 1,
        min_samples_factor: float = 2.0,
    ):
        """
        Args:
            update_frequency: steps between refits. 1 is affordable with exact
                accumulators (one Cholesky, ~45 MFLOP at D=512) and is the
                default, because every step of lag is a step of exploitability.
            min_samples_factor: refuse to refit until count > factor * D. A
                covariance from fewer samples than dimensions is singular, and
                the resulting scores would be noise rather than merely noisy.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.regularization = regularization
        self.update_frequency = update_frequency
        self.min_samples = int(min_samples_factor * feature_dim)

        self.register_buffer("count", torch.zeros(()))
        # Whether statistics have EVER been installed. The first refit must not
        # wait for a multiple of update_frequency: until it happens the scorer
        # returns constant zeros, the loss has no graph, and backward() dies
        # with "element 0 of tensors does not require grad" -- an error that
        # says nothing about the real cause. A stale update_frequency in config
        # produced exactly that.
        self.register_buffer("ever_fitted", torch.tensor(False))
        self.register_buffer("sum_x", torch.zeros(feature_dim))
        self.register_buffer("sum_outer", torch.zeros(feature_dim, feature_dim))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    def reset(self) -> None:
        """Clear the accumulators. Called at the start of each epoch, so the
        statistics describe the CURRENT feature space rather than a mixture of
        every epoch's."""
        self.count.zero_()
        self.sum_x.zero_()
        self.sum_outer.zero_()
        self.step.zero_()
        # `ever_fitted` is deliberately NOT cleared: the scorer keeps the last
        # good statistics across the epoch boundary, so scores stay defined
        # while the new epoch's accumulators fill up.

    @property
    def ready(self) -> bool:
        return bool(self.count > self.min_samples)

    @torch.no_grad()
    def update(
        self,
        features: torch.Tensor,
        patch_labels: torch.Tensor | None = None,
    ) -> int:
        """Accumulate normal patches.

        Args:
            features: (B, N, D) or (M, D) patch features.
            patch_labels: (B, N) or (M,); patches with a non-zero label are
                EXCLUDED. Synthetic anomalies are used only for their masks, and
                folding one into the normal statistics would teach the model
                that the anomaly is normal.

        Returns:
            number of samples accumulated from this call.
        """
        flat = features.reshape(-1, self.feature_dim).detach()
        if patch_labels is not None:
            keep = patch_labels.reshape(-1) == 0
            flat = flat[keep.to(flat.device)]
        if flat.numel() == 0:
            return 0

        x = flat.to(self.sum_x.dtype)
        self.count += x.shape[0]
        self.sum_x += x.sum(dim=0)
        self.sum_outer += x.T @ x
        return int(x.shape[0])

    @torch.no_grad()
    def statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, Sigma) from the accumulators, exactly.

        Sigma = (sum x x' - n mu mu') / (n - 1) + reg I, which is algebraically
        identical to centring first, but needs one pass instead of two.
        """
        if not self.ready:
            raise RuntimeError(
                f"only {int(self.count)} samples accumulated; need more than "
                f"{self.min_samples} to determine a {self.feature_dim}-d covariance"
            )
        n = self.count
        mu = self.sum_x / n
        sigma = (self.sum_outer - n * torch.outer(mu, mu)) / (n - 1)
        sigma = sigma + self.regularization * torch.eye(
            self.feature_dim, device=sigma.device, dtype=sigma.dtype
        )
        # Symmetrise: the outer-product accumulation is symmetric in exact
        # arithmetic but drifts in float, and Cholesky is unforgiving about it.
        return mu, 0.5 * (sigma + sigma.T)

    @torch.no_grad()
    def should_refit(self) -> bool:
        """Advance the step counter and say whether this step refits.

        The FIRST refit fires as soon as enough samples exist, regardless of
        update_frequency, because before it the model cannot produce a
        differentiable score at all.
        """
        self.step += 1
        if not self.ready:
            return False
        if not bool(self.ever_fitted):
            self.ever_fitted.fill_(True)
            return True
        return int(self.step) % self.update_frequency == 0

    def extra_repr(self) -> str:
        return (
            f"feature_dim={self.feature_dim}, update_frequency={self.update_frequency}, "
            f"count={int(self.count)}"
        )


def freeze_layernorm_scale(norm: nn.LayerNorm) -> bool:
    """Pin a LayerNorm to (weight=1, bias=0) and stop it learning.

    This is part (A) above. A LayerNorm without a learnable affine emits vectors
    with zero mean and unit variance across the feature axis, so ||x||_2 is
    exactly sqrt(D) for every input -- the global scale of the scored descriptor
    becomes a constant of the architecture rather than something the optimiser
    can drive toward zero to exploit stale statistics.

    Freezing rather than reconstructing the module keeps the checkpoint keys
    intact, so existing checkpoints still load.

    Returns True if anything was frozen.
    """
    if norm is None or not getattr(norm, "elementwise_affine", False):
        return False
    with torch.no_grad():
        norm.weight.fill_(1.0)
        if norm.bias is not None:
            norm.bias.zero_()
    norm.weight.requires_grad_(False)
    if norm.bias is not None:
        norm.bias.requires_grad_(False)
    return True
