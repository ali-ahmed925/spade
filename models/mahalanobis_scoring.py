"""Mahalanobis distance computation for anomaly scoring."""

import torch
import torch.nn as nn


class MahalanobisScoring(nn.Module):
    """Compute Mahalanobis-based anomaly scores."""
    
    def __init__(
        self,
        feature_dim: int,
        regularization: float = 1e-6,
        gamma: float = 1.3,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.regularization = regularization
        self.gamma = gamma
        
        # Register buffers for statistics (updated during training)
        self.register_buffer("mu", torch.zeros(feature_dim))
        self.register_buffer("sigma_inv", torch.eye(feature_dim))
        self.register_buffer("is_initialized", torch.tensor(False))
        
    def update_statistics(
        self,
        normal_patches: torch.Tensor,
        momentum: float = 0.1,
    ):
        """Update μ and Σ from normal patches.
        
        Args:
            normal_patches: (N_normal, D) normal patch embeddings.
            momentum: Exponential moving average momentum.
        """
        if normal_patches.numel() == 0:
            return
        
        # Compute statistics
        mu_new = normal_patches.mean(dim=0)
        centered = normal_patches - mu_new.unsqueeze(0)
        sigma_new = (centered.T @ centered) / (normal_patches.shape[0] - 1)
        
        # Add regularization - ensure all values are Python types
        if isinstance(self.regularization, torch.Tensor):
            reg_value = float(self.regularization.item())
        else:
            reg_value = float(self.regularization)
        
        if isinstance(self.feature_dim, torch.Tensor):
            feature_dim_int = int(self.feature_dim.item())
        else:
            feature_dim_int = int(self.feature_dim)
        
        # Get device and dtype as Python objects
        device = sigma_new.device
        dtype = sigma_new.dtype
        
        # Create identity matrix and add regularization
        eye_matrix = torch.eye(feature_dim_int, device=device, dtype=dtype)
        sigma_new = sigma_new + reg_value * eye_matrix
        
        # Update with momentum
        if not self.is_initialized:
            self.mu.data = mu_new
            self.sigma_inv.data = torch.linalg.inv(sigma_new)
            self.is_initialized.data = torch.tensor(True)
        else:
            self.mu.data = (1 - momentum) * self.mu + momentum * mu_new
            sigma_updated = (1 - momentum) * (
                torch.linalg.inv(self.sigma_inv) 
            ) + momentum * sigma_new
            self.sigma_inv.data = torch.linalg.inv(sigma_updated)
    
    def forward(
        self,
        patch_embeds: torch.Tensor,
        apply_amplification: bool = True,
    ) -> torch.Tensor:
        """Compute Mahalanobis anomaly scores.
        
        Args:
            patch_embeds: (B, N, D) patch embeddings.
            apply_amplification: Whether to apply non-linear amplification.
            
        Returns:
            (B, N) Mahalanobis anomaly scores.
        """
        if not self.is_initialized:
            # Return zeros if statistics not initialized
            return torch.zeros(
                patch_embeds.shape[0], patch_embeds.shape[1],
                device=patch_embeds.device, dtype=patch_embeds.dtype
            )
        
        B, N, D = patch_embeds.shape
        
        # Center patches
        centered = patch_embeds - self.mu.unsqueeze(0).unsqueeze(0)  # (B, N, D)
        
        # Compute Mahalanobis distance: (x - μ)ᵀ Σ⁻¹ (x - μ)
        # Reshape for batch matrix multiplication
        centered_flat = centered.view(B * N, D)  # (B*N, D)
        
        # Compute (x - μ)ᵀ Σ⁻¹ (x - μ) using batched matrix multiplication
        # More numerically stable: compute Σ⁻¹ (x - μ) first, then dot with (x - μ)
        sigma_inv_centered = torch.mm(centered_flat, self.sigma_inv)  # (B*N, D)
        
        # Compute quadratic form: (x - μ)ᵀ Σ⁻¹ (x - μ)
        # This is the dot product of (x - μ) with Σ⁻¹ (x - μ)
        quad_form = (centered_flat * sigma_inv_centered).sum(dim=1)  # (B*N,)
        
        # Ensure non-negative (should always be, but numerical errors can occur)
        quad_form = torch.clamp(quad_form, min=0.0)
        
        mahalanobis_scores = quad_form.view(B, N)  # (B, N)
        
        # Apply non-linear amplification
        if apply_amplification:
            mahalanobis_scores = torch.pow(mahalanobis_scores + 1e-8, self.gamma)  # Add small epsilon for stability
        
        return mahalanobis_scores


    @torch.no_grad()
    def fit_from_normal_patches(self, normal_patches: torch.Tensor) -> dict[str, float]:
        """Fit mu and Sigma once, in closed form, from the FULL normal set.

        This replaces `update_statistics` for the final fit, and exists because
        that method does not estimate the covariance of the normal distribution:

            sigma_updated = (1 - m) * inv(sigma_inv) + m * sigma_new
            sigma_inv     = inv(sigma_updated)

        Every `sigma_new` is a covariance computed around its OWN window mean, so
        the running average accumulates within-window scatter and systematically
        misses the spread BETWEEN windows. The result is a smoothed average of
        local covariances, not the global one -- and it costs two 512x512
        inversions per update, whose numerical error compounds across training.

        Combined with a `deque(maxlen=20000)` holding roughly the last 20 images,
        the shipped statistics described a small, recent, arbitrary slice of the
        training set. This method takes every patch and does one inversion.
        """
        if normal_patches.ndim != 2 or normal_patches.shape[1] != self.feature_dim:
            raise ValueError(
                f"expected (N, {self.feature_dim}) patches, got {tuple(normal_patches.shape)}"
            )
        n_samples = normal_patches.shape[0]
        if n_samples <= self.feature_dim:
            raise ValueError(
                f"{n_samples} samples cannot determine a {self.feature_dim}-d covariance; "
                "fit from more of the training set"
            )

        patches = normal_patches.to(torch.float64)
        mu = patches.mean(dim=0)
        centered = patches - mu
        sigma = (centered.T @ centered) / (n_samples - 1)

        reg = float(self.regularization)
        sigma += reg * torch.eye(self.feature_dim, device=sigma.device, dtype=sigma.dtype)

        # Cholesky is both faster and a genuine positive-definiteness check: if
        # it fails, the covariance is singular and the score would be nonsense
        # rather than merely inaccurate.
        try:
            cholesky = torch.linalg.cholesky(sigma)
            sigma_inv = torch.cholesky_inverse(cholesky)
        except RuntimeError as exc:
            raise RuntimeError(
                f"covariance is not positive definite at regularization={reg}; "
                f"raise scoring.mahalanobis_reg or fit from more patches ({exc})"
            ) from exc

        self.mu.data = mu.to(self.mu.dtype)
        self.sigma_inv.data = sigma_inv.to(self.sigma_inv.dtype)
        self.is_initialized.data = torch.tensor(True, device=self.mu.device)

        return {
            "samples": float(n_samples),
            "condition_number": float(torch.linalg.cond(sigma).item()),
        }
