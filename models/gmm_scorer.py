"""Gaussian Mixture Model scoring for anomaly detection."""

from __future__ import annotations

import io
import pickle

import numpy as np
import torch
import torch.nn as nn

from utils.logging import get_logger

try:
    from sklearn.mixture import GaussianMixture
except ImportError:  # pragma: no cover
    GaussianMixture = None  # type: ignore[assignment]


logger = get_logger("gmm_scorer")
_SCORE_MAX = 1.0e6


class GMMScorer(nn.Module):
    """Fit a Gaussian mixture on normal patches and score by negative log-likelihood."""

    def __init__(
        self,
        feature_dim: int,
        n_components: int = 8,
        covariance_type: str = "full",
        reg_covar: float = 1e-4,
        gamma: float = 1.0,
        max_iter: int = 200,
        random_state: int | None = 42,
        fit_max_samples: int | None = 15000,
        fit_subsample_seed: int = 42,
    ):
        super().__init__()
        if GaussianMixture is None:
            raise ImportError("GMMScorer requires scikit-learn: pip install scikit-learn")

        self.feature_dim = int(feature_dim)
        self.n_components = int(n_components)
        self.covariance_type = covariance_type
        self.reg_covar = float(reg_covar)
        self.gamma = float(gamma)
        self.max_iter = int(max_iter)
        self.random_state = random_state

        if fit_max_samples is None or fit_max_samples == 0:
            self.fit_max_samples = None
        else:
            self.fit_max_samples = int(fit_max_samples)
        self.fit_subsample_seed = int(fit_subsample_seed)

        self.register_buffer("is_initialized", torch.tensor(False, dtype=torch.bool))
        self._gmm: GaussianMixture | None = None

    def get_extra_state(self) -> object:
        """Store fitted sklearn object inside checkpoint."""
        if self._gmm is None:
            return None
        buf = io.BytesIO()
        pickle.dump(self._gmm, buf, protocol=pickle.HIGHEST_PROTOCOL)
        return torch.frombuffer(bytearray(buf.getvalue()), dtype=torch.uint8).clone()

    def set_extra_state(self, state: object) -> None:
        if state is None:
            self._gmm = None
            self.is_initialized.fill_(False)
            return
        try:
            raw = state.detach().cpu().numpy().tobytes() if isinstance(state, torch.Tensor) else state
            self._gmm = pickle.loads(bytes(raw))
            self.is_initialized.fill_(True)
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to restore GMM from checkpoint: %s", e)
            self._gmm = None
            self.is_initialized.fill_(False)

    def _fit_gmm_with_fallbacks(self, X: np.ndarray, k_eff: int) -> GaussianMixture | None:
        attempts: list[tuple[str, float, int, str]] = []
        base_cov = self.covariance_type
        base_reg = float(self.reg_covar)

        for reg in [base_reg, 1e-3, 1e-2, 5e-2, 1e-1, 1.0]:
            attempts.append((base_cov, reg, k_eff, "kmeans"))
        for reg in [1e-2, 1e-1, 1.0]:
            attempts.append((base_cov, reg, k_eff, "random"))
            attempts.append(("diag", reg, k_eff, "kmeans"))
        for k_try in (min(4, k_eff), 2, 1):
            if 1 <= k_try <= k_eff:
                attempts.append((base_cov, 1e-1, k_try, "kmeans"))
                attempts.append(("diag", 1e-1, k_try, "kmeans"))

        seen: set[tuple[str, float, int, str]] = set()
        last_err: Exception | None = None
        for cov_type, reg, k, init_p in attempts:
            key = (cov_type, reg, k, init_p)
            if key in seen:
                continue
            seen.add(key)
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=cov_type,
                    reg_covar=reg,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    init_params=init_p,
                    n_init=1,
                )
                gmm.fit(X)
                logger.info(
                    "GMM converged (cov=%s, reg_covar=%.4g, K=%d, init=%s)",
                    cov_type,
                    reg,
                    k,
                    init_p,
                )
                return gmm
            except Exception as e:
                last_err = e
                continue

        if last_err is not None:
            logger.warning("GMM last error: %s", last_err)
        return None

    def fit(self, normal_patches: torch.Tensor) -> None:
        if normal_patches.numel() == 0:
            logger.warning("GMMScorer.fit: empty normal_patches; skipping")
            return

        X = normal_patches.detach().float().cpu().numpy()
        n_total, d = X.shape
        if d != self.feature_dim:
            raise ValueError(f"Expected feature_dim {self.feature_dim}, got {d}")

        cap = self.fit_max_samples
        if cap is not None and cap > 0 and n_total > cap:
            rng = np.random.default_rng(self.fit_subsample_seed)
            idx = rng.choice(n_total, size=cap, replace=False)
            X = X[idx]
            logger.info(
                "GMM fit subsample: using %d / %d normal patches (seed=%d)",
                cap,
                n_total,
                self.fit_subsample_seed,
            )

        n_samples, d = X.shape
        k_eff = min(self.n_components, max(1, n_samples - 1))
        logger.info(
            "Fitting GMM with K=%d components on N=%d patches (feature_dim=%d)...",
            k_eff,
            n_samples,
            d,
        )
        Xf = np.asarray(X, dtype=np.float64)
        gmm = self._fit_gmm_with_fallbacks(Xf, k_eff)
        if gmm is None:
            logger.error("GMM fit failed — scorer will output zeros until next fit.")
            self._gmm = None
            self.is_initialized.fill_(False)
            return

        self._gmm = gmm
        self.is_initialized.fill_(True)
        try:
            logger.info("GMM fit OK — AIC=%.4f, BIC=%.4f", float(gmm.aic(Xf)), float(gmm.bic(Xf)))
        except Exception:
            logger.info("GMM fit OK (AIC/BIC unavailable)")

    def update_statistics(self, normal_patches: torch.Tensor, momentum: float | None = None) -> None:
        del momentum
        self.fit(normal_patches)

    def forward(self, patch_embeds: torch.Tensor, apply_amplification: bool = True) -> torch.Tensor:
        if not self.is_initialized.item() or self._gmm is None:
            return torch.zeros(
                patch_embeds.shape[0],
                patch_embeds.shape[1],
                device=patch_embeds.device,
                dtype=patch_embeds.dtype,
            )

        B, N, D = patch_embeds.shape
        if D != self.feature_dim:
            raise ValueError(f"Expected feature_dim {self.feature_dim}, got {D}")

        x = patch_embeds.detach().float().cpu().numpy().reshape(-1, D)
        try:
            logp = self._gmm.score_samples(x)
        except Exception as e:  # pragma: no cover
            logger.warning("GMM score_samples failed: %s; returning zeros", e)
            return torch.zeros(B, N, device=patch_embeds.device, dtype=patch_embeds.dtype)

        nll = -np.asarray(logp, dtype=np.float64)
        nll = np.clip(nll, 0.0, _SCORE_MAX)
        scores = torch.as_tensor(nll, device=patch_embeds.device, dtype=patch_embeds.dtype).view(B, N)
        if apply_amplification and self.gamma != 1.0:
            scores = torch.pow(scores + 1e-8, self.gamma)
        scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
        return scores
