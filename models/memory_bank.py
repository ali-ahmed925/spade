"""Coreset-subsampled memory bank scored by nearest-neighbour distance.

WHY THIS REPLACES A GAUSSIAN
----------------------------
`MahalanobisScoring` models the normal patch distribution with ONE Gaussian:
one mu in R^512 and one Sigma_inv in R^512x512, pooled over every patch of
every position of every training image. A screw image contains thread, head,
shank and dark background, and screws appear at many rotations -- genuinely
different modes. A single Gaussian puts mu in the empty space between them and
widens Sigma until it covers all of them, at which point a real defect sits
comfortably inside the normal ellipsoid. That is the measured signature:
defect elevation 1.5-3x on the classes that fail, against 8-75x where it works.

PaDiM sidesteps this by fitting a separate Gaussian per spatial position.
PatchCore removes the assumption entirely: keep the normal patches themselves
and score by distance to the nearest one. Nothing about the normal set has to
be unimodal, Gaussian, or aligned. That is what this implements.

WHY CORESET RATHER THAN EVERY PATCH
-----------------------------------
Screw has 320 training images at 1024 patches -- 327,680 vectors, 671 MB at
512-d. Greedy k-center subsampling keeps the points that cover the set, so the
bank retains the outlying normal modes that a random sample would drop. Those
modes are exactly what a rare-but-legitimate normal patch needs to match.

FITTED, NOT LEARNED
-------------------
Like the Mahalanobis statistics, the bank is fitted after training rather than
optimised. Until `fit` is called it is unfitted and scores zero, so enabling it
cannot change a training run. Everything it needs is closed-form.
"""

from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def greedy_coreset_indices(
    features: torch.Tensor,
    n_select: int,
    projection_dim: int | None = 128,
    seed: int = 0,
) -> torch.Tensor:
    """Greedy k-center: repeatedly take the point furthest from those chosen.

    Args:
        features: (N, D) candidate vectors.
        n_select: how many to keep.
        projection_dim: run the SELECTION in this many dimensions via a
            Johnson-Lindenstrauss random projection. PatchCore does the same:
            pairwise distances are approximately preserved, and selection over
            327k x 512 becomes tractable. The FULL-dimension vectors are what
            gets stored -- the projection only decides which ones.
        seed: makes the selection reproducible, which matters because the bank
            is part of the checkpoint.

    Returns:
        (n_select,) long tensor of indices into `features`.
    """
    n_total = features.shape[0]
    if n_select >= n_total:
        return torch.arange(n_total, device=features.device)
    if n_select < 1:
        raise ValueError(f"n_select must be >= 1, got {n_select}")

    generator = torch.Generator(device="cpu").manual_seed(seed)

    working = features
    if projection_dim is not None and projection_dim < features.shape[1]:
        projection = torch.randn(
            features.shape[1], projection_dim, generator=generator
        ).to(features.device, features.dtype)
        projection /= projection_dim ** 0.5
        working = features @ projection

    start = int(torch.randint(n_total, (1,), generator=generator).item())
    selected = torch.empty(n_select, dtype=torch.long, device=features.device)
    selected[0] = start

    # min_distance[i] = distance from point i to the nearest selected point.
    min_distance = (working - working[start]).pow(2).sum(dim=1)

    for step in range(1, n_select):
        nxt = int(torch.argmax(min_distance).item())
        selected[step] = nxt
        distance = (working - working[nxt]).pow(2).sum(dim=1)
        min_distance = torch.minimum(min_distance, distance)

    return selected


class CoresetMemoryBank(nn.Module):
    """Nearest-neighbour anomaly scoring against a coreset of normal patches."""

    def __init__(
        self,
        feature_dim: int,
        coreset_ratio: float = 0.01,
        max_size: int = 50000,
        k: int = 1,
        projection_dim: int | None = 128,
        seed: int = 0,
        query_chunk: int = 1024,
    ):
        """
        Args:
            coreset_ratio: fraction of candidate patches to keep.
            max_size: hard cap, so a large category cannot blow up the
                checkpoint or the scoring cost.
            k: how many neighbours to average. k=1 is PatchCore's choice and is
                the most sensitive; larger k trades sensitivity for robustness
                to a single unrepresentative bank vector.
            query_chunk: queries scored per distance matrix. The full
                (n_queries, bank_size) matrix is what dominates memory here.
        """
        super().__init__()
        if not 0.0 < coreset_ratio <= 1.0:
            raise ValueError(f"coreset_ratio must be in (0, 1], got {coreset_ratio}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        self.feature_dim = feature_dim
        self.coreset_ratio = coreset_ratio
        self.max_size = max_size
        self.k = k
        self.projection_dim = projection_dim
        self.seed = seed
        self.query_chunk = query_chunk

        self.register_buffer("bank", torch.zeros(0, feature_dim))
        self.register_buffer("is_fitted", torch.tensor(False))

    # ── fitting ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def fit(self, features: torch.Tensor) -> dict[str, int]:
        """Build the bank from normal patches.

        Args:
            features: (N, D) normal patch descriptors, ideally every one in the
                training set. This is the whole point -- the previous statistics
                path saw a 20k rolling window, roughly the last 20 images.

        Returns:
            {"candidates": N, "selected": bank size}
        """
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"expected (N, {self.feature_dim}) features, got {tuple(features.shape)}"
            )
        if features.shape[0] == 0:
            raise ValueError("cannot fit a memory bank from zero patches")

        target = max(1, min(round(features.shape[0] * self.coreset_ratio), self.max_size))
        indices = greedy_coreset_indices(
            features, target, projection_dim=self.projection_dim, seed=self.seed
        )
        self.bank = features[indices].contiguous().to(self.bank.dtype)
        self.is_fitted = torch.tensor(True, device=self.bank.device)
        return {"candidates": int(features.shape[0]), "selected": int(self.bank.shape[0])}

    def reset(self) -> None:
        self.bank = torch.zeros(0, self.feature_dim, device=self.bank.device, dtype=self.bank.dtype)
        self.is_fitted = torch.tensor(False, device=self.bank.device)

    @property
    def fitted(self) -> bool:
        return bool(self.is_fitted) and self.bank.shape[0] > 0

    # ── scoring ──────────────────────────────────────────────────────────
    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: (B, N, D) patch descriptors to score.

        Returns:
            (B, N) distance to the nearest normal patch. Zeros when unfitted, so
            an unfitted bank contributes nothing rather than raising -- training
            runs before the bank exists.
        """
        b, n, d = queries.shape
        if not self.fitted:
            return torch.zeros(b, n, device=queries.device, dtype=queries.dtype)
        if d != self.feature_dim:
            raise ValueError(f"expected {self.feature_dim}-d queries, got {d}")

        # torch.cdist computes ||a||^2 + ||b||^2 - 2a.b via matrix multiply once
        # the input is large, which is what makes a 25k-vector bank affordable
        # but costs precision near zero: a vector that IS in the bank scores
        # ~5e-4 rather than 0. That is far below the mean nearest-neighbour
        # distance the stream is normalised by, so it does not affect ranking --
        # but it is why the tests assert against a scale rather than exact zero.
        bank = self.bank.to(queries.dtype)
        k = min(self.k, bank.shape[0])
        flat = queries.reshape(b * n, d)

        scores = []
        for start in range(0, flat.shape[0], self.query_chunk):
            chunk = flat[start : start + self.query_chunk]
            distances = torch.cdist(chunk, bank)          # (chunk, bank_size)
            if k == 1:
                scores.append(distances.min(dim=1).values)
            else:
                scores.append(distances.topk(k, dim=1, largest=False).values.mean(dim=1))

        return torch.cat(scores, dim=0).view(b, n)

    @torch.no_grad()
    def patchcore_reweight(
        self,
        queries: torch.Tensor,
        patch_scores: torch.Tensor,
        b: int = 9,
    ) -> torch.Tensor:
        """PatchCore's image-level score: the max patch distance, re-weighted.

        From "Towards Total Recall in Industrial Anomaly Detection" (Roth et al.,
        CVPR 2022), the image score is not simply the maximum patch distance.
        With

            m_test_star = the test patch with the largest NN distance
            m_star      = its nearest neighbour in the memory bank
            s_star      = ||m_test_star - m_star||
            N_b(m_star) = the b nearest bank patches TO m_star

        the reported score is

            s = ( 1 - exp(s_star) / sum_{m in N_b(m_star)} exp(||m_test_star - m||) ) * s_star

        The intent is to discount a large distance when m_star is itself an
        isolated, rarely-matched nominal patch: if the test patch is far from
        everything in that neighbourhood the denominator is large, w approaches
        1, and the score stands. Because m_star is its own nearest neighbour it
        belongs to N_b(m_star), so the denominator always contains the numerator
        and w lies in [0, 1).

        KNOWN DISCREPANCY, deliberately not followed here: anomalib's
        implementation takes the b nearest neighbours of the TEST patch rather
        than of m_star (openvinotoolkit/anomalib issue #286). Reported there as
        making no measurable AUROC difference on their data. This implements the
        PAPER, since that is the published method being compared against.

        The value of b used in the paper's experiments could NOT be verified
        from an authoritative source; 9 is the common implementation default and
        is exposed as a parameter rather than baked in.

        Args:
            queries: (B, N, D) patch descriptors.
            patch_scores: (B, N) their nearest-neighbour distances.

        Returns:
            (B,) re-weighted image scores.
        """
        if not self.fitted:
            return patch_scores.max(dim=1).values

        bank = self.bank.to(queries.dtype)
        b_eff = min(b, bank.shape[0])
        out = []
        for i in range(queries.shape[0]):
            star_idx = int(torch.argmax(patch_scores[i]))
            m_test_star = queries[i, star_idx]                       # (D,)
            s_star = patch_scores[i, star_idx]

            # m_star: nearest bank vector to the max-distance test patch
            to_bank = torch.cdist(m_test_star[None], bank)[0]         # (M,)
            m_star = bank[int(torch.argmin(to_bank))]                 # (D,)

            # N_b(m_star): b nearest bank vectors TO m_star (paper, not anomalib)
            around_star = torch.cdist(m_star[None], bank)[0]          # (M,)
            neighbour_idx = torch.topk(around_star, b_eff, largest=False).indices
            neighbours = bank[neighbour_idx]                          # (b, D)

            # distances from the TEST patch to that neighbourhood
            d = torch.cdist(m_test_star[None], neighbours)[0]         # (b,)

            # softmax written stably: exp(s*)/sum exp(d) == 1/sum exp(d - s*)
            w = 1.0 - 1.0 / torch.exp(d - s_star).sum().clamp_min(1e-12)
            out.append(w.clamp(0.0, 1.0) * s_star)
        return torch.stack(out)

    def extra_repr(self) -> str:
        state = f"{self.bank.shape[0]} vectors" if self.fitted else "unfitted"
        return (
            f"feature_dim={self.feature_dim}, k={self.k}, "
            f"coreset_ratio={self.coreset_ratio}, {state}"
        )
