"""Fit the normal model from the COMPLETE training distribution.

WHY THIS EXISTS
---------------
The statistics that shipped in every checkpoint were fitted from
`deque(maxlen=20000)`. At 1024 patches per image that holds roughly the last
20 images of the last epoch, out of 320 for screw -- 6% of the data, chosen by
batch order rather than by sampling. Nothing refit them afterwards: training
ended whenever it ended, and `eval.py` explicitly does not update statistics.

So the normal model was a smoothed average of covariances over overlapping 20k
windows, of whichever images happened to come last. This pass replaces it with
a single closed-form fit over every normal patch in the training set, plus the
coreset memory bank, both computed once and written into the checkpoint.

No labels, no synthetic anomalies, no test data. `train/good` only.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_normal_features(
    model,
    loader: DataLoader,
    device: torch.device,
    max_patches: int = 500_000,
    seed: int = 0,
    logger=None,
) -> dict[str, torch.Tensor]:
    """One pass over train/good, accumulating patch features on CPU.

    MEMORY SAFETY (P5). The cap is enforced PER STREAM as an absolute number of
    rows, not as a byte budget, and the raw local pathway is 2816-d -- 327,680
    of those is 3.7 GB, well past what the 512-d default was sized for. So the
    effective cap is derived from `max_patches` scaled by the widest stream's
    dimension, and the caller no longer has to know to lower it by hand.

    SAMPLING (P2). Patches are kept by an independent Bernoulli draw from a
    seeded generator, never by truncation, and the draw is shared across streams
    so the descriptor, local and frequency rows correspond to the SAME patches.
    Taking the first N would bias the fit toward whichever images sort first.

    Returns:
        {"descriptors": (M, D_desc), "local": (M, D_local), "frequency": (M, D_f)}
        on CPU, float32. Streams that are disabled are absent.
    """
    was_training = model.training
    model.eval()

    n_images = len(loader.dataset)
    total_expected = max(1, n_images * model.num_patches)

    # Widest stream decides the budget: 500k x 512 floats is ~1 GB, so hold that
    # constant in BYTES rather than in rows.
    widest = model.descriptor_dim
    if getattr(model, "local_enabled", False):
        widest = max(widest, model.local_dim)
    effective_cap = collection_cap(max_patches, widest)

    keep_probability = min(1.0, effective_cap / total_expected)
    generator = torch.Generator().manual_seed(seed)

    chunks: dict[str, list[torch.Tensor]] = {}

    def stash(name, tensor):
        chunks.setdefault(name, []).append(tensor.float().cpu())

    for batch in loader:
        images = batch["image"].to(device)
        built = model.build_descriptors(images, return_attention=False)

        streams = {"descriptors": built["descriptors"].reshape(-1, model.descriptor_dim)}
        local = built.get("local_features")
        if local is not None:
            streams["local"] = local.reshape(-1, model.local_dim)
        if getattr(model, "use_frequency", False) and model.freq_extractor is not None:
            freq = model.compute_frequency_features(images)
            streams["frequency"] = freq.reshape(-1, freq.shape[-1])

        if keep_probability < 1.0:
            rows = next(iter(streams.values())).shape[0]
            mask = (torch.rand(rows, generator=generator) < keep_probability).to(device)
            streams = {k: v[mask] for k, v in streams.items()}

        for name, tensor in streams.items():
            stash(name, tensor)

    if was_training:
        model.train()

    out = {name: torch.cat(parts, dim=0) for name, parts in chunks.items()}

    if logger is not None:
        kept = out["descriptors"].shape[0]
        gib = sum(t.numel() * 4 for t in out.values()) / 2**30
        logger.info(
            f"normal fit: {n_images} images, {total_expected} patches available, "
            f"{kept} kept (p={keep_probability:.3f}, cap={effective_cap}, "
            f"widest stream {widest}-d, {gib:.2f} GiB held)"
        )
    return out


def collection_cap(max_patches: int, widest_dim: int, reference_dim: int = 512) -> int:
    """Rows to keep, so the byte budget is constant across feature widths.

    `max_patches` was sized for a 512-d stream (500k x 512 float32 ~ 1 GB). The
    raw local pathway is 2816-d, so the same row count is 5.5x the memory --
    3.7 GB for screw, which is how the raw path was left needing a manual
    `max_patches` reduction the caller had to know about. Holding BYTES constant
    instead makes both widths safe with one setting.
    """
    if widest_dim <= 0:
        return max_patches
    return max(1, min(max_patches, int(max_patches * reference_dim / widest_dim)))


def _subsample(features: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """A deterministic RANDOM subset of rows, never a contiguous prefix.

    `features[:n]` looks like a sample but is not one: when the dataset is under
    the collection cap no Bernoulli draw happens, so the rows are in dataset
    order and a prefix is simply the first n/1024 images. That is exactly the
    non-random slicing the full-data fit exists to eliminate, and it had crept
    back into the calibration and diagnostic steps.
    """
    total = features.shape[0]
    if total <= n:
        return features
    generator = torch.Generator().manual_seed(seed)
    index = torch.randperm(total, generator=generator)[:n]
    return features[index]


@torch.no_grad()
def feature_geometry(features: torch.Tensor) -> dict[str, float]:
    """Measure whether the trained features still have usable structure.

    kNN needs the normal set to stay STRUCTURED -- head, shaft and background
    patches occupying distinct regions. Two ways that can be lost, and this
    separates them:

      * `norm`: mean L2 length. The detection loss cannot be reduced by
        reshaping the features (mean Mahalanobis distance over the fitting set
        equals the dimension, exactly, for any distribution) -- but it CAN be
        reduced by shrinking them faster than the statistics refit tracks. A
        2x shrink between refits drops the loss 4x. If this falls across
        epochs, that treadmill is running.

      * `effective_rank`: participation ratio of the covariance spectrum,
        (sum L)^2 / sum L^2. Equals D for an isotropic distribution and falls
        toward 1 as variance concentrates into fewer directions. This is the
        one that matters: a defect living along a removed direction becomes
        invisible to a nearest-neighbour search.

    Both are diagnostics. Neither is optimised for.
    """
    x = features.float()
    centered = x - x.mean(dim=0, keepdim=True)
    n, d = centered.shape
    eigenvalues = torch.linalg.svdvals(centered).pow(2) / max(n - 1, 1)
    total = eigenvalues.sum().clamp_min(1e-12)
    return {
        "norm": float(x.norm(dim=1).mean()),
        "effective_rank": float(total.pow(2) / eigenvalues.pow(2).sum().clamp_min(1e-12)),
        "dim": float(d),
    }


@torch.no_grad()
def fit_normal_model(
    model,
    loader: DataLoader,
    device: torch.device,
    max_patches: int = 500_000,
    seed: int = 0,
    logger=None,
) -> dict[str, float]:
    """Fit the memory bank, the Mahalanobis statistics and the local scale.

    Everything here is closed-form. Call it once, after the last training epoch
    and before saving, or standalone against an existing checkpoint.
    """
    features = collect_normal_features(
        model, loader, device, max_patches=max_patches, seed=seed, logger=logger
    )
    report: dict[str, float] = {}

    # ── contextual stream: one closed-form fit over everything ──
    stats = model.mahalanobis_scorer.fit_from_normal_patches(
        features["descriptors"].to(device)
    )
    report["mahalanobis_samples"] = stats["samples"]
    report["mahalanobis_condition"] = stats["condition_number"]
    if logger is not None:
        logger.info(
            f"Mahalanobis refit from {int(stats['samples'])} patches "
            f"(condition number {stats['condition_number']:.3g})"
        )

    # ── local stream: coreset bank, then its own scale ──
    if "local" in features and model.local_enabled:
        local = features["local"].to(device)
        bank_info = model.memory_bank.fit(local)
        report["bank_candidates"] = float(bank_info["candidates"])
        report["bank_size"] = float(bank_info["selected"])

        # The scale is the mean nearest-neighbour distance of the training
        # patches themselves. Banked patches score 0 and are ~1% of the set, so
        # the mean is set by the patches that are NOT in the bank -- which is
        # the population the scale has to calibrate.
        distances = model.memory_bank(local.unsqueeze(0)).squeeze(0)
        model.local_scale.fill_(float(distances.mean().clamp_min(1e-8)))
        report["local_scale"] = float(model.local_scale)
        if logger is not None:
            logger.info(
                f"memory bank: {bank_info['selected']} of {bank_info['candidates']} "
                f"patches kept, mean kNN distance {report['local_scale']:.4f}"
            )

    # ── frequency stream: the same closed-form treatment (P1) ──
    # This was previously left entirely on the training-time EMA path, so a
    # shipped checkpoint carried frequency statistics fitted from the last ~20
    # images while contributing gamma=0.1 of every reported score.
    if "frequency" in features and getattr(model, "freq_mahalanobis_scorer", None) is not None:
        freq = features["frequency"].to(device)
        freq_stats = model.freq_mahalanobis_scorer.fit_from_normal_patches(freq)
        report["frequency_samples"] = freq_stats["samples"]
        report["frequency_condition"] = freq_stats["condition_number"]
        freq_scores = model.freq_mahalanobis_scorer(
            _subsample(features["frequency"], 50_000, seed + 4).to(device).unsqueeze(0)
        ).squeeze(0)
        model.freq_scale.fill_(float(freq_scores.mean().clamp_min(1e-8)))
        report["freq_scale"] = float(model.freq_scale)
        if logger is not None:
            logger.info(
                f"frequency refit from {int(freq_stats['samples'])} patches "
                f"(condition {freq_stats['condition_number']:.3g}), "
                f"scale {report['freq_scale']:.4f}"
            )

    # The contextual scale is EMA'd during training and may be stale or unset if
    # training was short. Recompute it here from the same full pass, so both
    # streams are calibrated against the same data.
    contextual = model.mahalanobis_scorer(
        _subsample(features["descriptors"], 50_000, seed + 1).to(device).unsqueeze(0)
    ).squeeze(0)
    model.mahal_scale.fill_(float(contextual.mean().clamp_min(1e-8)))
    model.stream_scales_initialized.fill_(True)
    report["mahal_scale"] = float(model.mahal_scale)

    # ── geometry diagnostics ──
    # Whether trained fusion is better or worse than raw ViT features for kNN is
    # an empirical question these numbers answer directly, rather than one to
    # argue about from the shape of the loss.
    sample = _subsample(features["descriptors"], 20_000, seed + 2)
    context_geometry = feature_geometry(sample)
    report["descriptor_norm"] = context_geometry["norm"]
    report["descriptor_effective_rank"] = context_geometry["effective_rank"]

    if "local" in features and model.local_enabled:
        local_sample = _subsample(features["local"], 20_000, seed + 3)
        local_geometry = feature_geometry(local_sample)
        report["local_norm"] = local_geometry["norm"]
        report["local_effective_rank"] = local_geometry["effective_rank"]

        # The only route by which the detection loss can rescale the fused
        # features: LayerNorm pins each descriptor to unit variance, so a
        # uniform shrink has to come through this learnable gain.
        gain = getattr(getattr(model.fusion, "out_norm", None), "weight", None)
        if gain is not None:
            report["fusion_gain"] = float(gain.detach().abs().mean())

        if logger is not None:
            logger.info(
                f"geometry: local norm {report['local_norm']:.3f}, "
                f"effective rank {report['local_effective_rank']:.1f}/{local_geometry['dim']:.0f}"
                + (f", fusion gain {report['fusion_gain']:.4f}" if gain is not None else "")
            )

    return report
