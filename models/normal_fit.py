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

    Args:
        max_patches: cap on patches kept per stream. 500k x 512-d float32 is
            about 1 GB, which is the practical ceiling for holding the set in
            RAM before fitting. When the dataset exceeds it, patches are kept by
            an independent Bernoulli draw rather than by truncation -- taking
            the first N would bias the fit toward whichever images sort first.

    Returns:
        {"local": (N, D_local), "descriptors": (N, D_desc)} on CPU, float32.
        "local" is absent when the local pathway is disabled.
    """
    was_training = model.training
    model.eval()

    n_images = len(loader.dataset)
    total_expected = max(1, n_images * model.num_patches)
    keep_probability = min(1.0, max_patches / total_expected)
    generator = torch.Generator().manual_seed(seed)

    local_chunks: list[torch.Tensor] = []
    descriptor_chunks: list[torch.Tensor] = []

    for batch in loader:
        images = batch["image"].to(device)
        built = model.build_descriptors(images, return_attention=False)

        descriptors = built["descriptors"].reshape(-1, model.descriptor_dim)
        local = built.get("local_features")
        if local is not None:
            local = local.reshape(-1, model.local_dim)

        if keep_probability < 1.0:
            mask = torch.rand(descriptors.shape[0], generator=generator) < keep_probability
            mask = mask.to(descriptors.device)
            descriptors = descriptors[mask]
            if local is not None:
                local = local[mask]

        descriptor_chunks.append(descriptors.float().cpu())
        if local is not None:
            local_chunks.append(local.float().cpu())

    if was_training:
        model.train()

    out = {"descriptors": torch.cat(descriptor_chunks, dim=0)}
    if local_chunks:
        out["local"] = torch.cat(local_chunks, dim=0)

    if logger is not None:
        kept = out["descriptors"].shape[0]
        logger.info(
            f"normal fit: {n_images} images, {total_expected} patches available, "
            f"{kept} kept (p={keep_probability:.3f})"
        )
    return out


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

    # The contextual scale is EMA'd during training and may be stale or unset if
    # training was short. Recompute it here from the same full pass, so both
    # streams are calibrated against the same data.
    contextual = model.mahalanobis_scorer(
        features["descriptors"][: min(50_000, features["descriptors"].shape[0])]
        .to(device).unsqueeze(0)
    ).squeeze(0)
    model.mahal_scale.fill_(float(contextual.mean().clamp_min(1e-8)))
    model.stream_scales_initialized.fill_(True)
    report["mahal_scale"] = float(model.mahal_scale)

    return report
