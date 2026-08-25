"""Evaluation metrics for anomaly detection."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve


def compute_image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Image-level AUROC.

    Args:
        labels: (N,) binary ground-truth labels (0=normal, 1=anomaly).
        scores: (N,) predicted anomaly scores.

    Returns:
        AUROC value.
    """
    return float(roc_auc_score(labels, scores))


def compute_pixel_auroc(masks: np.ndarray, score_maps: np.ndarray) -> float:
    """Pixel/patch-level AUROC.

    Args:
        masks: (N, H, W) binary ground-truth masks.
        score_maps: (N, H, W) predicted anomaly score maps.

    Returns:
        Pixel-level AUROC.
    """
    return float(roc_auc_score(masks.ravel(), score_maps.ravel()))


def compute_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision score.

    Args:
        labels: binary ground-truth labels (any shape, will be flattened).
        scores: predicted anomaly scores (same shape as labels).

    Returns:
        Average precision value.
    """
    return float(average_precision_score(labels.ravel(), scores.ravel()))


def compute_best_f1(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Best F1 score and its threshold.

    Args:
        labels: binary ground-truth labels (flattened internally).
        scores: predicted anomaly scores (flattened internally).

    Returns:
        (best_f1, threshold)
    """
    precision, recall, thresholds = precision_recall_curve(labels.ravel(), scores.ravel())
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return float(f1_scores[best_idx]), float(thresholds[best_idx])


def compute_pro(
    masks: np.ndarray,
    score_maps: np.ndarray,
    max_fpr: float = 0.3,
    num_thresholds: int = 100,
) -> float:
    """Per-Region Overlap (PRO), integrated up to `max_fpr` and normalized.

    Pixel AUROC is dominated by large defects: one big blob localized well can
    hide complete failure on small ones. PRO weights every connected ground-truth
    component equally regardless of area, which is why the MVTec literature
    reports it alongside pixel AUROC for localization claims.

    For each threshold: for every connected component in the ground truth,
    compute the fraction of that component covered by the prediction, then
    average over components (not pixels). Plot that against the false-positive
    rate over normal pixels, integrate up to `max_fpr`, and divide by `max_fpr`
    so a perfect localizer scores 1.0.

    Args:
        masks: (N, H, W) binary ground-truth masks.
        score_maps: (N, H, W) predicted anomaly score maps (raw scores).
        max_fpr: integration limit on the false-positive-rate axis.
        num_thresholds: number of thresholds sampled across the score range.

    Returns:
        Normalized PRO-AUC in [0, 1], or nan if the masks contain no defect.
    """
    from scipy.ndimage import label

    masks = np.asarray(masks).astype(bool)
    scores = np.asarray(score_maps, dtype=np.float64)
    if masks.shape != scores.shape:
        raise ValueError(f"shape mismatch: masks {masks.shape} vs scores {scores.shape}")
    if not masks.any():
        return float("nan")

    flat_masks = masks.reshape(len(masks), -1)
    flat_scores = scores.reshape(len(scores), -1)

    # Each ground-truth component as (image_index, flat pixel indices).
    components: list[tuple[int, np.ndarray]] = []
    for i, m in enumerate(masks):
        labelled, n = label(m)
        flat_labelled = labelled.reshape(-1)
        for cid in range(1, n + 1):
            components.append((i, np.flatnonzero(flat_labelled == cid)))
    if not components:
        return float("nan")

    normal = ~flat_masks
    n_normal = int(normal.sum())
    if n_normal == 0:
        return float("nan")

    lo, hi = float(scores.min()), float(scores.max())
    if hi <= lo:
        return float("nan")

    # Walk thresholds from strict to permissive so FPR increases monotonically
    # and we can stop as soon as we pass the integration limit.
    fprs: list[float] = []
    pros: list[float] = []
    for t in np.linspace(hi, lo, num_thresholds):
        pred = flat_scores >= t
        fpr = float((pred & normal).sum()) / n_normal
        pro = float(np.mean([pred[i][idx].mean() for i, idx in components]))
        fprs.append(fpr)
        pros.append(pro)
        if fpr > max_fpr:
            break  # keep this point: it is the right endpoint for interpolation

    # Anchor at the origin: an empty prediction has zero FPR and zero overlap.
    fprs_arr = np.asarray([0.0] + fprs)
    pros_arr = np.asarray([0.0] + pros)
    order = np.argsort(fprs_arr)
    fprs_arr, pros_arr = fprs_arr[order], pros_arr[order]

    # Integrate over the FULL [0, max_fpr] range by interpolating the curve onto
    # a uniform grid. Simply integrating the sampled points is wrong: a good
    # localizer reaches high PRO at near-zero FPR, so its curve barely spans the
    # x-axis and the raw area under the sampled points is tiny. np.interp holds
    # the last value beyond the sampled range, which is the correct behaviour
    # since PRO is non-decreasing in FPR.
    grid = np.linspace(0.0, max_fpr, 256)
    pro_on_grid = np.interp(grid, fprs_arr, pros_arr)
    # numpy 2 renamed trapz -> trapezoid and REMOVED trapz, so getattr with a
    # np.trapz default would raise while evaluating the default argument.
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapz(pro_on_grid, grid) / max_fpr)
