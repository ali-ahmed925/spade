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



