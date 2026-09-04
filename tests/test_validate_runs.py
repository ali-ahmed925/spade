"""validate() must be EXECUTED by a test, not merely imported.

A NameError inside a function body does not surface at import time. `import
train` succeeded while validate() referenced an undefined `compute_image_auroc`,
and the run died three minutes into epoch 1 -- after the full training epoch and
the normal-model fit had already been paid for.

So these call it for real, on the stub backbone, with the same shapes the
training loop passes.
"""

import numpy as np
import pytest
import torch

import train as train_module
from models.normal_fit import fit_normal_model
from tests.stub_blip2 import make_stub_spade

IMAGE_SIZE = 224


class _ValLoader:
    """Mimics the validation loader's batch dict exactly."""

    def __init__(self, model, n=6, seed=0, all_normal=False):
        torch.manual_seed(seed)
        n_patches = model.num_patches
        self._batches = []
        for i in range(0, n, 2):
            labels = torch.zeros(2, dtype=torch.long) if all_normal else torch.tensor(
                [i % 2, (i + 1) % 2], dtype=torch.long
            )
            patch_labels = torch.zeros(2, n_patches)
            patch_labels[labels == 1, :20] = 1.0
            self._batches.append({
                "image": torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE),
                "label": labels,
                "patch_labels": patch_labels,
            })
        self.dataset = range(n)

    def __iter__(self):
        return iter(self._batches)


class _FitLoader:
    def __init__(self, n=16, seed=1):
        torch.manual_seed(seed)
        images = torch.randn(n, 3, IMAGE_SIZE, IMAGE_SIZE)
        self._batches = [{"image": images[i : i + 2]} for i in range(0, n, 2)]
        self.dataset = range(n)

    def __iter__(self):
        return iter(self._batches)


@pytest.fixture
def fitted_model():
    model = make_stub_spade(image_size=IMAGE_SIZE)
    fit_normal_model(model, _FitLoader(), torch.device("cpu"))
    return model


def test_validate_actually_runs(fitted_model):
    metrics = train_module.validate(fitted_model, _ValLoader(fitted_model), torch.device("cpu"))
    assert "val/image_auroc" in metrics
    assert 0.0 <= metrics["val/image_auroc"] <= 1.0


def test_validate_reports_every_stream(fitted_model):
    """The per-stream columns are the whole point of the instrumentation."""
    metrics = train_module.validate(fitted_model, _ValLoader(fitted_model), torch.device("cpu"))
    streams = [k for k in metrics if k.startswith("val/stream_")]
    assert "val/stream_local_knn" in streams
    assert "val/stream_contextual_mahalanobis" in streams


def test_validate_reports_separability(fitted_model):
    metrics = train_module.validate(fitted_model, _ValLoader(fitted_model), torch.device("cpu"))
    for key in ("val/score_normal_mean", "val/score_anomalous_mean",
                "val/defect_elevation", "val/drowned_fraction"):
        assert key in metrics, f"{key} missing"


def test_both_aggregations_are_accepted(fitted_model):
    for aggregation in ("max", "topk_mean"):
        metrics = train_module.validate(
            fitted_model, _ValLoader(fitted_model), torch.device("cpu"), aggregation=aggregation
        )
        assert np.isfinite(metrics["val/image_auroc"])


def test_an_all_normal_split_does_not_abort_the_run(fitted_model):
    """roc_auc_score raises with a single class. That must be nan, not a crash
    twenty minutes into training."""
    metrics = train_module.validate(
        fitted_model, _ValLoader(fitted_model, all_normal=True), torch.device("cpu")
    )
    assert np.isnan(metrics["val/image_auroc"])


def test_safe_auroc_edges():
    assert np.isnan(train_module._safe_auroc(np.array([]), np.array([])))
    assert np.isnan(train_module._safe_auroc(np.zeros(5), np.random.rand(5)))
    assert train_module._safe_auroc(np.array([0, 1]), np.array([0.1, 0.9])) == 1.0
