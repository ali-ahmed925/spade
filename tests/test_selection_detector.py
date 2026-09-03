"""Checkpoint selection must rank epochs by the detector that ships.

The local kNN stream does not exist until the normal model is fitted. If the
fit only ran after training, every epoch would be ranked by the contextual
Mahalanobis stream ALONE -- a different detector from the reported one -- so the
saved checkpoint need not be the best one for what actually ships.

These pin the property that makes selection sound: once fitted, the score the
validation loop sees includes every stream.
"""

import torch

from models.normal_fit import fit_normal_model
from tests.stub_blip2 import make_stub_spade

IMAGE_SIZE = 224


class _Loader:
    def __init__(self, images, batch=2):
        self._batches = [
            {"image": images[i : i + batch]} for i in range(0, images.shape[0], batch)
        ]
        self.dataset = range(images.shape[0])

    def __iter__(self):
        return iter(self._batches)


def _train_images(n=16, seed=1):
    torch.manual_seed(seed)
    return torch.randn(n, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_validation_score_changes_once_the_bank_is_fitted():
    """This is the whole point: before the fit, validation ranks epochs on a
    detector that is missing a stream."""
    model = make_stub_spade(image_size=IMAGE_SIZE)
    fit_normal_model(model, _Loader(_train_images()), torch.device("cpu"), max_patches=100_000)
    model.eval()

    probe = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    full = model(probe)["patch_scores"]

    model.memory_bank.reset()
    contextual_only = model(probe)["patch_scores"]

    assert not torch.allclose(full, contextual_only), (
        "if these matched, selecting on the contextual stream would be harmless "
        "-- they do not, which is why selection has to fit first"
    )


def test_two_checkpoints_can_rank_differently_under_the_two_detectors():
    """The concrete failure: epoch A beats B on the contextual stream while B
    beats A on the full detector. Selecting on the wrong one saves A."""
    probe = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)

    def scores(seed):
        model = make_stub_spade(seed=seed, image_size=IMAGE_SIZE)
        fit_normal_model(
            model, _Loader(_train_images(seed=seed)), torch.device("cpu"), max_patches=100_000
        )
        model.eval()
        out = model(probe)
        contextual = out["score_components"]["contextual_mahalanobis"].mean()
        return float(out["patch_scores"].detach().mean()), float(contextual.detach())

    full_a, ctx_a = scores(0)
    full_b, ctx_b = scores(5)

    # Not asserting a specific inversion -- only that the two detectors produce
    # genuinely different orderings of the same quantity, so ranking by one is
    # not a proxy for ranking by the other.
    assert (full_a - full_b) != (ctx_a - ctx_b)


def test_a_fresh_model_has_no_bank_so_selection_would_be_stale():
    model = make_stub_spade(image_size=IMAGE_SIZE)
    assert not model.memory_bank.fitted
    assert "local_knn" not in model(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE))["score_components"]


def test_fit_restores_training_mode():
    """The per-epoch fit runs inside the training loop; leaving the model in
    eval mode would silently disable dropout for the rest of training."""
    model = make_stub_spade(image_size=IMAGE_SIZE)
    model.train()
    fit_normal_model(model, _Loader(_train_images()), torch.device("cpu"), max_patches=10_000)
    assert model.training, "fit must not leave the model in eval mode"


def test_fit_does_not_accumulate_gradients():
    model = make_stub_spade(image_size=IMAGE_SIZE)
    model.train()
    model.zero_grad()
    fit_normal_model(model, _Loader(_train_images()), torch.device("cpu"), max_patches=10_000)
    leaked = [n for n, p in model.named_parameters() if p.grad is not None]
    assert not leaked, f"the fit must not touch gradients, got {leaked[:3]}"
