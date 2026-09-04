"""The run record must survive everything that has destroyed one before.

A lost tmux session took a whole sweep's output; a wandb UsageError killed a
training run outright; and the wandb quota can run out mid-run. So the contract
is: disk is written first and unconditionally, and no telemetry failure can
propagate into the training loop.
"""

import json

import numpy as np
import pytest

from utils.run_logger import RunLogger, replay_to_wandb


class _ExplodingWandb:
    """Every call fails, the way an exhausted quota behaves."""

    class run:  # noqa: N801 - mirrors the wandb module surface
        summary: dict = {}

    config = property(lambda self: (_ for _ in ()).throw(RuntimeError("quota exceeded")))

    def log(self, *args, **kwargs):
        raise RuntimeError("quota exceeded")


def _records(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle]


def test_disk_record_is_written_without_wandb(tmp_path):
    log = RunLogger(tmp_path, wandb_run=None, name="r")
    log.log({"train/loss": 1.5}, step=3)
    log.finish()

    events = _records(log.path)
    assert events[0]["event"] == "start"
    assert "git_commit" in events[0]
    assert any(e.get("train/loss") == 1.5 and e["step"] == 3 for e in events)
    assert events[-1]["event"] == "finish"


def test_a_failing_wandb_never_reaches_the_caller(tmp_path):
    """The exact failure mode: quota exhausted at epoch 12."""
    log = RunLogger(tmp_path, wandb_run=_ExplodingWandb(), name="r")
    for step in range(5):
        log.log({"epoch/loss": float(step)}, step=step)     # must not raise
    log.log_histogram("dist/x", np.random.randn(100), step=5)
    log.summary("best", 0.9)
    log.finish()

    logged = [e for e in _records(log.path) if "epoch/loss" in e]
    assert len(logged) == 5, "every metric must still be on disk"


def test_histogram_summary_lands_on_disk(tmp_path):
    log = RunLogger(tmp_path, name="r")
    log.log_histogram("dist/scores", np.arange(101, dtype=float), step=1)
    entry = next(e for e in _records(log.path) if e["event"] == "histogram")
    assert entry["n"] == 101
    assert entry["p50"] == pytest.approx(50.0)
    assert entry["min"] == 0.0 and entry["max"] == 100.0


def test_steps_never_go_backwards(tmp_path):
    """wandb rejects a decreasing step; the record must be monotonic."""
    log = RunLogger(tmp_path, name="r")
    log.log({"a": 1}, step=10)
    log.log({"a": 2}, step=4)
    steps = [e["step"] for e in _records(log.path) if "a" in e]
    assert steps == [10, 10]


def test_non_serialisable_values_do_not_break_the_record(tmp_path):
    import torch

    log = RunLogger(tmp_path, name="r")
    log.log({"tensor": torch.tensor(2.5), "array": np.float32(1.5), "obj": object()})
    entry = next(e for e in _records(log.path) if "tensor" in e)
    assert entry["tensor"] == pytest.approx(2.5)
    assert entry["array"] == pytest.approx(1.5)
    assert isinstance(entry["obj"], str)


def test_config_is_recorded(tmp_path):
    log = RunLogger(tmp_path, name="r")
    log.log_config({"category": "screw", "vit": {"image_size": 448}})
    entry = next(e for e in _records(log.path) if e["event"] == "config")
    assert entry["category"] == "screw"
    assert entry["vit"]["image_size"] == 448


def test_replay_reads_a_finished_record(tmp_path):
    """The recovery path: a run logged offline can be pushed to a fresh wandb
    account afterwards, without retraining."""
    log = RunLogger(tmp_path, name="r")
    log.log_config({"category": "screw"})
    log.log({"val/image_auroc": 0.87}, step=1, event="epoch")
    log.summary("best_epoch", 3)
    log.finish()

    seen = {"config": {}, "logged": [], "summary": {}}

    class _Recorder:
        def __init__(self): self.config = self; self.summary = seen["summary"]
        def update(self, d, **kw): seen["config"].update(d)
        def log(self, payload, step=None): seen["logged"].append((payload, step))
        def finish(self): pass
        def __setitem__(self, k, v): seen["summary"][k] = v

    import utils.run_logger as module

    class _FakeWandb:
        @staticmethod
        def init(**kwargs): return _Recorder()

    original = module.__dict__.get("wandb")
    import sys
    sys.modules["wandb"] = _FakeWandb
    try:
        replay_to_wandb(log.path, project="x")
    finally:
        if original is not None:
            sys.modules["wandb"] = original

    assert seen["config"].get("category") == "screw"
    assert any(p.get("val/image_auroc") == 0.87 for p, _ in seen["logged"])
