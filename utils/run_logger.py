"""Structured run record: JSONL on disk always, Weights & Biases when available.

WHY DISK IS PRIMARY
-------------------
Two things have already destroyed run records in this project: a lost tmux
session took a whole sweep's terminal output with it, and a wandb UsageError
killed a training run outright. So the ordering here is deliberate:

  1. every metric is appended to a local JSONL file, synchronously;
  2. the same metric is mirrored to wandb, inside a try/except that can never
     raise into the training loop.

If the wandb quota runs out at epoch 12, epochs 1-20 are still fully recorded on
disk and can be replayed into a fresh wandb project afterwards. Nothing about
the run depends on the network.

WHAT GETS RECORDED
------------------
Enough to reconstruct why a run behaved the way it did, months later:
the merged config, the git commit, per-step losses and gradient norms, per-epoch
validation broken down BY STREAM, the full normal-fit report (bank size,
covariance conditioning, stream scales, feature geometry), every checkpoint
decision with its reason, and score distributions for normal vs anomalous
images.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "unknown"


def _jsonable(value: Any) -> Any:
    """Coerce numpy/torch scalars so json.dump never dies mid-run."""
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


class RunLogger:
    """One record per run. Never raises into the caller."""

    def __init__(
        self,
        run_dir: str | Path,
        wandb_run=None,
        logger=None,
        name: str = "run",
    ):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{name}.jsonl"
        self.wandb = wandb_run
        self.logger = logger
        self.started = time.time()
        self._step = 0
        self._wandb_failures = 0

        self._append({"event": "start", "git_commit": _git_commit(), "path": str(self.path)})
        if logger is not None:
            logger.info(f"run record: {self.path}")

    # ── plumbing ─────────────────────────────────────────────────────────
    def _append(self, record: dict) -> None:
        record = {"t": round(time.time() - self.started, 3), **_jsonable(record)}
        try:
            with open(self.path, "a") as handle:
                handle.write(json.dumps(record) + "\n")
        except Exception as exc:  # noqa: BLE001 - a logging failure must not stop training
            if self.logger is not None:
                self.logger.warning(f"run record write failed: {exc}")

    def _to_wandb(self, payload: dict, step: int | None) -> None:
        if self.wandb is None:
            return
        try:
            self.wandb.log(payload, step=step)
        except Exception as exc:  # noqa: BLE001
            self._wandb_failures += 1
            if self._wandb_failures in (1, 10, 100) and self.logger is not None:
                self.logger.warning(
                    f"wandb.log failed ({type(exc).__name__}: {exc}); "
                    f"{self._wandb_failures} so far. The JSONL record at "
                    f"{self.path} is unaffected and remains complete."
                )

    # ── public API ───────────────────────────────────────────────────────
    def log(self, metrics: dict, step: int | None = None, event: str = "metrics") -> None:
        """Scalars. `step` must be non-decreasing; omitted means 'reuse the last'."""
        if step is not None:
            self._step = max(self._step, int(step))
        self._append({"event": event, "step": self._step, **metrics})
        self._to_wandb(metrics, self._step)

    def log_config(self, config: dict, event: str = "config") -> None:
        self._append({"event": event, **config})
        if self.wandb is not None:
            try:
                self.wandb.config.update(_jsonable(config), allow_val_change=True)
            except Exception:  # noqa: BLE001
                pass

    def log_histogram(self, name: str, values, step: int | None = None) -> None:
        """Distribution of a quantity. Summary statistics always land on disk;
        the full histogram only goes to wandb, because a JSONL line holding
        100k floats per epoch is not a useful record."""
        array = np.asarray(values, dtype=np.float64).ravel()
        if array.size == 0:
            return
        if step is not None:
            self._step = max(self._step, int(step))
        percentiles = np.percentile(array, [1, 25, 50, 75, 99])
        self._append({
            "event": "histogram", "step": self._step, "name": name,
            "n": int(array.size), "mean": float(array.mean()), "std": float(array.std()),
            "min": float(array.min()), "max": float(array.max()),
            "p1": float(percentiles[0]), "p25": float(percentiles[1]),
            "p50": float(percentiles[2]), "p75": float(percentiles[3]),
            "p99": float(percentiles[4]),
        })
        if self.wandb is not None:
            try:
                import wandb as _wandb

                self._to_wandb({name: _wandb.Histogram(array)}, self._step)
            except Exception:  # noqa: BLE001
                pass

    def log_images(self, name: str, images: list, captions: list[str] | None = None,
                   step: int | None = None) -> None:
        """Visual artefacts (heatmaps, overlays). wandb only -- the disk record
        notes that they were sent, not the pixels."""
        if step is not None:
            self._step = max(self._step, int(step))
        self._append({"event": "images", "step": self._step, "name": name, "n": len(images)})
        if self.wandb is None or not images:
            return
        try:
            import wandb as _wandb

            payload = [
                _wandb.Image(img, caption=(captions[i] if captions else None))
                for i, img in enumerate(images)
            ]
            self._to_wandb({name: payload}, self._step)
        except Exception:  # noqa: BLE001
            pass

    def summary(self, key: str, value: Any) -> None:
        self._append({"event": "summary", "key": key, "value": value})
        if self.wandb is not None:
            try:
                self.wandb.run.summary[key] = _jsonable(value)
            except Exception:  # noqa: BLE001
                pass

    def finish(self) -> None:
        self._append({
            "event": "finish",
            "duration_s": round(time.time() - self.started, 1),
            "wandb_failures": self._wandb_failures,
        })
        if self.wandb is not None:
            try:
                self.wandb.finish()
            except Exception:  # noqa: BLE001
                pass


def replay_to_wandb(jsonl_path: str | Path, **init_kwargs) -> None:
    """Push a completed disk record into a fresh wandb run.

    The reason the disk record is primary: if the quota runs out mid-run, or
    wandb was off entirely, the whole run can still be visualised afterwards --
    including from a different account -- without retraining.
    """
    import wandb

    run = wandb.init(**init_kwargs)
    with open(jsonl_path) as handle:
        for line in handle:
            record = json.loads(line)
            event = record.get("event")
            if event == "config":
                run.config.update(
                    {k: v for k, v in record.items() if k not in ("event", "t")},
                    allow_val_change=True,
                )
            elif event in ("metrics", "epoch", "fit", "selection"):
                payload = {
                    k: v for k, v in record.items()
                    if k not in ("event", "t", "step") and isinstance(v, (int, float))
                }
                if payload:
                    run.log(payload, step=record.get("step"))
            elif event == "summary":
                run.summary[record["key"]] = record["value"]
    run.finish()
