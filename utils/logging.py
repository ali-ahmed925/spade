"""Lightweight logging helpers."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def run_log_path(kind: str, category: str | None = None) -> str:
    """A unique log file per run: logs/<kind>/<category>_<timestamp>.log.

    Every entry point writes one. Terminal scrollback is not a record -- a lost
    tmux session took a full sweep's output with it once, and a single appended
    logs/train.log interleaves every category and every run into one file that
    cannot be read back per-experiment.

    SPADE_LOG_DIR overrides the root for a run that should log elsewhere.
    """
    root = Path(os.environ.get("SPADE_LOG_DIR", "logs")) / kind
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{category}_{stamp}.log" if category else f"{stamp}.log"
    return str(root / name)


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Create a logger that writes to stdout and optionally to a file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # avoid duplicate handlers on repeated calls

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Optional file handler
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger



