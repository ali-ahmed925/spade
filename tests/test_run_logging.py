"""Every run must leave a file behind.

Terminal scrollback is not a record: a lost tmux session took a whole sweep's
output with it, and the previous single appended logs/train.log interleaved
every category and every run into one unreadable file.

Torch-free so it runs anywhere.
"""

import re

from utils.logging import get_logger, run_log_path


def test_path_is_namespaced_by_kind_and_category():
    p = run_log_path("train", "wood")
    assert re.fullmatch(r"logs/train/wood_\d{8}-\d{6}\.log", p), p


def test_path_survives_a_missing_category():
    assert re.fullmatch(r"logs/eval/\d{8}-\d{6}\.log", run_log_path("eval")), "no-category form"


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SPADE_LOG_DIR", str(tmp_path))
    assert run_log_path("train", "screw").startswith(str(tmp_path))


def test_runs_do_not_collide(monkeypatch, tmp_path):
    """Two categories must never write to the same file."""
    monkeypatch.setenv("SPADE_LOG_DIR", str(tmp_path))
    assert run_log_path("train", "wood") != run_log_path("train", "screw")


def test_logger_actually_writes_the_file(tmp_path):
    path = tmp_path / "sub" / "run.log"
    logger = get_logger("test_run_logging_writes", log_file=str(path))
    logger.info("epoch 1 complete")
    assert path.exists(), "parent directory should be created"
    assert "epoch 1 complete" in path.read_text()
