import os
from types import SimpleNamespace

import pytest

from dms.utils.config import load_config, merge_cli_args


def test_environment_override(monkeypatch):
    """Environment variables should override YAML defaults."""
    from pathlib import Path
    monkeypatch.setenv("DMS_EAR_THRESH", "0.5")
    config_path = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
    cfg = load_config(config_path)
    assert cfg.ear_thresh == 0.5


def test_cli_override(monkeypatch):
    """Command‑line arguments should override environment variables and YAML."""
    from pathlib import Path
    monkeypatch.setenv("DMS_EAR_THRESH", "0.5")
    config_path = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
    cfg = load_config(config_path)
    # CLI sets ear_thresh back to YAML default 0.23 via args
    args = SimpleNamespace(ear_thresh=0.23)
    cfg = merge_cli_args(cfg, args)
    assert cfg.ear_thresh == 0.23