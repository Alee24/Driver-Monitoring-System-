"""Configuration loading and merging utilities.

This module defines a :class:`Config` dataclass that holds all configurable
parameters for the driver monitoring system.  Configuration values are loaded
from three sources, in ascending order of priority:

1. A YAML file (``config/default.yaml``).
2. Environment variables prefixed with ``DMS_`` (e.g. ``DMS_EAR_THRESH``).
3. Command‑line arguments (handled in :mod:`src.dms.app`).

Environment variables and command‑line arguments override YAML defaults.

Example usage::

    from dms.utils.config import load_config, merge_cli_args
    config = load_config(pathlib.Path('config/default.yaml'))
    config = merge_cli_args(config, args)  # where args is an argparse.Namespace

"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

EnvPrefix = "DMS_"

@dataclass
class Config:
    """Dataclass representing all configurable parameters.

    Each attribute has a sensible default but may be overridden via YAML,
    environment variables or command‑line arguments.
    """
    source: Any = 0
    width: int = 1280
    height: int = 720
    target_fps: int = 30

    # Thresholds
    ear_thresh: float = 0.23
    perclos_window: int = 120
    perclos_thresh: float = 0.35
    mar_yawn: float = 0.65
    gaze_secs: float = 1.0
    yaw_abs_deg: float = 25.0
    down_pitch_deg: float = 18.0

    # Phone detection
    enable_phone: bool = False
    yolo_model: str = "yolov8n.pt"
    phone_zone_ratio: float = 0.55

    # Audio
    speak_rate: int = 165
    audio_cooldown: float = 1.0

    # Display
    headless: bool = False

    def update(self, **kwargs: Any) -> None:
        """Update configuration attributes with provided keyword arguments.

        Unknown keys are ignored.  Types are coerced to the existing attribute
        types where possible.
        """
        for key, value in kwargs.items():
            if not hasattr(self, key) or value is None:
                continue
            current_type = type(getattr(self, key))
            # Coerce booleans from strings if necessary
            if current_type is bool and isinstance(value, str):
                setattr(self, key, value.lower() in {"1", "true", "yes", "on"})
            else:
                try:
                    setattr(self, key, current_type(value))
                except Exception:
                    # Fall back to string
                    setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the configuration."""
        return asdict(self)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except Exception:
            data = {}
    return data or {}


def _load_env(prefix: str = EnvPrefix) -> Dict[str, Any]:
    """Load configuration overrides from environment variables.

    Environment variable names are expected to be upper‑case and prefixed
    with ``prefix``.  For example, ``DMS_EAR_THRESH=0.25``.
    """
    result: Dict[str, Any] = {}
    for name, value in os.environ.items():
        if not name.startswith(prefix):
            continue
        key = name[len(prefix):].lower()
        result[key] = value
    return result


def load_config(config_path: Path) -> Config:
    """Load configuration from a YAML file and environment variables.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.  If the file does not exist,
        the defaults defined in :class:`Config` are used.

    Returns
    -------
    Config
        A configuration instance with values from YAML and environment variables applied.
    """
    # Start with defaults
    cfg = Config()
    # Load YAML
    yaml_data = _load_yaml(config_path)
    cfg.update(**yaml_data)
    # Load environment variables
    env_data = _load_env()
    cfg.update(**env_data)
    return cfg


def merge_cli_args(cfg: Config, args: Any) -> Config:
    """Merge command‑line arguments into a configuration instance.

    Only attributes present on :class:`Config` will be updated.  ``argparse``
    automatically fills unspecified arguments with ``None``, which are ignored.

    Parameters
    ----------
    cfg:
        The existing configuration instance.
    args:
        An ``argparse.Namespace`` or similar object with attributes matching
        configuration keys.

    Returns
    -------
    Config
        The updated configuration instance.
    """
    if args is None:
        return cfg
    kwargs = {k: getattr(args, k) for k in vars(args)}
    cfg.update(**kwargs)
    return cfg