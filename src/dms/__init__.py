"""Top‑level package for Driver Monitoring System (Path A).

The package is organised into multiple modules:

- :mod:`app` — Command‑line interface entry point.
- :mod:`video_loop` — The main loop that ties together input, processing and output.
- :mod:`face.metrics` — Functions to compute eye and mouth aspect ratios and PERCLOS.
- :mod:`headpose.pose` — Utility for estimating head pose from facial landmarks.
- :mod:`detectors.phone` — Optional YOLOv8n phone detector.
- :mod:`audio.alerts` — Audio alert helpers (text‑to‑speech and beep fallback).
- :mod:`hud.overlay` — Renders the heads‑up display on each frame.
- :mod:`utils.config` — Configuration loading and merging from YAML, environment and CLI.
- :mod:`utils.logging` — Logging helpers based on Loguru.
- :mod:`utils.timer` — Simple timing utilities (rate limiting and FPS measurement).

"""

from importlib.metadata import version as _get_version

try:
    __version__ = _get_version("dms_path_a")
except Exception:
    # Fallback if package metadata is not available
    __version__ = "0.0.0"

__all__ = ["__version__"]