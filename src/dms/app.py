"""Command‑line interface for the driver monitoring system.

This module defines a CLI using :mod:`argparse`.  It loads configuration
defaults from ``config/default.yaml``, merges environment variables and
command‑line overrides, configures logging and then invokes the main video
loop.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dms.utils.config import load_config, merge_cli_args
from dms.utils.logging import configure_logging
from dms.video_loop import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Driver monitoring system using MediaPipe Face Mesh (Path A)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Video source and resolution
    parser.add_argument("--source", type=str, default=None, help="Camera index or video file path")
    parser.add_argument("--width", type=int, default=None, help="Frame width")
    parser.add_argument("--height", type=int, default=None, help="Frame height")
    parser.add_argument("--target-fps", type=int, default=None, help="Target frames per second")

    # Thresholds
    parser.add_argument("--ear-thresh", type=float, default=None, help="Eye Aspect Ratio threshold for closed eyes")
    parser.add_argument("--perclos-window", type=int, default=None, help="Number of frames in the PERCLOS window")
    parser.add_argument("--perclos-thresh", type=float, default=None, help="PERCLOS threshold for drowsiness detection")
    parser.add_argument("--mar-yawn", type=float, default=None, help="Mouth Aspect Ratio threshold for yawning")
    parser.add_argument("--gaze-secs", type=float, default=None, help="Seconds allowed to look away before distraction alert")
    parser.add_argument("--yaw-abs-deg", type=float, default=None, help="Absolute yaw angle threshold (degrees)")
    parser.add_argument("--down-pitch-deg", type=float, default=None, help="Downward pitch angle threshold (degrees)")

    # Phone detection
    parser.add_argument("--enable-phone", type=str, default=None, help="Enable phone detection (true/false)")
    parser.add_argument("--yolo-model", type=str, default=None, help="Path to YOLOv8n weights file")
    parser.add_argument("--phone-zone-ratio", type=float, default=None, help="Fraction of frame height used for phone detection")

    # Audio
    parser.add_argument("--speak-rate", type=int, default=None, help="Words per minute for text‑to‑speech")
    parser.add_argument("--audio-cooldown", type=float, default=None, help="Cooldown between audio alerts (seconds)")

    # Display
    parser.add_argument("--headless", type=str, default=None, help="Run without showing video window (true/false)")

    # Logging
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Build path to config/default.yaml relative to this file
    config_path = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
    config = load_config(config_path)
    # Merge CLI arguments (overrides environment and YAML)
    config = merge_cli_args(config, args)
    # Configure logging
    configure_logging(level=args.log_level)
    # Run the video loop
    run(config)


if __name__ == "__main__":
    main()