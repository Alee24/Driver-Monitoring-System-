"""Timing utilities.

This module provides small helper classes to rate‑limit events (e.g. speech
alerts) and to measure frames per second.  These helpers are deliberately
simple to avoid dependencies on external libraries.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

@dataclass
class RateLimiter:
    """Simple rate limiter that allows an action only after a cooldown period.

    Parameters
    ----------
    cooldown:
        Minimum number of seconds that must elapse between allowed events.
    """
    cooldown: float
    _last: float = 0.0

    def allow(self) -> bool:
        """Return ``True`` if the event is allowed and update the timestamp."""
        now = time.monotonic()
        if now - self._last >= self.cooldown:
            self._last = now
            return True
        return False


@dataclass
class FPSMeter:
    """Running frames‑per‑second measurement.

    The meter computes the number of frames processed per second using an
    exponential moving average.  Call :meth:`update` once per frame and read
    the ``fps`` attribute to retrieve the smoothed FPS.
    """
    smoothing: float = 0.9
    fps: float = 0.0
    _last: float = 0.0

    def update(self) -> float:
        """Update the meter and return the current FPS estimate."""
        now = time.monotonic()
        if self._last == 0:
            self._last = now
            return self.fps
        dt = now - self._last
        if dt <= 0:
            return self.fps
        current = 1.0 / dt
        self.fps = (self.smoothing * self.fps) + ((1.0 - self.smoothing) * current)
        self._last = now
        return self.fps