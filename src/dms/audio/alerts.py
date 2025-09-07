"""
Audio alert helpers (continuous repeater, Windows-safe).

A background thread repeats the current active message every
`repeat_interval` seconds until the message is cleared (set to None).

Key differences:
- Calls `engine.stop()` before each say to interrupt any queued audio.
- Uses a background loop with precise scheduling (monotonic time).
- Minimal debug prints to verify it's firing (toggle by DEBUG_PRINT).
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import pyttsx3


DEBUG_PRINT = False  # set True to see [SPEAK] prints in your console


class AlertSpeaker:
    """
    Continuous alert speaker.

    Parameters
    ----------
    speak_rate: int
        Words per minute for TTS.
    repeat_interval: float
        Seconds between repeats while a message is active.
        Use 0.3–1.0 for near-continuous reminders.
    """

    def __init__(self, speak_rate: int = 165, repeat_interval: float = 1.0) -> None:
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", speak_rate)

        # state
        self._repeat_interval = max(0.0, float(repeat_interval))
        self._active_msg: Optional[str] = None
        self._stop = False

        # locks
        self._tts_lock = threading.Lock()      # serialize engine access
        self._state_lock = threading.Lock()    # protect message/interval

        # worker thread
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    # ----------------- public API -----------------

    def set_repeat_interval(self, seconds: float) -> None:
        with self._state_lock:
            self._repeat_interval = max(0.0, float(seconds))

    def set_active_message(self, message: Optional[str]) -> None:
        """
        Start/stop continuous speaking.
        - Pass a string to start repeating that message.
        - Pass None to stop speaking.
        """
        with self._state_lock:
            self._active_msg = message

    # Backward-compat: treat `speak()` as "activate message"
    def speak(self, message: str) -> None:
        self.set_active_message(message)

    def stop(self) -> None:
        self._stop = True
        try:
            self._engine.stop()
        except Exception:
            pass

    # ----------------- worker internals -----------------

    def _speak_once(self, text: str) -> None:
        try:
            with self._tts_lock:
                # IMPORTANT on Windows: stop any queued/ongoing speech so repeats are immediate
                try:
                    self._engine.stop()
                except Exception:
                    pass

                self._engine.say(text)
                if DEBUG_PRINT:
                    print(f"[SPEAK] {time.strftime('%H:%M:%S')} -> {text}")
                self._engine.runAndWait()
        except Exception:
            # Fallback beep if TTS fails
            print("\a", end="")

    def _loop(self) -> None:
        """Repeat the active message on schedule until cleared."""
        idle_sleep = 0.1
        next_t = time.monotonic()

        while not self._stop:
            # snapshot state
            with self._state_lock:
                msg = self._active_msg
                interval = self._repeat_interval

            if msg:
                now = time.monotonic()
                if now >= next_t:
                    self._speak_once(msg)
                    # schedule next repeat
                    next_t = now + (interval if interval > 0 else 0.01)
                else:
                    time.sleep(min(idle_sleep, max(0.0, next_t - now)))
            else:
                # no active message
                time.sleep(idle_sleep)
                next_t = time.monotonic()
