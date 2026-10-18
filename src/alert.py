"""
================================================================================
AI-Based Driver Fatigue and Drowsiness Prediction System
Module: src/alert.py
Purpose: Audio buzzer alarm and visual warning alert manager.
         Member 2: Machine Learning & Application
================================================================================
"""

import time
import threading
import sys
from typing import Optional

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    AUDIO_ALARM_ENABLED,
    AUDIO_FREQUENCY_HZ,
    AUDIO_DURATION_MS
)

# Attempt to import Windows sound library or cross-platform fallback
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class AlertManager:
    """
    Manages non-blocking audio buzzer alerts and visual warning messages.
    Includes debouncing so the buzzer does not overlap or stall the video loop.
    """

    def __init__(
        self,
        cooldown_seconds: float = 3.0,
        frequency_hz: int = AUDIO_FREQUENCY_HZ,
        duration_ms: int = AUDIO_DURATION_MS
    ):
        self.cooldown_seconds = cooldown_seconds
        self.frequency_hz = frequency_hz
        self.duration_ms = duration_ms

        self.last_alert_time: float = 0.0
        self.is_playing: bool = False
        self._lock = threading.Lock()

    def trigger_audio_alarm(self, force: bool = False) -> bool:
        """
        Triggers an audio buzzer in a non-blocking background thread.
        Honors cooldown period unless force is True.

        Returns:
            True if alarm was triggered, False if suppressed by cooldown.
        """
        if not AUDIO_ALARM_ENABLED:
            return False

        now = time.time()
        with self._lock:
            if not force and (now - self.last_alert_time) < self.cooldown_seconds:
                return False

            if self.is_playing:
                return False

            self.last_alert_time = now
            self.is_playing = True

        thread = threading.Thread(target=self._play_sound_worker, daemon=True)
        thread.start()
        return True

    def _play_sound_worker(self):
        """Worker executing in background thread to play tone without blocking UI."""
        try:
            if HAS_WINSOUND:
                # Windows hardware buzzer / beep
                winsound.Beep(self.frequency_hz, self.duration_ms)
            else:
                # Terminal bell character as fallback on Unix / non-windows
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception as e:
            # Fallback gracefully if sound driver is unavailable
            pass
        finally:
            with self._lock:
                self.is_playing = False

    @staticmethod
    def get_warning_message(state: str) -> Optional[str]:
        """Returns standard warning banner text according to state."""
        if state == "Drowsy":
            return "CRITICAL WARNING: HIGH DROWSINESS DETECTED! PLEASE PULL OVER AND TAKE A BREAK."
        elif state == "Early Fatigue":
            return "ADVISORY: SIGNS OF FATIGUE DETECTED. CONSIDER VENTILATION OR SCHEDULING A REST."
        return None


if __name__ == "__main__":
    print("[Self-Test] Testing AlertManager...")
    mgr = AlertManager(cooldown_seconds=1.0)
    msg = mgr.get_warning_message("Drowsy")
    print(f"Warning message for Drowsy state: {msg}")
    assert "CRITICAL" in msg
    print("Testing non-blocking buzzer trigger...")
    triggered = mgr.trigger_audio_alarm(force=True)
    print(f"Buzzer triggered: {triggered}")
    time.sleep(0.7)
    print("[Self-Test] AlertManager verified successfully.")
