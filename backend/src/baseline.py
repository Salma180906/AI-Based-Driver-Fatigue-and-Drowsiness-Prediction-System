"""
================================================================================
AI-Based Driver Fatigue and Drowsiness Prediction System
Module: src/baseline.py
Purpose: Personalized baseline calibration (30-60s) establishing the driver's
         individual physiological and behavioral norms.
         Member 2: Machine Learning & Application
================================================================================
"""

import time
import numpy as np
from typing import Dict, Any, Optional, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CALIBRATION_DURATION_SECONDS,
    DEFAULT_EAR_THRESHOLD
)


class PersonalizedBaseline:
    """
    Learns the driver's individual alert baseline during an initial calibration
    phase (30–60 seconds). Calculates deviations between live telemetry and
    the personalized baseline to adapt to different facial geometries.
    """

    def __init__(self, calibration_duration_sec: float = CALIBRATION_DURATION_SECONDS):
        self.calibration_duration_sec = calibration_duration_sec

        self.is_calibrating = False
        self.is_calibrated = False
        self.is_auto_calibrated = False
        self.calibration_start_time: Optional[float] = None

        # Buffers for explicit manual calibration telemetry
        self.calib_ears: List[float] = []
        self.calib_mars: List[float] = []
        self.calib_pitches: List[float] = []
        self.calib_yaws: List[float] = []
        self.calib_gazes: List[float] = []
        self.calib_blink_durations: List[float] = []

        # Rapid auto-calibration buffer (accumulates during first 25-30 alert frames)
        self.auto_ears: List[float] = []
        self.auto_mars: List[float] = []
        self.auto_pitches: List[float] = []
        self.auto_yaws: List[float] = []
        self.auto_samples_needed: int = 25

        # Stored baseline reference values
        self.baseline_ear: float = 0.28
        self.baseline_dynamic_ear_threshold: float = DEFAULT_EAR_THRESHOLD
        self.baseline_mar: float = 0.25
        self.baseline_pitch: float = 0.0
        self.baseline_yaw: float = 0.0
        self.baseline_blink_duration: float = 0.20
        self.baseline_gaze_stability: float = 1.0

    def reset(self):
        """Resets all baseline calibration buffers."""
        self.is_calibrating = False
        self.is_calibrated = False
        self.is_auto_calibrated = False
        self.calibration_start_time = None
        self.calib_ears.clear()
        self.calib_mars.clear()
        self.calib_pitches.clear()
        self.calib_yaws.clear()
        self.calib_gazes.clear()
        self.calib_blink_durations.clear()
        self.auto_ears.clear()
        self.auto_mars.clear()
        self.auto_pitches.clear()
        self.auto_yaws.clear()
        self.baseline_ear = 0.28
        self.baseline_dynamic_ear_threshold = DEFAULT_EAR_THRESHOLD
        self.baseline_mar = 0.25
        self.baseline_pitch = 0.0
        self.baseline_yaw = 0.0

    def start_calibration(self):
        """Initiates the personalized calibration period."""
        self.is_calibrating = True
        self.is_calibrated = False
        self.calibration_start_time = time.time()
        self.calib_ears.clear()
        self.calib_mars.clear()
        self.calib_pitches.clear()
        self.calib_yaws.clear()
        self.calib_gazes.clear()
        self.calib_blink_durations.clear()

    def update_auto_baseline(self, ear: float, mar: float, pitch: float, yaw: float):
        """
        Rapidly learns the driver's resting baseline in the first 25 frames
        without requiring manual intervention.
        """
        if self.is_calibrated:
            # If already explicitly calibrated, just smoothly adapt if alert
            if ear > self.baseline_dynamic_ear_threshold and mar < 0.40 and abs(yaw - self.baseline_yaw) < 15.0:
                self.baseline_ear = float(0.995 * self.baseline_ear + 0.005 * ear)
            return

        if len(self.auto_ears) < self.auto_samples_needed:
            if ear > 0.12 and mar < 0.45:
                self.auto_ears.append(ear)
                self.auto_mars.append(mar)
                self.auto_pitches.append(pitch)
                self.auto_yaws.append(yaw)

            if len(self.auto_ears) >= self.auto_samples_needed:
                # Establish initial baseline
                self.baseline_ear = float(np.percentile(self.auto_ears, 70))
                # Set adaptive threshold strictly bounded so open eyes are NEVER classified as closed
                self.baseline_dynamic_ear_threshold = float(max(0.13, min(0.18, self.baseline_ear * 0.65)))
                self.baseline_mar = float(np.median(self.auto_mars))
                self.baseline_pitch = float(np.median(self.auto_pitches))
                self.baseline_yaw = float(np.median(self.auto_yaws))
                self.is_auto_calibrated = True

    def update_calibration(
        self,
        ear: float,
        mar: float,
        pitch: float,
        gaze_ratio: float,
        yaw: float = 0.0,
        blink_duration: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Ingests a frame's metrics during explicit calibration and checks completion.
        """
        if not self.is_calibrating:
            return {
                "is_calibrating": False,
                "is_calibrated": self.is_calibrated or self.is_auto_calibrated,
                "progress": 1.0 if (self.is_calibrated or self.is_auto_calibrated) else 0.0,
                "remaining_seconds": 0.0
            }

        now = time.time()
        elapsed = now - (self.calibration_start_time or now)
        progress = min(1.0, elapsed / self.calibration_duration_sec)
        remaining = max(0.0, self.calibration_duration_sec - elapsed)

        # Collect observations
        if 0.10 <= ear <= 0.45:
            self.calib_ears.append(ear)
        if 0.10 <= mar <= 0.90:
            self.calib_mars.append(mar)
        self.calib_pitches.append(pitch)
        self.calib_yaws.append(yaw)
        self.calib_gazes.append(gaze_ratio)
        if blink_duration and blink_duration > 0.05:
            self.calib_blink_durations.append(blink_duration)

        if elapsed >= self.calibration_duration_sec:
            self._finalize_calibration()

        return {
            "is_calibrating": self.is_calibrating,
            "is_calibrated": self.is_calibrated,
            "progress": round(progress, 2),
            "remaining_seconds": round(remaining, 1),
            "samples_collected": len(self.calib_ears)
        }

    def _finalize_calibration(self):
        """Computes reference values from collected calibration samples."""
        self.is_calibrating = False
        self.is_calibrated = True

        if len(self.calib_ears) > 10:
            self.baseline_ear = float(np.percentile(self.calib_ears, 75))
            self.baseline_dynamic_ear_threshold = float(max(0.13, min(0.18, self.baseline_ear * 0.65)))
        else:
            self.baseline_ear = 0.28
            self.baseline_dynamic_ear_threshold = 0.17

        if len(self.calib_mars) > 10:
            self.baseline_mar = float(np.median(self.calib_mars))

        if len(self.calib_pitches) > 10:
            self.baseline_pitch = float(np.median(self.calib_pitches))

        if len(self.calib_yaws) > 10:
            self.baseline_yaw = float(np.median(self.calib_yaws))

        if len(self.calib_blink_durations) > 0:
            self.baseline_blink_duration = float(np.mean(self.calib_blink_durations))
        else:
            self.baseline_blink_duration = 0.20

        self.baseline_gaze_stability = 1.0

    def compute_deviations(
        self,
        current_ear_mean: float,
        current_avg_blink_duration: float,
        current_pitch: float,
        current_yaw: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculates relative deviations from the personalized baseline.
        """
        ear_dev = (self.baseline_ear - current_ear_mean) / max(0.1, self.baseline_ear)
        ear_dev = float(np.clip(ear_dev, -0.5, 1.0))

        blink_ratio = current_avg_blink_duration / max(0.1, self.baseline_blink_duration)
        blink_ratio = float(np.clip(blink_ratio, 0.5, 4.0))

        pitch_sag = float(self.baseline_pitch - current_pitch)
        yaw_dev = float(abs(current_yaw - self.baseline_yaw))

        return {
            "EAR_deviation_from_baseline": round(ear_dev, 4),
            "blink_duration_ratio": round(blink_ratio, 3),
            "pitch_deviation": round(pitch_sag, 2),
            "yaw_deviation": round(yaw_dev, 2)
        }

    def get_baseline_summary(self) -> Dict[str, Any]:
        """Returns the active baseline configuration."""
        return {
            "is_calibrated": self.is_calibrated or self.is_auto_calibrated,
            "baseline_ear": round(self.baseline_ear, 3),
            "dynamic_ear_threshold": round(self.baseline_dynamic_ear_threshold, 3),
            "baseline_mar": round(self.baseline_mar, 3),
            "baseline_pitch": round(self.baseline_pitch, 1),
            "baseline_yaw": round(self.baseline_yaw, 1),
            "baseline_blink_duration": round(self.baseline_blink_duration, 3)
        }


if __name__ == "__main__":
    print("[Self-Test] Testing PersonalizedBaseline...")
    calib = PersonalizedBaseline(calibration_duration_sec=1.0)
    calib.start_calibration()
    # Feed 30 alert frames
    for _ in range(30):
        calib.update_calibration(
            ear=0.30,
            mar=0.28,
            pitch=2.0,
            gaze_ratio=0.5,
            blink_duration=0.18
        )
    time.sleep(1.1)
    status = calib.update_calibration(ear=0.30, mar=0.28, pitch=2.0, gaze_ratio=0.5)
    assert status["is_calibrated"], "Baseline calibration should be complete"
    summary = calib.get_baseline_summary()
    print(f"Calibrated summary: {summary}")

    # Test deviation when eyes droop to 0.21
    devs = calib.compute_deviations(
        current_ear_mean=0.21,
        current_avg_blink_duration=0.36,
        current_pitch=-10.0
    )
    print(f"Deviations under drowsiness: {devs}")
    assert devs["EAR_deviation_from_baseline"] > 0.20, "Deviation should detect eye drooping"
    print("[Self-Test] Personalized baseline verified successfully.")
