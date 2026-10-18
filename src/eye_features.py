"""
================================================================================
AI-Based Driver Fatigue and Drowsiness Prediction System
Module: src/eye_features.py
Purpose: Eye Aspect Ratio (EAR), blink detection, blink duration measurement,
         and prolonged eye closure tracking.
         Member 1: Computer Vision & Feature Extraction
================================================================================
"""

import time
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from collections import deque

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    DEFAULT_EAR_THRESHOLD,
    BLINK_MIN_DURATION_SECONDS,
    BLINK_MAX_DURATION_SECONDS,
    PROLONGED_CLOSURE_SECONDS
)


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Computes standard Euclidean distance between two 2D/3D points."""
    return float(np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2])))


def calculate_ear(eye_landmarks: np.ndarray) -> float:
    """
    Computes Eye Aspect Ratio (EAR) for a single eye given 6 landmark points:
    Order: [corner_outer, top_1, top_2, corner_inner, bottom_2, bottom_1]
    Formula: (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
    """
    if eye_landmarks is None or len(eye_landmarks) < 6:
        return 0.0

    p1 = eye_landmarks[0]
    p2 = eye_landmarks[1]
    p3 = eye_landmarks[2]
    p4 = eye_landmarks[3]
    p5 = eye_landmarks[4]
    p6 = eye_landmarks[5]

    # Vertical distances
    dist_v1 = euclidean_distance(p2, p6)
    dist_v2 = euclidean_distance(p3, p5)

    # Horizontal distance
    dist_h = euclidean_distance(p1, p4)

    if dist_h == 0:
        return 0.0

    ear = (dist_v1 + dist_v2) / (2.0 * dist_h)
    return float(ear)


def extract_eye_landmarks(landmarks_px: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts left and right eye landmark points from the 468/478 landmarks array.
    """
    left_eye = landmarks_px[LEFT_EYE_INDICES]
    right_eye = landmarks_px[RIGHT_EYE_INDICES]
    return left_eye, right_eye


class EyeBlinkTracker:
    """
    State-machine based eye blink tracker.
    Distinguishes between:
      1. Normal blinks (0.08s - 0.45s)
      2. Flutter blinks
      3. Prolonged eye closures / micro-sleeps (> 1.2s)
    Maintains blink counts, current closure duration, and recent blink durations.
    """

    def __init__(
        self,
        ear_threshold: float = DEFAULT_EAR_THRESHOLD,
        min_blink_sec: float = BLINK_MIN_DURATION_SECONDS,
        max_blink_sec: float = BLINK_MAX_DURATION_SECONDS,
        prolonged_closure_sec: float = PROLONGED_CLOSURE_SECONDS
    ):
        self.ear_threshold = ear_threshold
        self.min_blink_sec = min_blink_sec
        self.max_blink_sec = max_blink_sec
        self.prolonged_closure_sec = prolonged_closure_sec

        # State tracking
        self.is_closed = False
        self.closure_start_timestamp: Optional[float] = None
        self.current_closure_duration: float = 0.0

        # Metrics history
        self.total_blinks = 0
        self.prolonged_closure_count = 0
        self.recent_blink_durations: deque = deque(maxlen=50)

    def set_ear_threshold(self, threshold: float):
        """Allows dynamic adjustment based on personalized calibration."""
        self.ear_threshold = max(0.12, min(0.18, threshold))

    def update(self, ear: float, current_timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Updates tracker state with current EAR and timestamp.

        Args:
            ear: Current average EAR value.
            current_timestamp: Time in seconds (defaults to time.time()).

        Returns:
            Dict containing instantaneous metrics:
            - is_eye_closed: bool
            - blink_occurred: bool
            - last_blink_duration: float
            - current_closure_duration: float
            - is_prolonged_closure: bool
            - total_blinks: int
        """
        if current_timestamp is None:
            current_timestamp = time.time()

        blink_occurred = False
        last_blink_duration = 0.0
        is_prolonged = False

        if ear < self.ear_threshold:
            # Eyes are currently closed
            if not self.is_closed:
                self.is_closed = True
                self.closure_start_timestamp = current_timestamp
                self.current_closure_duration = 0.0
            else:
                self.current_closure_duration = current_timestamp - (self.closure_start_timestamp or current_timestamp)

            if self.current_closure_duration >= self.prolonged_closure_sec:
                is_prolonged = True
        else:
            # Eyes are currently open
            if self.is_closed:
                # Transition: Closed -> Open
                closure_duration = current_timestamp - (self.closure_start_timestamp or current_timestamp)
                self.is_closed = False
                self.closure_start_timestamp = None
                self.current_closure_duration = 0.0

                if closure_duration >= self.prolonged_closure_sec:
                    self.prolonged_closure_count += 1
                    is_prolonged = True

                if self.min_blink_sec <= closure_duration <= self.max_blink_sec:
                    self.total_blinks += 1
                    blink_occurred = True
                    last_blink_duration = closure_duration
                    self.recent_blink_durations.append(closure_duration)
                elif closure_duration > self.max_blink_sec:
                    # Slow blink or prolonged closure
                    self.total_blinks += 1
                    blink_occurred = True
                    last_blink_duration = closure_duration
                    self.recent_blink_durations.append(closure_duration)
            else:
                self.current_closure_duration = 0.0

        avg_blink_dur = (
            float(np.mean(self.recent_blink_durations))
            if len(self.recent_blink_durations) > 0
            else 0.25
        )

        return {
            "ear": float(ear),
            "is_eye_closed": bool(self.is_closed),
            "blink_occurred": blink_occurred,
            "last_blink_duration": float(last_blink_duration),
            "current_closure_duration": float(self.current_closure_duration),
            "is_prolonged_closure": is_prolonged,
            "total_blinks": int(self.total_blinks),
            "avg_recent_blink_duration": float(avg_blink_dur)
        }

    def reset(self):
        """Resets counters and state."""
        self.is_closed = False
        self.closure_start_timestamp = None
        self.current_closure_duration = 0.0
        self.total_blinks = 0
        self.prolonged_closure_count = 0
        self.recent_blink_durations.clear()


if __name__ == "__main__":
    print("[Self-Test] Testing Eye Features & Blink Tracker...")
    # Generate mock eye landmarks: 6 points for eye
    # Top points y=10, bottom points y=20, width=40 -> EAR = (10+10)/(2*40) = 0.25
    mock_open_eye = np.array([
        [100, 15],  # p1: outer corner
        [115, 10],  # p2: top-1
        [125, 10],  # p3: top-2
        [140, 15],  # p4: inner corner
        [125, 20],  # p5: bottom-2
        [115, 20]   # p6: bottom-1
    ], dtype=np.float32)

    ear_open = calculate_ear(mock_open_eye)
    print(f"Calculated open EAR: {ear_open:.3f}")
    assert ear_open > 0.20, "EAR calculation check failed"

    tracker = EyeBlinkTracker(ear_threshold=0.20)
    t = 0.0
    # Simulate open eyes
    out1 = tracker.update(ear=0.28, current_timestamp=t)
    assert not out1["is_eye_closed"]

    # Simulate blink: close eyes for 0.2s
    t += 0.033
    out2 = tracker.update(ear=0.15, current_timestamp=t)
    assert out2["is_eye_closed"]

    t += 0.20
    out3 = tracker.update(ear=0.28, current_timestamp=t)
    assert out3["blink_occurred"], "Blink should have been detected!"
    print(f"Detected blink with duration: {out3['last_blink_duration']:.3f}s")
    print("[Self-Test] Eye features verified successfully.")
