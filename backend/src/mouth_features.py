"""
================================================================================
AI-Based Driver Fatigue and Drowsiness Prediction System
Module: src/mouth_features.py
Purpose: Mouth Aspect Ratio (MAR) computation and duration-based yawn detection.
         Member 1: Computer Vision & Feature Extraction
================================================================================
"""

import time
import numpy as np
from typing import Tuple, Optional, Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DEFAULT_MAR_YAWN_THRESHOLD,
    YAWN_MIN_DURATION_SECONDS,
    MOUTH_CORNER_LEFT,
    MOUTH_CORNER_RIGHT,
    MOUTH_UPPER_LIP,
    MOUTH_LOWER_LIP
)


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Computes standard Euclidean distance between two 2D/3D points."""
    return float(np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2])))


def calculate_mar(landmarks_px: np.ndarray) -> float:
    """
    Computes Mouth Aspect Ratio (MAR) using 3 vertical distances across lips
    divided by horizontal distance between mouth corners.

    Landmarks used:
      - Left corner: 61, Right corner: 291
      - Upper inner lip: 13, Lower inner lip: 14
      - Upper lip secondary: 81, Lower lip secondary: 178
      - Upper lip tertiary: 311, Lower lip tertiary: 402

    Formula: (|p81 - p178| + |p13 - p14| + |p311 - p402|) / (3 * |p61 - p291|)
    """
    if landmarks_px is None or len(landmarks_px) < 468:
        return 0.0

    p_corner_l = landmarks_px[61]
    p_corner_r = landmarks_px[291]

    p_upper_mid = landmarks_px[13]
    p_lower_mid = landmarks_px[14]

    p_upper_l = landmarks_px[81]
    p_lower_l = landmarks_px[178]

    p_upper_r = landmarks_px[311]
    p_lower_r = landmarks_px[402]

    dist_h = euclidean_distance(p_corner_l, p_corner_r)
    if dist_h == 0:
        return 0.0

    dist_v1 = euclidean_distance(p_upper_l, p_lower_l)
    dist_v2 = euclidean_distance(p_upper_mid, p_lower_mid)
    dist_v3 = euclidean_distance(p_upper_r, p_lower_r)

    mar = (dist_v1 + dist_v2 + dist_v3) / (3.0 * dist_h)
    return float(mar)


class YawnTracker:
    """
    Duration-based yawn detector.
    Distinguishes talking or smiling from actual yawning by requiring
    mouth opening (MAR >= threshold) to persist for >= YAWN_MIN_DURATION_SECONDS (e.g. 1.5s).
    """

    def __init__(
        self,
        mar_threshold: float = DEFAULT_MAR_YAWN_THRESHOLD,
        min_yawn_sec: float = YAWN_MIN_DURATION_SECONDS
    ):
        self.mar_threshold = mar_threshold
        self.min_yawn_sec = min_yawn_sec

        # State tracking
        self.is_open = False
        self.open_start_timestamp: Optional[float] = None
        self.current_open_duration: float = 0.0
        self.yawn_registered_for_current_opening = False

        self.total_yawns = 0
        self.recent_yawn_durations = []

    def set_mar_threshold(self, threshold: float):
        """Allows tuning based on personal mouth geometry."""
        self.mar_threshold = max(0.40, min(0.85, threshold))

    def update(self, mar: float, current_timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Updates yawn state with current MAR and timestamp.

        Returns:
            Dict containing:
            - mar: float
            - is_yawning: bool (actively yawning now)
            - yawn_completed: bool (just finished a yawn)
            - current_yawn_duration: float
            - total_yawns: int
        """
        if current_timestamp is None:
            current_timestamp = time.time()

        yawn_completed = False
        is_actively_yawning = False

        if mar >= self.mar_threshold:
            # Mouth is wide open
            if not self.is_open:
                self.is_open = True
                self.open_start_timestamp = current_timestamp
                self.current_open_duration = 0.0
                self.yawn_registered_for_current_opening = False
            else:
                self.current_open_duration = current_timestamp - (self.open_start_timestamp or current_timestamp)

            # Check if duration qualifies as a yawn
            if self.current_open_duration >= self.min_yawn_sec:
                is_actively_yawning = True
                if not self.yawn_registered_for_current_opening:
                    self.total_yawns += 1
                    self.yawn_registered_for_current_opening = True
        else:
            # Mouth is closed / normal
            if self.is_open:
                duration = current_timestamp - (self.open_start_timestamp or current_timestamp)
                if self.yawn_registered_for_current_opening:
                    yawn_completed = True
                    self.recent_yawn_durations.append(duration)
                self.is_open = False
                self.open_start_timestamp = None
                self.current_open_duration = 0.0
                self.yawn_registered_for_current_opening = False
            else:
                self.current_open_duration = 0.0

        return {
            "mar": float(mar),
            "is_yawning": is_actively_yawning,
            "yawn_completed": yawn_completed,
            "current_yawn_duration": float(self.current_open_duration),
            "total_yawns": int(self.total_yawns)
        }

    def reset(self):
        """Resets counters and state."""
        self.is_open = False
        self.open_start_timestamp = None
        self.current_open_duration = 0.0
        self.yawn_registered_for_current_opening = False
        self.total_yawns = 0
        self.recent_yawn_durations.clear()


if __name__ == "__main__":
    print("[Self-Test] Testing Mouth Features & Yawn Tracker...")
    # Mock landmarks for mouth
    mock_lms = np.zeros((468, 3), dtype=np.float32)
    # Set lip corners
    mock_lms[61] = [100, 100, 0]
    mock_lms[291] = [200, 100, 0]
    # Set vertical lip points (open mouth)
    mock_lms[13] = [150, 70, 0]   # upper
    mock_lms[14] = [150, 130, 0]  # lower (diff 60)
    mock_lms[81] = [125, 75, 0]
    mock_lms[178] = [125, 125, 0] # diff 50
    mock_lms[311] = [175, 75, 0]
    mock_lms[402] = [175, 125, 0] # diff 50

    mar_val = calculate_mar(mock_lms)
    print(f"Calculated MAR on mock open mouth: {mar_val:.3f}")
    assert mar_val > 0.50, "MAR calculation check failed"

    tracker = YawnTracker(mar_threshold=0.50, min_yawn_sec=1.5)
    t = 0.0
    tracker.update(mar=0.30, current_timestamp=t)
    t += 0.5
    tracker.update(mar=0.65, current_timestamp=t)
    t += 1.6  # > 1.5s open
    res = tracker.update(mar=0.65, current_timestamp=t)
    assert res["is_yawning"] and res["total_yawns"] == 1, "Yawn detection failed"
    print(f"Detected yawn successfully: {res}")
    print("[Self-Test] Mouth features verified successfully.")
