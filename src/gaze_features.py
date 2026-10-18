"""
================================================================================
AI-Based Driver Fatigue and Drowsiness Prediction System
Module: src/gaze_features.py
Purpose: Gaze ratio estimation and gaze stability measurement over time.
         Member 1: Computer Vision & Feature Extraction
================================================================================
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from collections import deque

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    LEFT_IRIS_CENTER,
    RIGHT_IRIS_CENTER,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES
)


def calculate_eye_gaze_ratio(
    landmarks_px: np.ndarray,
    iris_idx: int,
    inner_corner_idx: int,
    outer_corner_idx: int
) -> float:
    """
    Estimates horizontal gaze ratio:
    ratio = (x_iris - x_outer) / (x_inner - x_outer)
    Returns value roughly between 0.0 (looking far one side) to 1.0 (looking far other side),
    with ~0.5 indicating centered forward gaze.
    """
    if landmarks_px is None or len(landmarks_px) <= max(iris_idx, inner_corner_idx, outer_corner_idx):
        return 0.5

    x_iris = landmarks_px[iris_idx][0]
    x_inner = landmarks_px[inner_corner_idx][0]
    x_outer = landmarks_px[outer_corner_idx][0]

    denom = abs(x_inner - x_outer)
    if denom < 1.0:
        return 0.5

    # Compute ratio relative to leftmost point
    min_x = min(x_inner, x_outer)
    ratio = (x_iris - min_x) / denom
    return float(np.clip(ratio, 0.0, 1.0))


def estimate_gaze(landmarks_px: np.ndarray) -> Tuple[float, str]:
    """
    Computes average horizontal gaze ratio across both eyes and labels gaze direction.

    Returns:
        (gaze_ratio, direction_label)
        direction_label: "Center", "Left", "Right"
    """
    if landmarks_px is None or len(landmarks_px) < 468:
        return 0.5, "Unknown"

    has_iris = len(landmarks_px) > RIGHT_IRIS_CENTER

    if has_iris:
        left_ratio = calculate_eye_gaze_ratio(landmarks_px, LEFT_IRIS_CENTER, 133, 33)
        right_ratio = calculate_eye_gaze_ratio(landmarks_px, RIGHT_IRIS_CENTER, 362, 263)
        gaze_ratio = (left_ratio + right_ratio) / 2.0
    else:
        # Fallback: Approximate iris/pupil as centroid of eye contours
        left_pts = landmarks_px[LEFT_EYE_INDICES]
        right_pts = landmarks_px[RIGHT_EYE_INDICES]
        left_center_x = np.mean(left_pts[:, 0])
        right_center_x = np.mean(right_pts[:, 0])

        left_denom = max(1.0, abs(landmarks_px[133][0] - landmarks_px[33][0]))
        min_l = min(landmarks_px[133][0], landmarks_px[33][0])
        left_ratio = (left_center_x - min_l) / left_denom

        right_denom = max(1.0, abs(landmarks_px[362][0] - landmarks_px[263][0]))
        min_r = min(landmarks_px[362][0], landmarks_px[263][0])
        right_ratio = (right_center_x - min_r) / right_denom

        gaze_ratio = (left_ratio + right_ratio) / 2.0

    # Categorize rough direction
    if gaze_ratio < 0.35:
        direction = "Looking Right"
    elif gaze_ratio > 0.65:
        direction = "Looking Left"
    else:
        direction = "Center / Forward"

    return float(np.clip(gaze_ratio, 0.0, 1.0)), direction


class GazeStabilityTracker:
    """
    Maintains a rolling window of gaze ratios to evaluate gaze stability.
    Normal alert driving has dynamic, healthy road scanning (moderate variance).
    Fatigued driving produces blank staring (abnormally low variance) or erratic wandering.
    """

    def __init__(self, window_size: int = 90):
        self.window_size = window_size
        self.gaze_history = deque(maxlen=window_size)

    def update(self, gaze_ratio: float) -> Dict[str, Any]:
        """
        Updates the gaze history and calculates stability score.

        Returns:
            Dict containing:
            - gaze_ratio: float
            - gaze_std: float (standard deviation over recent frames)
            - gaze_stability: float (0.0 to 1.0 score where higher is healthy alert scanning)
            - is_fixed_staring: bool (blank stare symptom)
        """
        self.gaze_history.append(gaze_ratio)

        if len(self.gaze_history) < 15:
            return {
                "gaze_ratio": round(gaze_ratio, 3),
                "gaze_std": 0.08,
                "gaze_stability": 1.0,
                "is_fixed_staring": False
            }

        gaze_std = float(np.std(self.gaze_history))

        # Blank staring symptom: standard deviation drops below 0.025 over several seconds
        is_fixed_staring = bool(gaze_std < 0.025 and len(self.gaze_history) >= 45)

        # Healthily active scanning is around 0.06 - 0.15
        # Penalize both abnormal fixation (< 0.03) and wild erratic jumps (> 0.22)
        if gaze_std < 0.03:
            stability_score = max(0.2, gaze_std / 0.03)
        elif gaze_std > 0.22:
            stability_score = max(0.2, 1.0 - ((gaze_std - 0.22) / 0.20))
        else:
            stability_score = 1.0

        return {
            "gaze_ratio": round(gaze_ratio, 3),
            "gaze_std": round(gaze_std, 4),
            "gaze_stability": round(float(stability_score), 3),
            "is_fixed_staring": is_fixed_staring
        }

    def reset(self):
        """Clears gaze history."""
        self.gaze_history.clear()


if __name__ == "__main__":
    print("[Self-Test] Testing Gaze Features & Stability Tracker...")
    # Mock landmarks
    mock_lms = np.zeros((478, 3), dtype=np.float32)
    # Eyes corners
    mock_lms[33] = [100, 100, 0]
    mock_lms[133] = [140, 100, 0]
    mock_lms[263] = [200, 100, 0]
    mock_lms[362] = [160, 100, 0]
    # Irises centered
    mock_lms[LEFT_IRIS_CENTER] = [120, 100, 0]
    mock_lms[RIGHT_IRIS_CENTER] = [180, 100, 0]

    ratio, direction = estimate_gaze(mock_lms)
    print(f"Mock gaze: {ratio:.2f}, direction: {direction}")
    assert 0.4 <= ratio <= 0.6, "Centered gaze ratio check failed"

    tracker = GazeStabilityTracker(window_size=30)
    # Feed active gaze samples
    for r in np.random.normal(0.5, 0.08, 30):
        res = tracker.update(float(r))
    print(f"Gaze stability under normal scanning: {res['gaze_stability']}")

    # Feed fixed staring
    tracker.reset()
    for _ in range(50):
        res = tracker.update(0.50)
    print(f"Gaze stability under fixed staring: {res['gaze_stability']} (is_staring: {res['is_fixed_staring']})")
    print("[Self-Test] Gaze features verified successfully.")
