"""
================================================================================
AI-Based Driver Fatigue and Drowsiness Prediction System
Module: src/head_pose.py
Purpose: 3D Head Pose estimation (Pitch, Yaw, Roll) via solvePnP and
         temporal head-nodding pattern detection.
         Member 1: Computer Vision & Feature Extraction
================================================================================
"""

import cv2
import time
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from collections import deque

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    HEAD_POSE_LANDMARK_INDICES,
    HEAD_PITCH_NOD_THRESHOLD,
    HEAD_NOD_DURATION_WINDOW
)

# Standard 3D facial model coordinates in millimeters (anthropometric model)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (landmark 1)
    (0.0, -330.0, -65.0),         # Chin (landmark 152)
    (-225.0, 170.0, -135.0),      # Left eye outer corner (landmark 33)
    (225.0, 170.0, -135.0),       # Right eye outer corner (landmark 263)
    (-150.0, -150.0, -125.0),     # Left mouth corner (landmark 61)
    (150.0, -150.0, -125.0)       # Right mouth corner (landmark 291)
], dtype=np.float64)


def estimate_head_pose(
    landmarks_px: np.ndarray,
    image_shape: Tuple[int, int]
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Estimates 3D head orientation (Pitch, Yaw, Roll) in degrees using cv2.solvePnP.

    Args:
        landmarks_px: Facial landmarks in pixel coordinates (>= 468 points).
        image_shape: (height, width) of the input video frame.

    Returns:
        (pitch, yaw, roll, rvec, tvec)
        - Pitch: +Up / -Down (nodding axis)
        - Yaw: +Turn Right / -Turn Left
        - Roll: +Tilt Right / -Tilt Left
    """
    h, w = image_shape[:2]
    if landmarks_px is None or len(landmarks_px) < 300 or h == 0 or w == 0:
        return 0.0, 0.0, 0.0, np.zeros((3, 1)), np.zeros((3, 1))

    # 2D image points corresponding to the 6 3D model landmarks
    image_points_2d = np.array([
        landmarks_px[1][:2],    # Nose tip
        landmarks_px[152][:2],  # Chin
        landmarks_px[33][:2],   # Left eye outer corner
        landmarks_px[263][:2],  # Right eye outer corner
        landmarks_px[61][:2],   # Left mouth corner
        landmarks_px[291][:2]   # Right mouth corner
    ], dtype=np.float64)

    # Approximate camera matrix assuming focal length = image width
    focal_length = float(w)
    center = (float(w / 2.0), float(h / 2.0))
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # Assume no lens distortion

    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0, np.zeros((3, 1)), np.zeros((3, 1))

    # Convert rotation vector to rotation matrix
    rotation_mat, _ = cv2.Rodrigues(rvec)

    # Decompose rotation matrix into Euler angles
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)

    # RQDecomp3x3 returns angles in degrees: [pitch, yaw, roll]
    pitch = float(angles[0])
    yaw = float(angles[1])
    roll = float(angles[2])

    # Convert to standard cockpit reference (-90 to +90 centered at 0 forward)
    # Accounting for 180° camera-to-world coordinate system flip
    if pitch > 90.0:
        pitch = pitch - 180.0
    elif pitch < -90.0:
        pitch = pitch + 180.0

    if roll > 90.0:
        roll = roll - 180.0
    elif roll < -90.0:
        roll = roll + 180.0

    return pitch, yaw, roll, rvec, tvec


def draw_head_pose_axes(
    frame_bgr: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_shape: Tuple[int, int],
    axis_length: float = 60.0
) -> np.ndarray:
    """Draws 3D Cartesian orientation axes (X=Red, Y=Green, Z=Blue) protruding from nose."""
    h, w = camera_shape[:2]
    focal_length = float(w)
    center = (float(w / 2.0), float(h / 2.0))
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # Define 3D axes protruding from nose origin (0, 0, 0)
    axis_points_3d = np.array([
        (0.0, 0.0, 0.0),
        (axis_length, 0.0, 0.0),     # X axis (Pitch/Right)
        (0.0, axis_length, 0.0),     # Y axis (Yaw/Down)
        (0.0, 0.0, axis_length)      # Z axis (Roll/Forward)
    ], dtype=np.float64)

    image_points_2d, _ = cv2.projectPoints(
        axis_points_3d, rvec, tvec, camera_matrix, dist_coeffs
    )

    p_origin = tuple(image_points_2d[0].ravel().astype(int))
    p_x = tuple(image_points_2d[1].ravel().astype(int))
    p_y = tuple(image_points_2d[2].ravel().astype(int))
    p_z = tuple(image_points_2d[3].ravel().astype(int))

    annotated = frame_bgr.copy()
    cv2.line(annotated, p_origin, p_x, (0, 0, 255), 2)  # Red: X
    cv2.line(annotated, p_origin, p_y, (0, 255, 0), 2)  # Green: Y
    cv2.line(annotated, p_origin, p_z, (255, 0, 0), 2)  # Blue: Z

    return annotated


class HeadNodTracker:
    """
    Tracks temporal downward head movements (nodding patterns).
    When a driver becomes drowsy, their head repeatedly sags forward (negative pitch)
    and then jerks or recovers upward.
    """

    def __init__(
        self,
        nod_pitch_threshold: float = HEAD_PITCH_NOD_THRESHOLD,
        nod_cycle_window: float = HEAD_NOD_DURATION_WINDOW
    ):
        self.nod_pitch_threshold = nod_pitch_threshold
        self.nod_cycle_window = nod_cycle_window

        self.is_dipping = False
        self.dip_start_timestamp: Optional[float] = None
        self.total_nods = 0
        self.recent_nods = deque(maxlen=20)

        # Recent pitch values for variation / standard deviation
        self.pitch_history = deque(maxlen=150)
        self.yaw_history = deque(maxlen=150)

    def set_nod_threshold(self, threshold: float):
        """Allows tuning based on driver's normal upright posture."""
        self.nod_pitch_threshold = threshold

    def update(
        self,
        pitch: float,
        yaw: float,
        roll: float,
        current_timestamp: Optional[float] = None,
        baseline_pitch: float = 0.0
    ) -> Dict[str, Any]:
        """
        Updates head nodding state with current head angles.

        Returns:
            Dict with:
            - pitch: float
            - yaw: float
            - roll: float
            - is_head_down: bool
            - nod_detected: bool
            - is_repeated_nodding: bool
            - recent_nod_count: int
            - total_nods: int
            - head_pose_variation: float (std dev of recent orientation)
        """
        if current_timestamp is None:
            current_timestamp = time.time()

        self.pitch_history.append(pitch)
        self.yaw_history.append(yaw)

        nod_detected = False
        delta_pitch = pitch - baseline_pitch
        # Head is dipping if below absolute threshold or relative dip from baseline >= 7.5 deg
        is_head_down = (pitch <= self.nod_pitch_threshold) or (delta_pitch <= -7.5)

        current_dip_duration = 0.0
        if is_head_down:
            if not self.is_dipping:
                self.is_dipping = True
                self.dip_start_timestamp = current_timestamp
            current_dip_duration = current_timestamp - (self.dip_start_timestamp or current_timestamp)
        else:
            if self.is_dipping:
                # Driver's head recovered back up!
                duration = current_timestamp - (self.dip_start_timestamp or current_timestamp)
                if 0.12 <= duration <= self.nod_cycle_window:
                    self.total_nods += 1
                    nod_detected = True
                    self.recent_nods.append(current_timestamp)
                self.is_dipping = False
                self.dip_start_timestamp = None

        # Check for repeated nodding: multiple nods in rolling 5s window
        recent_nods_count = sum(1 for t in self.recent_nods if (current_timestamp - t) <= 5.0)
        is_repeated_nodding = bool(recent_nods_count >= 2)

        # Standard deviation of head pose over recent frames
        pose_var = float(np.std(self.pitch_history)) if len(self.pitch_history) > 5 else 0.0

        # Rapid nodding detection: vertical oscillation with amplitude >= 9 deg
        recent_pitches = list(self.pitch_history)[-30:] if len(self.pitch_history) >= 15 else []
        if len(recent_pitches) >= 15:
            pitch_range = max(recent_pitches) - min(recent_pitches)
            if pitch_range >= 9.0 and min(recent_pitches) <= -7.0 and pose_var >= 3.0:
                is_repeated_nodding = True

        return {
            "pitch": float(pitch),
            "yaw": float(yaw),
            "roll": float(roll),
            "is_head_down": bool(is_head_down),
            "head_down_duration": float(current_dip_duration),
            "nod_detected": bool(nod_detected),
            "is_repeated_nodding": bool(is_repeated_nodding),
            "recent_nod_count": int(recent_nods_count),
            "total_nods": int(self.total_nods),
            "head_pose_variation": float(round(pose_var, 3))
        }

    def reset(self):
        """Resets counters and history."""
        self.is_dipping = False
        self.dip_start_timestamp = None
        self.total_nods = 0
        self.recent_nods.clear()
        self.pitch_history.clear()
        self.yaw_history.clear()


if __name__ == "__main__":
    print("[Self-Test] Testing Head Pose & Nod Tracker...")
    # Mock landmarks for head pose
    mock_lms = np.zeros((468, 3), dtype=np.float32)
    # Nose
    mock_lms[1] = [320, 240, 0]
    # Chin
    mock_lms[152] = [320, 340, 0]
    # Left eye outer
    mock_lms[33] = [260, 200, 0]
    # Right eye outer
    mock_lms[263] = [380, 200, 0]
    # Mouth corners
    mock_lms[61] = [280, 300, 0]
    mock_lms[291] = [360, 300, 0]

    pitch, yaw, roll, rvec, tvec = estimate_head_pose(mock_lms, (480, 640))
    print(f"Estimated Pose on Frontal Face -> Pitch: {pitch:.1f}, Yaw: {yaw:.1f}, Roll: {roll:.1f}")

    tracker = HeadNodTracker(nod_pitch_threshold=-15.0, nod_cycle_window=2.0)
    t = 0.0
    tracker.update(pitch=0.0, yaw=0.0, roll=0.0, current_timestamp=t)
    t += 0.5
    # Head dips down
    tracker.update(pitch=-20.0, yaw=0.0, roll=0.0, current_timestamp=t)
    t += 0.8
    # Head jerks back up
    out = tracker.update(pitch=-2.0, yaw=0.0, roll=0.0, current_timestamp=t)
    assert out["nod_detected"] and out["total_nods"] == 1, "Head nod detection failed"
    print(f"Detected head nod successfully: {out}")
    print("[Self-Test] Head pose verified successfully.")
