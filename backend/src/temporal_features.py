"""
================================================================================
AI-Based Driver Fatigue and Drowsiness Prediction System
Module: src/temporal_features.py
Purpose: Rolling temporal window aggregation (10–30s) computing PERCLOS,
         blink frequency, blink duration, yawn rate, nodding rate, and statistics.
         Member 1 & Member 2 Integration
================================================================================
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ROLLING_WINDOW_SECONDS,
    DEFAULT_EAR_THRESHOLD
)


class TemporalWindowAggregator:
    """
    Maintains a FIFO sliding window of frame-level telemetry over a defined duration
    (e.g., 10–30 seconds) to extract robust temporal behavioral features.
    """

    def __init__(
        self,
        window_seconds: float = ROLLING_WINDOW_SECONDS,
        ear_threshold: float = DEFAULT_EAR_THRESHOLD
    ):
        self.window_seconds = window_seconds
        self.ear_threshold = ear_threshold

        # Frame samples: (timestamp, ear, is_closed, mar, pitch, yaw, roll, gaze_ratio)
        self.frame_buffer: deque = deque()

        # Timestamped discrete event logs: (timestamp, duration_or_value)
        self.blink_events: deque = deque()
        self.yawn_events: deque = deque()
        self.nod_events: deque = deque()

    def set_ear_threshold(self, threshold: float):
        """Updates EAR threshold used for PERCLOS calculation."""
        self.ear_threshold = threshold

    def add_frame_sample(
        self,
        timestamp: float,
        ear: float,
        is_closed: bool,
        mar: float,
        pitch: float,
        yaw: float,
        roll: float,
        gaze_ratio: float
    ):
        """Appends a new frame sample and purges records older than window_seconds."""
        self.frame_buffer.append({
            "t": timestamp,
            "ear": ear,
            "is_closed": is_closed,
            "mar": mar,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "gaze": gaze_ratio
        })
        self._purge_old_data(timestamp)

    def log_blink_event(self, timestamp: float, duration: float):
        """Logs a completed blink event with its duration."""
        self.blink_events.append((timestamp, duration))

    def log_yawn_event(self, timestamp: float, duration: float):
        """Logs a completed yawn event with its duration."""
        self.yawn_events.append((timestamp, duration))

    def log_nod_event(self, timestamp: float):
        """Logs a completed head nodding cycle."""
        self.nod_events.append((timestamp, 1.0))

    def _purge_old_data(self, current_time: float):
        """Removes data outside the rolling temporal window."""
        cutoff = current_time - self.window_seconds

        while self.frame_buffer and self.frame_buffer[0]["t"] < cutoff:
            self.frame_buffer.popleft()

        while self.blink_events and self.blink_events[0][0] < cutoff:
            self.blink_events.popleft()

        while self.yawn_events and self.yawn_events[0][0] < cutoff:
            self.yawn_events.popleft()

        while self.nod_events and self.nod_events[0][0] < cutoff:
            self.nod_events.popleft()

    def get_temporal_features(self, current_time: Optional[float] = None) -> Dict[str, float]:
        """
        Computes aggregated temporal statistical features over the rolling window.

        Returns:
            Dict matching ML feature columns:
            - EAR_mean
            - EAR_min
            - blink_count
            - blink_frequency (blinks/minute)
            - avg_blink_duration
            - max_eye_closure_duration
            - PERCLOS (proportion of frames eye was closed)
            - MAR_mean
            - yawn_count
            - head_nod_count
            - head_pose_variation
            - gaze_stability
            - fatigue_trend_val
        """
        if current_time is None:
            current_time = time.time()

        self._purge_old_data(current_time)

        num_frames = len(self.frame_buffer)
        if num_frames == 0:
            return {
                "EAR_mean": 0.28,
                "EAR_min": 0.28,
                "blink_count": 0.0,
                "blink_frequency": 15.0,
                "avg_blink_duration": 0.20,
                "max_eye_closure_duration": 0.0,
                "PERCLOS": 0.0,
                "MAR_mean": 0.30,
                "yawn_count": 0.0,
                "head_nod_count": 0.0,
                "head_pose_variation": 2.0,
                "gaze_stability": 1.0,
                "fatigue_trend_val": 0.0
            }

        # Effective duration spanned by current buffer
        span_seconds = max(1.0, self.frame_buffer[-1]["t"] - self.frame_buffer[0]["t"])

        ears = [f["ear"] for f in self.frame_buffer]
        mars = [f["mar"] for f in self.frame_buffer]
        pitches = [f["pitch"] for f in self.frame_buffer]
        gazes = [f["gaze"] for f in self.frame_buffer]
        closed_flags = [f["is_closed"] or (f["ear"] < self.ear_threshold) for f in self.frame_buffer]

        # 1. EAR statistics
        ear_mean = float(np.mean(ears))
        ear_min = float(np.min(ears))

        # 2. PERCLOS: Percentage of Eye Closure over temporal window
        perclos = float(sum(closed_flags) / float(num_frames))

        # 3. Maximum continuous closure duration within buffer
        max_closure_dur = 0.0
        cur_closure = 0.0
        for i in range(len(self.frame_buffer)):
            if closed_flags[i]:
                if i > 0:
                    cur_closure += (self.frame_buffer[i]["t"] - self.frame_buffer[i - 1]["t"])
                else:
                    cur_closure = 0.033
                if cur_closure > max_closure_dur:
                    max_closure_dur = cur_closure
            else:
                cur_closure = 0.0

        # 4. Blink metrics
        blink_count = len(self.blink_events)
        # Convert to blinks per minute
        blink_freq = float((blink_count / span_seconds) * 60.0)

        if blink_count > 0:
            durations = [ev[1] for ev in self.blink_events]
            avg_blink_duration = float(np.mean(durations))
        else:
            avg_blink_duration = 0.20  # Nominal normal duration

        # 5. Mouth & Yawning
        mar_mean = float(np.mean(mars))
        yawn_count = len(self.yawn_events)

        # 6. Head Pose & Nodding
        head_nod_count = len(self.nod_events)
        head_pose_variation = float(np.std(pitches)) if len(pitches) > 1 else 0.0

        # 7. Gaze Stability
        gaze_std = float(np.std(gazes)) if len(gazes) > 1 else 0.08
        if gaze_std < 0.025:
            # Fixed staring
            gaze_stability = max(0.2, gaze_std / 0.025)
        elif gaze_std > 0.20:
            gaze_stability = max(0.2, 1.0 - ((gaze_std - 0.20) / 0.20))
        else:
            gaze_stability = 1.0

        # 8. Fatigue Trend Slope: Linear regression slope of EAR across window
        if num_frames > 5:
            times = np.array([f["t"] - self.frame_buffer[0]["t"] for f in self.frame_buffer])
            ear_arr = np.array(ears)
            # Slope: positive = eyes opening more, negative = eyes progressively drooping
            slope, _ = np.polyfit(times, ear_arr, 1)
            fatigue_trend_val = float(-slope * 10.0)  # Invert so higher = more fatigue
        else:
            fatigue_trend_val = 0.0

        return {
            "EAR_mean": round(ear_mean, 4),
            "EAR_min": round(ear_min, 4),
            "blink_count": float(blink_count),
            "blink_frequency": round(blink_freq, 2),
            "avg_blink_duration": round(avg_blink_duration, 4),
            "max_eye_closure_duration": round(max_closure_dur, 4),
            "PERCLOS": round(perclos, 4),
            "MAR_mean": round(mar_mean, 4),
            "yawn_count": float(yawn_count),
            "head_nod_count": float(head_nod_count),
            "head_pose_variation": round(head_pose_variation, 3),
            "gaze_stability": round(float(gaze_stability), 3),
            "fatigue_trend_val": round(fatigue_trend_val, 4)
        }

    def reset(self):
        """Clears all sliding window buffers."""
        self.frame_buffer.clear()
        self.blink_events.clear()
        self.yawn_events.clear()
        self.nod_events.clear()


if __name__ == "__main__":
    print("[Self-Test] Testing TemporalWindowAggregator...")
    aggregator = TemporalWindowAggregator(window_seconds=10.0, ear_threshold=0.22)
    start_t = 100.0

    # Simulate 300 frames (10 seconds at 30 fps)
    for i in range(300):
        t = start_t + (i / 30.0)
        # Normal open eye 0.28, except a couple of closures
        ear = 0.15 if 100 <= i <= 110 else 0.28
        is_closed = ear < 0.22
        aggregator.add_frame_sample(
            timestamp=t,
            ear=ear,
            is_closed=is_closed,
            mar=0.30,
            pitch=0.0,
            yaw=0.0,
            roll=0.0,
            gaze_ratio=0.50
        )
        if i == 110:
            aggregator.log_blink_event(t, 0.33)

    feats = aggregator.get_temporal_features(start_t + 10.0)
    print(f"Aggregated Temporal Features: {feats}")
    assert feats["PERCLOS"] > 0.0, "PERCLOS should be positive"
    assert feats["blink_count"] == 1.0, "Blink count check failed"
    print("[Self-Test] Temporal aggregator verified successfully.")
