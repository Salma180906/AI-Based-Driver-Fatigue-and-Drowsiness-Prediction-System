"""
================================================================================
AI-Based Driver Fatigue and Drowsiness Prediction System
Module: src/fatigue_score.py
Purpose: Dynamic fatigue score (0-100%), temporal trend progression
         (Stable / Increasing / Decreasing), and 3-level classification
         (Alert / Early Fatigue / Drowsy).
         Member 2: Machine Learning & Application
================================================================================
"""

import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    TREND_STABLE,
    TREND_INCREASING,
    TREND_DECREASING,
    FATIGUE_SCORE_ALERT_MAX,
    FATIGUE_SCORE_EARLY_FATIGUE_MAX,
    STATE_ALERT,
    STATE_EARLY_FATIGUE,
    STATE_DROWSY
)


class FatigueTrendScorer:
    """
    Computes a composite Fatigue Score (0 - 100%) and evaluates the
    temporal trend (Stable, Increasing, Decreasing) over time.

    Academic Design Note:
    Supervised ML models trained on NTHU-DDD output binary predictions
    (Alert vs. Drowsy). The intermediate 'Early Fatigue' state and the
    fatigue score are derived through a documented multi-factor scoring
    layer integrating ML probability, PERCLOS, slow blink rates, yawning,
    and head nodding patterns.
    """

    def __init__(self, history_window: int = 60):
        # Stores recent composite fatigue scores for trend slope estimation
        self.score_history = deque(maxlen=history_window)
        self.timestamps = deque(maxlen=history_window)

    def calculate_score(
        self,
        ml_drowsy_prob: float,
        perclos: float,
        avg_blink_duration: float,
        yawn_count: float,
        head_nod_count: float,
        ear_deviation_from_baseline: float,
        max_eye_closure_duration: float,
        current_time: float
    ) -> Dict[str, Any]:
        """
        Synthesizes a 0 - 100% fatigue score from temporal behavioral indicators.

        Weight distribution:
        - ML Model Prediction Probability: 40%
        - PERCLOS (% eye closure): 25%
        - Slow Blink & Duration Escalation: 15%
        - Yawning Occurrence: 10%
        - Head Nodding Pattern: 10%
        """
        # 1. ML Model Component (0 to 40 pts)
        ml_score = float(np.clip(ml_drowsy_prob * 40.0, 0.0, 40.0))

        # 2. PERCLOS Component (0 to 25 pts)
        # PERCLOS >= 0.25 (25% closed) yields maximum points
        perclos_ratio = min(1.0, perclos / 0.25)
        perclos_score = perclos_ratio * 25.0

        # 3. Blink Duration Component (0 to 15 pts)
        # Normal blink ~0.20s; sluggish blinks > 0.45s indicate fatigue
        blink_dur_excess = max(0.0, avg_blink_duration - 0.20)
        blink_ratio = min(1.0, blink_dur_excess / 0.25)
        # Combine with baseline EAR deviation if eyes are drooping
        ear_droop_factor = max(0.0, ear_deviation_from_baseline)
        blink_score = min(15.0, (blink_ratio * 10.0) + (ear_droop_factor * 10.0))

        # 4. Yawning Component (0 to 10 pts)
        # >= 2 yawns in window yields maximum points
        yawn_score = min(10.0, (yawn_count / 2.0) * 10.0)

        # 5. Head Nodding Component (0 to 10 pts)
        # >= 2 nodding events yields maximum points
        nod_score = min(10.0, (head_nod_count / 2.0) * 10.0)

        # Composite Sum
        composite = ml_score + perclos_score + blink_score + yawn_score + nod_score

        # Emergency override: prolonged continuous eye closure (micro-sleep > 1.2s)
        if max_eye_closure_duration >= 1.2:
            composite = max(composite, 85.0)

        raw_score = float(np.clip(composite, 0.0, 100.0))

        # Record for trend calculation
        self.score_history.append(raw_score)
        self.timestamps.append(current_time)

        # Determine Trend (Stable, Increasing, Decreasing)
        trend = self._determine_trend()

        # Determine 3-Level State
        if raw_score < FATIGUE_SCORE_ALERT_MAX:
            state = STATE_ALERT
        elif raw_score < FATIGUE_SCORE_EARLY_FATIGUE_MAX:
            # If fatigue trend is sharply increasing, can trigger early fatigue notice
            state = STATE_EARLY_FATIGUE
        else:
            state = STATE_DROWSY

        return {
            "fatigue_score": round(raw_score, 1),
            "trend": trend,
            "state": state,
            "components": {
                "ml_score": round(ml_score, 1),
                "perclos_score": round(perclos_score, 1),
                "blink_score": round(blink_score, 1),
                "yawn_score": round(yawn_score, 1),
                "nod_score": round(nod_score, 1)
            }
        }

    def _determine_trend(self) -> str:
        """Estimates trend slope over recent scores."""
        if len(self.score_history) < 10:
            return TREND_STABLE

        scores = list(self.score_history)
        n = len(scores)
        # Simple linear slope across index
        x = np.arange(n)
        slope, _ = np.polyfit(x, scores, 1)

        # Thresholds for trend direction
        if slope > 0.35:
            return TREND_INCREASING
        elif slope < -0.35:
            return TREND_DECREASING
        else:
            return TREND_STABLE

    def reset(self):
        """Clears score history."""
        self.score_history.clear()
        self.timestamps.clear()


if __name__ == "__main__":
    print("[Self-Test] Testing FatigueTrendScorer...")
    scorer = FatigueTrendScorer()

    # 1. Alert profile test
    alert_res = scorer.calculate_score(
        ml_drowsy_prob=0.05,
        perclos=0.02,
        avg_blink_duration=0.18,
        yawn_count=0,
        head_nod_count=0,
        ear_deviation_from_baseline=0.0,
        max_eye_closure_duration=0.15,
        current_time=1.0
    )
    print(f"Alert state result: {alert_res['state']} (Score: {alert_res['fatigue_score']}%)")
    assert alert_res["state"] == STATE_ALERT, "Alert state check failed"

    # 2. Drowsy profile test
    for t in range(2, 20):
        drowsy_res = scorer.calculate_score(
            ml_drowsy_prob=0.85,
            perclos=0.30,
            avg_blink_duration=0.45,
            yawn_count=2,
            head_nod_count=2,
            ear_deviation_from_baseline=0.35,
            max_eye_closure_duration=1.4,
            current_time=float(t)
        )
    print(f"Drowsy state result: {drowsy_res['state']} (Score: {drowsy_res['fatigue_score']}%, Trend: {drowsy_res['trend']})")
    assert drowsy_res["state"] == STATE_DROWSY, "Drowsy state check failed"
    print("[Self-Test] Fatigue trend scorer verified successfully.")
