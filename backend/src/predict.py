"""
================================================================================
AI-Based Driver Fatigue and Drowsiness Prediction System
Module: src/predict.py
Purpose: Real-time inference coordinator integrating face detection,
         instantaneous feature tracking, rolling temporal aggregation,
         personalized baseline calibration, ML model inference, and alert triggers.
         Member 1 & Member 2 Integration
================================================================================
"""

import cv2
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    MODEL_PATH,
    RF_MODEL_PATH,
    SVM_MODEL_PATH,
    LR_MODEL_PATH,
    SCALER_PATH,
    FEATURE_NAMES_PATH,
    FEATURE_COLUMNS,
    STATE_ALERT,
    STATE_DROWSINESS,
    STATE_DISTRACTION,
    STATE_EARLY_FATIGUE,
    STATE_DROWSY,
    COLOR_ALERT,
    COLOR_DROWSINESS,
    COLOR_DISTRACTION,
    COLOR_EARLY_FATIGUE,
    COLOR_DROWSY,
    DEFAULT_EAR_THRESHOLD
)
from src.facial_landmarks import FaceMeshDetector
from src.face_detection import assess_lighting_condition, enhance_frame_if_needed
from src.eye_features import calculate_ear, extract_eye_landmarks, EyeBlinkTracker
from src.mouth_features import calculate_mar, YawnTracker
from src.head_pose import estimate_head_pose, draw_head_pose_axes, HeadNodTracker
from src.gaze_features import estimate_gaze, GazeStabilityTracker
from src.temporal_features import TemporalWindowAggregator
from src.baseline import PersonalizedBaseline
from src.fatigue_score import FatigueTrendScorer
from src.alert import AlertManager


class RealTimePredictor:
    """
    End-to-end real-time driver monitoring inference engine.
    Processes live camera frames and outputs comprehensive fatigue diagnostics.
    Supports dynamic multi-model selection (Random Forest, SVM, Logistic Regression)
    and full animated simulation testing.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        scaler_path: Path = SCALER_PATH,
        feature_names_path: Path = FEATURE_NAMES_PATH,
        enable_audio_alert: bool = True
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.feature_names_path = Path(feature_names_path)

        # Computer Vision Modules
        self.detector = FaceMeshDetector()
        self.blink_tracker = EyeBlinkTracker()
        self.yawn_tracker = YawnTracker()
        self.nod_tracker = HeadNodTracker()
        self.gaze_tracker = GazeStabilityTracker()
        self.aggregator = TemporalWindowAggregator()

        # Calibration & Scoring Modules
        self.baseline = PersonalizedBaseline()
        self.trend_scorer = FatigueTrendScorer()
        self.alert_manager = AlertManager() if enable_audio_alert else None

        # Multi-Model Store
        self.models: Dict[str, Any] = {}
        self.active_model_name: str = "Random Forest"
        self.model = None
        self.scaler = None
        self.feature_names = FEATURE_COLUMNS
        self._load_ml_assets()

        # Telemetry state
        self.frame_count = 0
        self.last_prediction_time = time.time()
        self.consecutive_no_face_frames = 0
        self.sim_phase = 0.0
        self.distraction_start_time = None
        self.distraction_duration = 0.0
        self.last_nod_time = None

    def reset(self):
        """Resets all temporal aggregators, tracking queues, and state counters for a fresh session."""
        threshold = self.baseline.baseline_dynamic_ear_threshold if (self.baseline and (self.baseline.is_calibrated or self.baseline.is_auto_calibrated)) else DEFAULT_EAR_THRESHOLD
        self.baseline.reset()
        self.blink_tracker = EyeBlinkTracker(ear_threshold=threshold)
        self.yawn_tracker = YawnTracker()
        self.nod_tracker = HeadNodTracker()
        self.gaze_tracker = GazeStabilityTracker()
        self.aggregator = TemporalWindowAggregator(ear_threshold=threshold)
        self.trend_scorer = FatigueTrendScorer()
        self.frame_count = 0
        self.last_prediction_time = time.time()
        self.distraction_start_time = None
        self.distraction_duration = 0.0
        self.last_nod_time = None
        self.consecutive_no_face_frames = 0
        print("[Predictor] Driver state and temporal aggregators successfully reset.")

    def _load_ml_assets(self):
        """Loads all candidate models, primary model, scaler, and feature definitions."""
        try:
            if self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
            if self.feature_names_path.exists():
                self.feature_names = joblib.load(self.feature_names_path)

            # Load individual model artifacts
            model_file_map = {
                "Random Forest": RF_MODEL_PATH if RF_MODEL_PATH.exists() else self.model_path,
                "SVM (RBF Kernel)": SVM_MODEL_PATH,
                "Logistic Regression": LR_MODEL_PATH
            }

            for name, path in model_file_map.items():
                if path.exists():
                    try:
                        self.models[name] = joblib.load(path)
                    except Exception as e:
                        print(f"[Predictor] Could not load '{name}' from '{path}': {e}")

            # Fallback primary model
            if self.model_path.exists() and "Random Forest" not in self.models:
                self.models["Random Forest"] = joblib.load(self.model_path)

            if "Random Forest" in self.models:
                self.active_model_name = "Random Forest"
                self.model = self.models["Random Forest"]
            elif self.models:
                self.active_model_name = list(self.models.keys())[0]
                self.model = self.models[self.active_model_name]

            print(f"[Predictor] Loaded {len(self.models)} ML model(s): {list(self.models.keys())}. Active: '{self.active_model_name}'.")
        except Exception as e:
            print(f"[Predictor] Error loading ML assets: {e}")

    def set_active_model(self, model_name: str) -> bool:
        """Dynamically switches active ML model for real-time inference."""
        if model_name in self.models:
            self.active_model_name = model_name
            self.model = self.models[model_name]
            return True
        return False

    def start_calibration(self):
        """Triggers personalized driver baseline calibration."""
        self.baseline.start_calibration()

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        draw_annotations: bool = True,
        timestamp: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Main processing method called for every video frame.

        Returns:
            annotated_frame: np.ndarray
            telemetry: Dict containing all live facial, temporal, and prediction stats.
        """
        self.frame_count += 1
        now = timestamp if timestamp is not None else time.time()
        annotated = frame_bgr.copy() if frame_bgr is not None else None

        if frame_bgr is None or frame_bgr.size == 0:
            return annotated, self._empty_telemetry("No frame received")

        # 1. Lighting assessment & optional adaptive enhancement
        lighting = assess_lighting_condition(frame_bgr)
        processed_input = enhance_frame_if_needed(frame_bgr)

        # 2. Facial Landmark Detection (MediaPipe)
        landmarks_px = self.detector.process_frame(processed_input)

        if landmarks_px is None:
            self.consecutive_no_face_frames += 1
            status_note = f"No face detected ({self.consecutive_no_face_frames} frames)"
            telemetry = self._empty_telemetry(status_note)
            telemetry["lighting"] = lighting
            return annotated, telemetry

        self.consecutive_no_face_frames = 0

        # 3. Feature Extraction (Member 1)
        # Eyes & EAR
        l_eye, r_eye = extract_eye_landmarks(landmarks_px)
        ear_l = calculate_ear(l_eye)
        ear_r = calculate_ear(r_eye)
        ear = float((ear_l + ear_r) / 2.0)

        # Mouth & MAR
        mar = calculate_mar(landmarks_px)

        # Head Pose & Nodding
        pitch, yaw, roll, rvec, tvec = estimate_head_pose(landmarks_px, frame_bgr.shape[:2])

        # Gaze Estimation & Stability
        gaze_ratio, gaze_direction = estimate_gaze(landmarks_px)
        gaze_info = self.gaze_tracker.update(gaze_ratio)

        # Continuous / Rapid Auto-Baseline Calibration
        self.baseline.update_auto_baseline(ear=ear, mar=mar, pitch=pitch, yaw=yaw)

        # Dynamic physiological thresholds based on driver's personalized baseline
        effective_ear_thresh = self.baseline.baseline_dynamic_ear_threshold
        self.blink_tracker.set_ear_threshold(effective_ear_thresh)
        self.aggregator.set_ear_threshold(effective_ear_thresh)

        effective_mar_thresh = max(0.36, min(0.55, self.baseline.baseline_mar * 1.50))
        self.yawn_tracker.set_mar_threshold(effective_mar_thresh)

        blink_info = self.blink_tracker.update(ear, now)
        if blink_info["blink_occurred"]:
            self.aggregator.log_blink_event(now, blink_info["last_blink_duration"])

        yawn_info = self.yawn_tracker.update(mar, now)
        if yawn_info["yawn_completed"]:
            self.aggregator.log_yawn_event(now, yawn_info["current_yawn_duration"])

        nod_info = self.nod_tracker.update(pitch, yaw, roll, now, baseline_pitch=self.baseline.baseline_pitch)
        if nod_info["nod_detected"]:
            self.aggregator.log_nod_event(now)

        # 4. Temporal Aggregation (Member 1 & 2)
        self.aggregator.add_frame_sample(
            timestamp=now,
            ear=ear,
            is_closed=blink_info["is_eye_closed"],
            mar=mar,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            gaze_ratio=gaze_ratio
        )
        temporal_feats = self.aggregator.get_temporal_features(now)

        # 5. Personalized Baseline Calibration (Member 2)
        calib_status = self.baseline.update_calibration(
            ear=ear,
            mar=mar,
            pitch=pitch,
            gaze_ratio=gaze_ratio,
            yaw=yaw,
            blink_duration=blink_info["last_blink_duration"] if blink_info["blink_occurred"] else None
        )

        deviations = self.baseline.compute_deviations(
            current_ear_mean=temporal_feats["EAR_mean"],
            current_avg_blink_duration=temporal_feats["avg_blink_duration"],
            current_pitch=pitch,
            current_yaw=yaw
        )

        # 6. ML Model Inference
        feature_vector = {
            "EAR_mean": float(temporal_feats["EAR_mean"]),
            "EAR_min": float(temporal_feats["EAR_min"]),
            "blink_count": float(temporal_feats["blink_count"]),
            "blink_frequency": float(temporal_feats["blink_frequency"]),
            "avg_blink_duration": float(temporal_feats["avg_blink_duration"]),
            "max_eye_closure_duration": float(temporal_feats["max_eye_closure_duration"]),
            "PERCLOS": float(temporal_feats["PERCLOS"]),
            "MAR_mean": float(temporal_feats["MAR_mean"]),
            "yawn_count": float(temporal_feats["yawn_count"]),
            "head_nod_count": float(temporal_feats["head_nod_count"]),
            "head_pose_variation": float(temporal_feats["head_pose_variation"]),
            "gaze_stability": float(temporal_feats["gaze_stability"]),
            "EAR_deviation_from_baseline": float(deviations.get("EAR_deviation_from_baseline", 0.0)),
            "blink_duration_ratio": float(deviations.get("blink_duration_ratio", 1.0)),
            "fatigue_trend_val": float(temporal_feats["fatigue_trend_val"])
        }

        # Distraction Tracking (Rule 2: Head turned left or right more than 5 seconds)
        delta_yaw = yaw - self.baseline.baseline_yaw
        is_head_turned = abs(delta_yaw) >= 15.0 or abs(yaw) >= 15.0

        if is_head_turned:
            if self.distraction_start_time is None:
                self.distraction_start_time = now
            self.distraction_duration = now - self.distraction_start_time
        else:
            self.distraction_start_time = None
            self.distraction_duration = 0.0

        is_distracted = self.distraction_duration >= 5.0

        # ML Model Inference - Executed for academic analysis & telemetry
        rf_prob = 0.0
        svm_prob = 0.0
        lr_prob = 0.0

        if self.scaler is not None and self.models:
            try:
                feat_df = pd.DataFrame([feature_vector])[self.feature_names]
                scaled_vec = self.scaler.transform(feat_df)
                if "Random Forest" in self.models:
                    rf_prob = float(self.models["Random Forest"].predict_proba(scaled_vec)[0][1])
                if "SVM (RBF Kernel)" in self.models:
                    svm_prob = float(self.models["SVM (RBF Kernel)"].predict_proba(scaled_vec)[0][1])
                if "Logistic Regression" in self.models:
                    lr_prob = float(self.models["Logistic Regression"].predict_proba(scaled_vec)[0][1])
            except Exception:
                pass

        ensemble_drowsy_prob = float(0.50 * rf_prob + 0.30 * svm_prob + 0.20 * lr_prob)

        model_probs = {
            "Random Forest": round(rf_prob, 3),
            "SVM (RBF Kernel)": round(svm_prob, 3),
            "Logistic Regression": round(lr_prob, 3),
            "Ensemble Consensus": round(ensemble_drowsy_prob, 3)
        }

        # Fatigue Trend & Composite Score Calculation
        score_info = self.trend_scorer.calculate_score(
            ml_drowsy_prob=ensemble_drowsy_prob,
            perclos=temporal_feats["PERCLOS"],
            avg_blink_duration=temporal_feats["avg_blink_duration"],
            yawn_count=temporal_feats["yawn_count"],
            head_nod_count=temporal_feats["head_nod_count"],
            ear_deviation_from_baseline=deviations["EAR_deviation_from_baseline"],
            max_eye_closure_duration=temporal_feats["max_eye_closure_duration"],
            current_time=now
        )

        fatigue_score = score_info["fatigue_score"]

        # Accurate State Classification based on user's exact rules:
        # Rule 3: Drowsiness -> Eyes closed (>=1.0s) OR Yawning (>=1.0s) OR Head down (>=1.2s) OR Nodding
        is_eyes_closed_drowsy = bool(blink_info["is_eye_closed"] and blink_info["current_closure_duration"] >= 1.0)
        is_yawning_drowsy = bool(yawn_info.get("is_yawning", False) and mar >= 0.42)
        
        delta_pitch = pitch - self.baseline.baseline_pitch
        is_head_down_drowsy = bool((pitch <= -16.0 or delta_pitch <= -15.0) and nod_info.get("head_down_duration", 0) >= 1.2)
        if nod_info.get("nod_detected", False) or nod_info.get("is_repeated_nodding", False):
            self.last_nod_time = now
        is_nodding_drowsy = bool(self.last_nod_time is not None and (now - self.last_nod_time <= 2.5))

        voice_alert_type = None
        buzzer_reason = None
        buzzer_reason_ta = None
        buzzer_reason_ta_phonetic = None

        if is_eyes_closed_drowsy:
            current_state = STATE_DROWSINESS
            voice_alert_type = "eyes_closed"
            buzzer_reason = "Open your eyes! Open your eyes!"
            buzzer_reason_ta = "கண்களை திறக்கவும்! கண்களை திறக்கவும்!"
            buzzer_reason_ta_phonetic = "Kankalai thiravungal! Kankalai thiravungal!"
            if self.alert_manager:
                self.alert_manager.trigger_audio_alarm()
        elif is_yawning_drowsy:
            current_state = STATE_DROWSINESS
            voice_alert_type = "yawning"
            buzzer_reason = "Don't yawn, take rest! Don't yawn, take rest!"
            buzzer_reason_ta = "கொட்டாவி விடாதீர்கள், ஓய்வெடுங்கள்! கொட்டாவி விடாதீர்கள், ஓய்வெடுங்கள்!"
            buzzer_reason_ta_phonetic = "Kottaavi vidaatheergal! Kottaavi vidaatheergal!"
            if self.alert_manager:
                self.alert_manager.trigger_audio_alarm()
        elif is_head_down_drowsy or is_nodding_drowsy:
            current_state = STATE_DROWSINESS
            voice_alert_type = "head_nodding"
            buzzer_reason = "You are asleep, take rest! You are asleep, take rest!"
            buzzer_reason_ta = "தூங்குகிறீர்கள், ஓய்வெடுங்கள்! தூங்குகிறீர்கள், ஓய்வெடுங்கள்!"
            buzzer_reason_ta_phonetic = "Thoongugireergal, oyvedungal! Thoongugireergal!"
            if self.alert_manager:
                self.alert_manager.trigger_audio_alarm()
        elif is_distracted:
            current_state = STATE_DISTRACTION
            voice_alert_type = "distraction"
            buzzer_reason = "Turn straight, focus on driving! Turn straight, focus on driving!"
            buzzer_reason_ta = "நேராகப் பாருங்கள், கவனமாக ஓட்டுங்கள்! நேராகப் பாருங்கள், கவனமாக ஓட்டுங்கள்!"
            buzzer_reason_ta_phonetic = "Neeraaga paarungal, kavanamaaga oottungal!"
            if self.alert_manager:
                self.alert_manager.trigger_audio_alarm()
        else:
            # Rule 1: Normal if eyes are open!
            current_state = STATE_ALERT
            voice_alert_type = None
            buzzer_reason = None
            buzzer_reason_ta = None
            buzzer_reason_ta_phonetic = None

        # 9. Visualization Annotations
        if draw_annotations:
            annotated = self.detector.draw_annotations(annotated, landmarks_px)
            annotated = draw_head_pose_axes(annotated, rvec, tvec, frame_bgr.shape[:2])

        # Assemble telemetry package
        telemetry = {
            "face_detected": True,
            "lighting": lighting,
            # Instantaneous metrics
            "ear": round(ear, 3),
            "mar": round(mar, 3),
            "pitch": round(pitch, 1),
            "yaw": round(yaw, 1),
            "roll": round(roll, 1),
            "gaze_ratio": round(gaze_ratio, 3),
            "gaze_direction": gaze_direction,
            "is_eye_closed": blink_info["is_eye_closed"],
            "is_yawning": yawn_info["is_yawning"],
            "is_head_down": nod_info["is_head_down"],
            # Temporal Window metrics (15s)
            "EAR_mean": temporal_feats["EAR_mean"],
            "EAR_min": temporal_feats["EAR_min"],
            "PERCLOS": temporal_feats["PERCLOS"],
            "blink_count": int(temporal_feats["blink_count"]),
            "blink_frequency": temporal_feats["blink_frequency"],
            "avg_blink_duration": temporal_feats["avg_blink_duration"],
            "max_eye_closure_duration": temporal_feats["max_eye_closure_duration"],
            "MAR_mean": temporal_feats["MAR_mean"],
            "yawn_count": int(temporal_feats["yawn_count"]),
            "head_nod_count": int(temporal_feats["head_nod_count"]),
            "head_pose_variation": temporal_feats["head_pose_variation"],
            "gaze_stability": temporal_feats["gaze_stability"],
            # Baseline calibration
            "calibration": calib_status,
            "baseline_summary": self.baseline.get_baseline_summary(),
            "deviations": deviations,
            # Machine Learning & Multi-Model Progression
            "models": model_probs,
            "model_probs": model_probs,
            "ml_drowsy_prob": round(ensemble_drowsy_prob, 3),
            "fatigue_score": fatigue_score,
            "fatigue_trend": score_info["trend"],
            "state": current_state,
            "voice_alert_type": voice_alert_type,
            "buzzer": bool(voice_alert_type is not None),
            "buzzer_reason": buzzer_reason,
            "buzzer_reason_ta": buzzer_reason_ta,
            "buzzer_reason_ta_phonetic": buzzer_reason_ta_phonetic,
            "is_drowsy": bool(current_state == STATE_DROWSINESS),
            "is_distracted": bool(current_state == STATE_DISTRACTION),
            "is_head_turned": bool(is_head_turned),
            "distraction_duration": round(self.distraction_duration, 1),
            "components": score_info["components"],
            "landmarks_px": landmarks_px.tolist() if landmarks_px is not None else []
        }

        return annotated, telemetry

    def _empty_telemetry(self, status_msg: str) -> Dict[str, Any]:
        """Provides default values when no face or frame is available."""
        return {
            "face_detected": False,
            "status_message": status_msg,
            "ear": 0.0,
            "mar": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "gaze_ratio": 0.5,
            "gaze_direction": "Unknown",
            "is_eye_closed": False,
            "is_yawning": False,
            "is_head_down": False,
            "EAR_mean": 0.0,
            "EAR_min": 0.0,
            "PERCLOS": 0.0,
            "blink_count": 0,
            "blink_frequency": 0.0,
            "avg_blink_duration": 0.0,
            "max_eye_closure_duration": 0.0,
            "MAR_mean": 0.0,
            "yawn_count": 0,
            "head_nod_count": 0,
            "head_pose_variation": 0.0,
            "gaze_stability": 1.0,
            "calibration": {"is_calibrating": False, "is_calibrated": False, "progress": 0.0},
            "baseline_summary": self.baseline.get_baseline_summary(),
            "deviations": {"EAR_deviation_from_baseline": 0.0, "blink_duration_ratio": 1.0, "pitch_deviation": 0.0},
            "active_model": "All 3 Models (Ensemble)",
            "models": {"Random Forest": 0.0, "SVM (RBF Kernel)": 0.0, "Logistic Regression": 0.0, "Ensemble Consensus": 0.0},
            "model_probs": {},
            "ml_drowsy_prob": 0.0,
            "fatigue_score": 0.0,
            "fatigue_trend": "Stable",
            "state": STATE_ALERT,
            "voice_alert_type": None,
            "buzzer": False,
            "buzzer_reason": None,
            "buzzer_reason_ta": None,
            "buzzer_reason_ta_phonetic": None,
            "is_drowsy": False,
            "is_distracted": False,
            "is_head_turned": False,
            "distraction_duration": 0.0,
            "components": {},
            "landmarks_px": []
        }

    def process_simulated_frame(
        self,
        scenario: str = "Normal Alert Driving",
        frame_idx: int = 0,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Synthesizes high-fidelity simulated driver behavioral telemetry and an
        animated automotive HUD frame representing the driver state.
        Allows 100% deterministic testing and demonstration of fatigue progression,
        trend tracking, model evaluation, and audio alerts without requiring a physical webcam.
        """
        now = time.time()
        self.frame_count += 1

        # 1. Determine physiological feature values based on active scenario
        if custom_params:
            ear = float(custom_params.get("ear", 0.28))
            mar = float(custom_params.get("mar", 0.26))
            pitch = float(custom_params.get("pitch", 0.0))
            yaw = float(custom_params.get("yaw", 0.0))
            gaze_ratio = float(custom_params.get("gaze_ratio", 0.5))
            is_closed = ear < self.blink_tracker.ear_threshold
            blink_dur = float(custom_params.get("blink_duration", 0.20))
            is_yawn = mar > self.yawn_tracker.mar_threshold
            is_nod = pitch < HEAD_PITCH_NOD_THRESHOLD
        elif scenario == "Normal Alert Driving":
            # Alert: natural blink cycle, high EAR, upright head, centered gaze
            is_blink_moment = (frame_idx % 75) in [0, 1, 2, 3]  # ~0.13s blink
            ear = 0.08 if is_blink_moment else float(0.29 + 0.02 * np.sin(frame_idx * 0.1))
            mar = float(0.24 + 0.03 * np.cos(frame_idx * 0.05))
            pitch = float(1.0 * np.sin(frame_idx * 0.05))
            yaw = float(2.0 * np.cos(frame_idx * 0.03))
            gaze_ratio = float(0.50 + 0.04 * np.sin(frame_idx * 0.08))
            is_closed = is_blink_moment
            blink_dur = 0.18
            is_yawn = False
            is_nod = False
        elif scenario == "Early Fatigue Warning":
            # Early Fatigue: slightly drooping eyes, sluggish blinks (0.35s), slight pitch droop
            is_blink_moment = (frame_idx % 50) in list(range(10))  # ~0.33s blink
            ear = 0.08 if is_blink_moment else float(0.21 + 0.02 * np.sin(frame_idx * 0.08))
            mar = float(0.48) if (frame_idx % 120) in list(range(20)) else float(0.32 + 0.04 * np.sin(frame_idx * 0.04))
            pitch = float(-6.0 + 2.0 * np.sin(frame_idx * 0.06))
            yaw = float(4.0 * np.sin(frame_idx * 0.04))
            gaze_ratio = float(0.48 + 0.12 * np.sin(frame_idx * 0.05))
            is_closed = is_blink_moment
            blink_dur = 0.36
            is_yawn = mar > 0.45
            is_nod = False
        elif scenario == "Severe Drowsiness & Microsleep":
            # Drowsy: Prolonged eye closure, micro-sleep, head downward nod, yawning
            is_closed = True
            ear = float(0.08 + 0.01 * np.random.rand())
            mar = float(0.68) if (frame_idx % 90) < 45 else float(0.38 + 0.05 * np.sin(frame_idx * 0.1))
            pitch = float(-18.0 + 2.5 * np.sin(frame_idx * 0.1))
            yaw = float(1.0 * np.sin(frame_idx * 0.05))
            gaze_ratio = 0.40
            blink_dur = 1.45
            is_yawn = mar > 0.55
            is_nod = True
        elif scenario == "Continuous Yawning Episode":
            # Yawning: high MAR lasting > 1.5s
            is_blink_moment = (frame_idx % 60) in [0, 1, 2]
            ear = 0.10 if is_blink_moment else float(0.24)
            mar = float(0.72 + 0.05 * np.sin(frame_idx * 0.15))
            pitch = float(-4.0 + 2.0 * np.sin(frame_idx * 0.08))
            yaw = 0.0
            gaze_ratio = 0.50
            is_closed = is_blink_moment
            blink_dur = 0.22
            is_yawn = True
            is_nod = False
        elif scenario == "Head Nodding / Saccadic Cycles":
            # Repeated head nod cycle: pitch drops from 0 down to -22 and recovers
            nod_phase = (frame_idx % 60) / 60.0
            pitch = float(-22.0 * np.sin(nod_phase * np.pi))
            ear = float(0.18 + 0.04 * np.sin(frame_idx * 0.1))
            mar = float(0.30)
            yaw = float(3.0 * np.cos(frame_idx * 0.1))
            gaze_ratio = 0.50
            is_closed = (frame_idx % 45) in list(range(8))
            blink_dur = 0.32
            is_yawn = False
            is_nod = pitch < -14.0
        else:
            ear = 0.28
            mar = 0.26
            pitch = 0.0
            yaw = 0.0
            gaze_ratio = 0.50
            is_closed = False
            blink_dur = 0.18
            is_yawn = False
            is_nod = False

        # Direction string
        if gaze_ratio < 0.38:
            gaze_dir = "Looking Right"
        elif gaze_ratio > 0.62:
            gaze_dir = "Looking Left"
        else:
            gaze_dir = "Center / Forward"

        # Log events into temporal window aggregator
        if is_closed:
            if not hasattr(self, "_sim_closure_start") or self._sim_closure_start is None:
                self._sim_closure_start = now
        else:
            if hasattr(self, "_sim_closure_start") and self._sim_closure_start is not None:
                dur = max(0.10, now - self._sim_closure_start)
                self.aggregator.log_blink_event(now, dur)
                self._sim_closure_start = None

        if is_yawn:
            self.aggregator.log_yawn_event(now, 1.8)
        if is_nod:
            self.aggregator.log_nod_event(now)

        self.aggregator.add_frame_sample(
            timestamp=now,
            ear=ear,
            is_closed=is_closed,
            mar=mar,
            pitch=pitch,
            yaw=yaw,
            roll=0.0,
            gaze_ratio=gaze_ratio
        )

        temporal_feats = self.aggregator.get_temporal_features(now)

        # Baseline calibration
        calib_status = self.baseline.update_calibration(
            ear=ear,
            mar=mar,
            pitch=pitch,
            gaze_ratio=gaze_ratio
        )
        deviations = self.baseline.compute_deviations(
            current_ear_mean=temporal_feats["EAR_mean"],
            current_avg_blink_duration=temporal_feats["avg_blink_duration"],
            current_pitch=pitch
        )

        # In microsleep scenario, force temporal features to reflect deep fatigue
        if scenario == "Severe Drowsiness & Microsleep":
            temporal_feats["max_eye_closure_duration"] = max(1.40, temporal_feats["max_eye_closure_duration"])
            temporal_feats["PERCLOS"] = max(0.35, temporal_feats["PERCLOS"])

        # Feature vector for ML models
        feature_vector = {
            "EAR_mean": temporal_feats["EAR_mean"],
            "EAR_min": temporal_feats["EAR_min"],
            "blink_count": temporal_feats["blink_count"],
            "blink_frequency": temporal_feats["blink_frequency"],
            "avg_blink_duration": temporal_feats["avg_blink_duration"],
            "max_eye_closure_duration": temporal_feats["max_eye_closure_duration"],
            "PERCLOS": temporal_feats["PERCLOS"],
            "MAR_mean": temporal_feats["MAR_mean"],
            "yawn_count": temporal_feats["yawn_count"],
            "head_nod_count": temporal_feats["head_nod_count"],
            "head_pose_variation": temporal_feats["head_pose_variation"],
            "gaze_stability": temporal_feats["gaze_stability"],
            "EAR_deviation_from_baseline": deviations["EAR_deviation_from_baseline"],
            "blink_duration_ratio": deviations["blink_duration_ratio"],
            "fatigue_trend_val": temporal_feats["fatigue_trend_val"]
        }

        # Multi-model inference
        model_probs = {}
        ml_prob_drowsy = 0.0
        if self.scaler is not None and self.models:
            try:
                feat_df = pd.DataFrame([feature_vector])[self.feature_names]
                scaled_vec = self.scaler.transform(feat_df)
                for m_name, m_inst in self.models.items():
                    try:
                        if hasattr(m_inst, "predict_proba"):
                            p = float(m_inst.predict_proba(scaled_vec)[0][1])
                        else:
                            p = float(m_inst.predict(scaled_vec)[0])
                        model_probs[m_name] = round(p, 3)
                    except Exception:
                        pass
            except Exception:
                pass

        if self.active_model_name in model_probs:
            ml_prob_drowsy = model_probs[self.active_model_name]
        else:
            ml_prob_drowsy = float(np.clip(temporal_feats["PERCLOS"] / 0.25, 0.0, 1.0))

        # Fatigue Score & Trend Calculation
        score_info = self.trend_scorer.calculate_score(
            ml_drowsy_prob=ml_prob_drowsy,
            perclos=temporal_feats["PERCLOS"],
            avg_blink_duration=temporal_feats["avg_blink_duration"],
            yawn_count=temporal_feats["yawn_count"],
            head_nod_count=temporal_feats["head_nod_count"],
            ear_deviation_from_baseline=deviations["EAR_deviation_from_baseline"],
            max_eye_closure_duration=temporal_feats["max_eye_closure_duration"],
            current_time=now
        )

        # Determine state according to simulated scenario and user rules
        if scenario == "Normal Alert Driving":
            current_state = STATE_ALERT
            voice_alert_type = None
            buzzer_reason = None
            buzzer_reason_ta = None
            buzzer_reason_ta_phonetic = None
        elif scenario == "Severe Drowsiness & Microsleep":
            current_state = STATE_DROWSINESS
            voice_alert_type = "eyes_closed"
            buzzer_reason = "Open your eyes! Open your eyes!"
            buzzer_reason_ta = "கண்களை திறக்கவும்! கண்களை திறக்கவும்!"
            buzzer_reason_ta_phonetic = "Kankalai thiravungal! Kankalai thiravungal!"
        elif scenario == "Continuous Yawning Episode":
            current_state = STATE_DROWSINESS
            voice_alert_type = "yawning"
            buzzer_reason = "Don't yawn, take rest! Don't yawn, take rest!"
            buzzer_reason_ta = "கொட்டாவி விடாதீர்கள், ஓய்வெடுங்கள்! கொட்டாவி விடாதீர்கள், ஓய்வெடுங்கள்!"
            buzzer_reason_ta_phonetic = "Kottaavi vidaatheergal! Kottaavi vidaatheergal!"
        elif scenario == "Head Nodding / Saccadic Cycles":
            current_state = STATE_DROWSINESS
            voice_alert_type = "head_nodding"
            buzzer_reason = "You are asleep, take rest! You are asleep, take rest!"
            buzzer_reason_ta = "தூங்குகிறீர்கள், ஓய்வெடுங்கள்! தூங்குகிறீர்கள், ஓய்வெடுங்கள்!"
            buzzer_reason_ta_phonetic = "Thoongugireergal, oyvedungal! Thoongugireergal!"
        elif abs(yaw) >= 15.0:
            current_state = STATE_DISTRACTION
            voice_alert_type = "distraction"
            buzzer_reason = "Turn straight, focus on driving! Turn straight, focus on driving!"
            buzzer_reason_ta = "நேராகப் பாருங்கள், கவனமாக ஓட்டுங்கள்! நேராகப் பாருங்கள், கவனமாக ஓட்டுங்கள்!"
            buzzer_reason_ta_phonetic = "Neeraaga paarungal, kavanamaaga oottungal!"
        else:
            current_state = STATE_ALERT
            voice_alert_type = None
            buzzer_reason = None
            buzzer_reason_ta = None
            buzzer_reason_ta_phonetic = None

        # 2. Render Animated Futuristic Cockpit HUD Frame (640x480)
        h, w = 480, 640
        frame = np.full((h, w, 3), 18, dtype=np.uint8)  # Deep dark navy slate canvas

        # Color palette depending on state
        if current_state == STATE_ALERT:
            hud_theme_color = (118, 230, 0)     # Neon Emerald Green
            status_text = "STATUS: ALERT (NORMAL)"
        elif current_state == STATE_DISTRACTION:
            hud_theme_color = (0, 171, 255)     # Glowing Amber
            status_text = "STATUS: DISTRACTION (HEAD TURNED)"
        else:
            hud_theme_color = (68, 23, 255)     # Critical Crimson Red
            status_text = "CRITICAL: DROWSY WARNING"

        # Background grid & cockpit borders
        for gy in range(0, h, 60):
            cv2.line(frame, (0, gy), (w, gy), (28, 38, 52), 1)
        for gx in range(0, w, 60):
            cv2.line(frame, (gx, 0), (gx, h), (28, 38, 52), 1)

        # Cockpit outer frame accents
        cv2.rectangle(frame, (10, 10), (w - 10, h - 10), hud_theme_color, 2)
        cv2.line(frame, (10, 45), (w - 10, 45), hud_theme_color, 1)
        cv2.line(frame, (10, h - 45), (w - 10, h - 45), (50, 65, 85), 1)

        # Header Title
        cv2.putText(frame, "AI COCKPIT HUD • SIMULATED DRIVER STREAM", (25, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 245, 255), 2)
        cv2.putText(frame, f"[{scenario.upper()}]", (w - 280, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, hud_theme_color, 1)

        # Dynamic Head Center (shifts with pitch & yaw)
        head_cx = int(w // 2 + yaw * 3.0)
        head_cy = int(h // 2 - 20 + pitch * 3.5)

        # Draw Head / Face Silhouette
        cv2.ellipse(frame, (head_cx, head_cy), (110, 140), 0, 0, 360, (70, 90, 120), 2)
        cv2.ellipse(frame, (head_cx, head_cy), (115, 145), 0, 0, 360, hud_theme_color, 1)

        # Pitch attitude horizon line
        horizon_y = int(head_cy + pitch * 2.0)
        cv2.line(frame, (head_cx - 70, horizon_y), (head_cx + 70, horizon_y), (120, 150, 180), 1)
        cv2.circle(frame, (head_cx, horizon_y), 3, hud_theme_color, -1)

        # Draw Eyes
        eye_y = head_cy - 25
        eye_lx = head_cx - 45
        eye_rx = head_cx + 45
        iris_offset_x = int((gaze_ratio - 0.5) * 20)

        # Left Eye
        if is_closed:
            cv2.line(frame, (eye_lx - 22, eye_y), (eye_lx + 22, eye_y), hud_theme_color, 3)
            cv2.putText(frame, "CLOSED", (eye_lx - 25, eye_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        else:
            eye_h = int(max(4, ear * 50))
            cv2.ellipse(frame, (eye_lx, eye_y), (22, eye_h), 0, 0, 360, (230, 240, 255), 2)
            cv2.circle(frame, (eye_lx + iris_offset_x, eye_y), 6, hud_theme_color, -1)
            cv2.circle(frame, (eye_lx + iris_offset_x, eye_y), 2, (255, 255, 255), -1)

        # Right Eye
        if is_closed:
            cv2.line(frame, (eye_rx - 22, eye_y), (eye_rx + 22, eye_y), hud_theme_color, 3)
            cv2.putText(frame, "CLOSED", (eye_rx - 25, eye_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        else:
            eye_h = int(max(4, ear * 50))
            cv2.ellipse(frame, (eye_rx, eye_y), (22, eye_h), 0, 0, 360, (230, 240, 255), 2)
            cv2.circle(frame, (eye_rx + iris_offset_x, eye_y), 6, hud_theme_color, -1)
            cv2.circle(frame, (eye_rx + iris_offset_x, eye_y), 2, (255, 255, 255), -1)

        # Draw Nose
        cv2.line(frame, (head_cx, head_cy - 5), (head_cx - 4, head_cy + 18), (100, 130, 160), 2)
        cv2.line(frame, (head_cx - 4, head_cy + 18), (head_cx + 8, head_cy + 18), (100, 130, 160), 2)

        # Draw Mouth
        mouth_y = head_cy + 55
        if mar > 0.50:
            mouth_h = int(mar * 40)
            cv2.ellipse(frame, (head_cx, mouth_y), (28, mouth_h), 0, 0, 360, hud_theme_color, -1)
            cv2.ellipse(frame, (head_cx, mouth_y), (28, mouth_h), 0, 0, 360, (255, 255, 255), 2)
            cv2.putText(frame, "YAWNING", (head_cx - 32, mouth_y + mouth_h + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 140, 255), 1)
        else:
            cv2.ellipse(frame, (head_cx, mouth_y), (28, 8), 0, 0, 180, (200, 220, 245), 2)

        # Animated scan line
        scan_y = int((frame_idx * 4) % (h - 80) + 40)
        cv2.line(frame, (15, scan_y), (w - 15, scan_y), (hud_theme_color[0] // 3, hud_theme_color[1] // 3, hud_theme_color[2] // 3), 1)

        # HUD Corner Indicators
        cv2.putText(frame, status_text, (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hud_theme_color, 2)
        cv2.putText(frame, f"FATIGUE: {score_info['fatigue_score']}% [{score_info['trend']}]", (25, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 240, 255), 2)

        # Bottom Telemetry Bar
        footer_text = f"EAR: {ear:.2f} | MAR: {mar:.2f} | PERCLOS: {temporal_feats['PERCLOS'] * 100:.1f}% | PITCH: {pitch:.1f}deg | GAZE: {gaze_dir}"
        cv2.putText(frame, footer_text, (25, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 205, 230), 1)

        telemetry = {
            "face_detected": True,
            "lighting": {"mean_brightness": 120.0, "contrast_std": 45.0, "lighting_status": "Good (Simulated)"},
            "ear": round(ear, 3),
            "mar": round(mar, 3),
            "pitch": round(pitch, 1),
            "yaw": round(yaw, 1),
            "roll": 0.0,
            "gaze_ratio": round(gaze_ratio, 3),
            "gaze_direction": gaze_dir,
            "is_eye_closed": is_closed,
            "is_yawning": is_yawn,
            "is_head_down": is_nod,
            "EAR_mean": temporal_feats["EAR_mean"],
            "EAR_min": temporal_feats["EAR_min"],
            "PERCLOS": temporal_feats["PERCLOS"],
            "blink_count": int(temporal_feats["blink_count"]),
            "blink_frequency": temporal_feats["blink_frequency"],
            "avg_blink_duration": temporal_feats["avg_blink_duration"],
            "max_eye_closure_duration": temporal_feats["max_eye_closure_duration"],
            "MAR_mean": temporal_feats["MAR_mean"],
            "yawn_count": int(temporal_feats["yawn_count"]),
            "head_nod_count": int(temporal_feats["head_nod_count"]),
            "head_pose_variation": temporal_feats["head_pose_variation"],
            "gaze_stability": temporal_feats["gaze_stability"],
            "calibration": calib_status,
            "baseline_summary": self.baseline.get_baseline_summary(),
            "deviations": deviations,
            "active_model": self.active_model_name,
            "model_probs": model_probs,
            "ml_drowsy_prob": round(ml_prob_drowsy, 3),
            "fatigue_score": score_info["fatigue_score"],
            "fatigue_trend": score_info["trend"],
            "state": current_state,
            "voice_alert_type": voice_alert_type,
            "buzzer": bool(voice_alert_type is not None),
            "buzzer_reason": buzzer_reason,
            "buzzer_reason_ta": buzzer_reason_ta,
            "buzzer_reason_ta_phonetic": buzzer_reason_ta_phonetic,
            "is_drowsy": bool(current_state == STATE_DROWSINESS),
            "is_distracted": bool(current_state == STATE_DISTRACTION),
            "is_head_turned": bool(abs(yaw) >= 16.0),
            "distraction_duration": 0.0,
            "components": score_info["components"]
        }

        return frame, telemetry

    def close(self):
        """Releases detector and resources."""
        self.detector.close()


def run_cli_webcam(camera_index: int = 0):
    """
    Runs real-time prediction directly in OpenCV window (useful for fast testing).
    Press 'c' to start calibration, 'q' to quit.
    """
    predictor = RealTimePredictor()
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[CLI] Could not open webcam device index {camera_index}.")
        return

    print("\n========================================================")
    print(" AI-Based Driver Fatigue Monitoring System (CLI Demo)")
    print(" Controls: 'c' = Start Calibration (30s), 'q' = Quit")
    print("========================================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, telem = predictor.process_frame(frame, draw_annotations=True)

        # Overlay status banner
        state = telem["state"]
        score = telem["fatigue_score"]
        trend = telem["fatigue_trend"]

        color_bgr = (0, 200, 0)
        if state == STATE_EARLY_FATIGUE:
            color_bgr = (0, 140, 255)
        elif state == STATE_DROWSY:
            color_bgr = (0, 0, 255)

        # Header banner
        cv2.putText(annotated, f"STATUS: {state.upper()}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)
        cv2.putText(annotated, f"Score: {score}% | Trend: {trend}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Metric column
        metrics_text = [
            f"EAR: {telem['ear']:.2f} (Mean: {telem['EAR_mean']:.2f})",
            f"PERCLOS: {telem['PERCLOS'] * 100:.1f}%",
            f"Blinks: {telem['blink_count']} (Dur: {telem['avg_blink_duration']:.2f}s)",
            f"MAR: {telem['mar']:.2f} (Yawns: {telem['yawn_count']})",
            f"Nods: {telem['head_nod_count']} | Gaze: {telem['gaze_direction']}"
        ]

        for i, text in enumerate(metrics_text):
            cv2.putText(annotated, text, (20, 110 + (i * 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        cv2.imshow("Driver Fatigue Monitoring System", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("[CLI] Calibration started!")
            predictor.start_calibration()

    cap.release()
    cv2.destroyAllWindows()
    predictor.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    args = parser.parse_args()

    print("[Self-Test] Initializing RealTimePredictor...")
    pred = RealTimePredictor()
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    _, t = pred.process_frame(dummy)
    print(f"[Self-Test] Processed dummy frame, state: {t['state']}, face_detected: {t['face_detected']}")
    pred.close()
    print("[Self-Test] RealTimePredictor verified successfully.")
