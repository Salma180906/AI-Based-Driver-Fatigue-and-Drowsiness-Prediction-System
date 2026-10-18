"""
================================================================================
AI-Based Driver Fatigue and Drowsiness Prediction System
Module: src/dataset_extractor.py
Purpose: Extracts facial and temporal behavioral features from NTHU-DDD video
         files into a structured CSV dataset.
         Member 1 & Member 2 Integration
================================================================================
"""

import os
import re
import cv2
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    FEATURES_CSV_PATH,
    NTHU_DATASET_RAW_PATH,
    ROLLING_WINDOW_SECONDS,
    DEFAULT_EAR_THRESHOLD,
    DEFAULT_MAR_YAWN_THRESHOLD
)
from src.facial_landmarks import FaceMeshDetector
from src.eye_features import calculate_ear, extract_eye_landmarks, EyeBlinkTracker
from src.mouth_features import calculate_mar, YawnTracker
from src.head_pose import estimate_head_pose, HeadNodTracker
from src.gaze_features import estimate_gaze
from src.temporal_features import TemporalWindowAggregator
from src.baseline import PersonalizedBaseline


class NTHUDatasetExtractor:
    """
    Crawls NTHU-DDD video directory structure, processes frames using MediaPipe,
    aggregates temporal metrics over rolling windows, and produces the feature CSV.
    """

    def __init__(
        self,
        dataset_root: str = NTHU_DATASET_RAW_PATH,
        output_csv_path: Path = FEATURES_CSV_PATH,
        sample_step_frames: int = 2,
        window_seconds: float = ROLLING_WINDOW_SECONDS
    ):
        self.dataset_root = Path(dataset_root)
        self.output_csv_path = Path(output_csv_path)
        self.sample_step_frames = sample_step_frames
        self.window_seconds = window_seconds

        self.detector = FaceMeshDetector()

    def discover_video_files(self) -> List[Dict[str, Any]]:
        """
        Scans dataset root for NTHU-DDD video files and matches corresponding label files.
        NTHU structure typical layout:
          <Root>/[Training_Dataset|Evaluation_Dataset|Testing_Dataset]/<Subject_ID>/<Scenario>/<action>.mp4
        """
        if not self.dataset_root.exists():
            print(f"[Extractor] Warning: Dataset root directory '{self.dataset_root}' does not exist.")
            return []

        video_extensions = [".mp4", ".avi", ".mkv", ".mov"]
        video_records = []

        for root, _, files in os.walk(self.dataset_root):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in video_extensions:
                    vpath = Path(root) / file

                    # Infer subject_id, scenario, and action
                    parts = list(vpath.parts)
                    subject_id = "unknown_subject"
                    scenario = "normal"
                    action_name = vpath.stem.lower()

                    # Extract subject ID if formatted like '001', '012', 'Subject1', etc.
                    for p in parts:
                        if re.match(r"^\d{2,3}$", p) or "subject" in p.lower():
                            subject_id = p
                        if any(s in p.lower() for s in ["glasses", "noglasses", "night", "sunglasses"]):
                            scenario = p

                    # Determine ground-truth label based on action name or txt annotations
                    # In NTHU-DDD:
                    # normal_driving / non-drowsy -> 0
                    # slow_blink / yawning / nodding / drowsy -> 1
                    label = 0
                    if any(k in action_name for k in ["slow_blink", "yawn", "nod", "drowsy", "sleep"]):
                        label = 1
                    elif "normal" in action_name:
                        label = 0
                    else:
                        label = 1 if "drowsy" in action_name else 0

                    # Check for accompanying txt annotation file (e.g. <video_name>_drowsy.txt)
                    txt_candidate = vpath.with_name(f"{vpath.stem}_drowsy.txt")
                    label_txt_path = txt_candidate if txt_candidate.exists() else None

                    video_records.append({
                        "video_path": vpath,
                        "subject_id": subject_id,
                        "scenario": scenario,
                        "action_name": action_name,
                        "label": label,
                        "label_txt_path": label_txt_path
                    })

        print(f"[Extractor] Discovered {len(video_records)} video files in '{self.dataset_root}'.")
        return video_records

    def process_single_video(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Processes a single video file, extracts features per temporal window,
        and returns a list of feature dictionary rows.
        """
        vpath = record["video_path"]
        subject_id = record["subject_id"]
        video_id = vpath.stem
        default_label = record["label"]
        txt_path = record["label_txt_path"]

        # Read frame-by-frame annotations if available
        frame_labels = []
        if txt_path and txt_path.exists():
            try:
                with open(txt_path, "r") as f:
                    frame_labels = [int(line.strip()) for line in f if line.strip().isdigit()]
            except Exception:
                frame_labels = []

        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"[Extractor] Could not open video file: {vpath}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        # Initialize feature trackers
        blink_tracker = EyeBlinkTracker()
        yawn_tracker = YawnTracker()
        nod_tracker = HeadNodTracker()
        aggregator = TemporalWindowAggregator(window_seconds=self.window_seconds)

        extracted_rows = []
        frame_idx = 0
        last_extraction_time = 0.0
        feature_interval_sec = 1.0  # Extract one feature row per second of video

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % self.sample_step_frames != 0:
                continue

            current_video_time = frame_idx / fps

            # Determine ground truth label for this frame
            if frame_idx < len(frame_labels):
                current_label = frame_labels[frame_idx]
            else:
                current_label = default_label

            landmarks_px = self.detector.process_frame(frame)
            if landmarks_px is not None:
                # 1. Eye features & EAR
                l_eye, r_eye = extract_eye_landmarks(landmarks_px)
                ear_l = calculate_ear(l_eye)
                ear_r = calculate_ear(r_eye)
                ear = (ear_l + ear_r) / 2.0

                blink_res = blink_tracker.update(ear, current_video_time)
                if blink_res["blink_occurred"]:
                    aggregator.log_blink_event(current_video_time, blink_res["last_blink_duration"])

                # 2. Mouth features & MAR
                mar = calculate_mar(landmarks_px)
                yawn_res = yawn_tracker.update(mar, current_video_time)
                if yawn_res["yawn_completed"]:
                    aggregator.log_yawn_event(current_video_time, yawn_res["current_yawn_duration"])

                # 3. Head Pose & Nodding
                pitch, yaw, roll, _, _ = estimate_head_pose(landmarks_px, frame.shape[:2])
                nod_res = nod_tracker.update(pitch, yaw, roll, current_video_time)
                if nod_res["nod_detected"]:
                    aggregator.log_nod_event(current_video_time)

                # 4. Gaze Ratio
                gaze_ratio, _ = estimate_gaze(landmarks_px)

                # 5. Push to rolling aggregator
                aggregator.add_frame_sample(
                    timestamp=current_video_time,
                    ear=ear,
                    is_closed=blink_res["is_eye_closed"],
                    mar=mar,
                    pitch=pitch,
                    yaw=yaw,
                    roll=roll,
                    gaze_ratio=gaze_ratio
                )

            # Sample temporal feature row every feature_interval_sec
            if (current_video_time - last_extraction_time) >= feature_interval_sec:
                last_extraction_time = current_video_time

                feats = aggregator.get_temporal_features(current_video_time)

                # Calculate deviations using nominal alert baseline for dataset feature rows
                ear_dev = (0.28 - feats["EAR_mean"]) / 0.28
                blink_dur_ratio = feats["avg_blink_duration"] / 0.20

                row = {
                    "subject_id": subject_id,
                    "video_id": video_id,
                    "timestamp": round(current_video_time, 2),
                    "EAR_mean": feats["EAR_mean"],
                    "EAR_min": feats["EAR_min"],
                    "blink_count": feats["blink_count"],
                    "blink_frequency": feats["blink_frequency"],
                    "avg_blink_duration": feats["avg_blink_duration"],
                    "max_eye_closure_duration": feats["max_eye_closure_duration"],
                    "PERCLOS": feats["PERCLOS"],
                    "MAR_mean": feats["MAR_mean"],
                    "yawn_count": feats["yawn_count"],
                    "head_nod_count": feats["head_nod_count"],
                    "head_pose_variation": feats["head_pose_variation"],
                    "gaze_stability": feats["gaze_stability"],
                    "EAR_deviation_from_baseline": round(float(np.clip(ear_dev, -0.5, 1.0)), 4),
                    "blink_duration_ratio": round(float(np.clip(blink_dur_ratio, 0.5, 4.0)), 3),
                    "fatigue_trend_val": feats["fatigue_trend_val"],
                    "label": int(current_label)
                }
                extracted_rows.append(row)

        cap.release()
        return extracted_rows

    def extract_from_directory(self) -> pd.DataFrame:
        """
        Extracts features from all discovered videos and writes to output CSV.
        """
        video_records = self.discover_video_files()
        if not video_records:
            print("[Extractor] No video records found. Check dataset path.")
            return pd.DataFrame()

        all_rows = []
        for i, rec in enumerate(video_records):
            print(f"[Extractor] Processing video ({i + 1}/{len(video_records)}): {rec['video_path'].name}...")
            rows = self.process_single_video(rec)
            all_rows.extend(rows)

        df = pd.DataFrame(all_rows)
        self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_csv_path, index=False)
        print(f"[Extractor] Successfully extracted {len(df)} feature windows to {self.output_csv_path}.")
        return df

    def close(self):
        """Releases detector resources."""
        self.detector.close()


def generate_benchmark_feature_dataset(
    output_path: Path = FEATURES_CSV_PATH,
    num_subjects: int = 12,
    windows_per_subject: int = 150,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates a structured, mathematically consistent benchmark feature dataset
    modeling NTHU-DDD distributions (Soukupová & Čech, PERCLOS standards).

    Ensures students can immediately run preprocessing, model training,
    evaluation, and live testing without waiting to download ~40GB of raw video.

    Features are generated on a per-subject basis to ensure subject-aware separation
    without data leakage.
    """
    np.random.seed(random_seed)
    records = []

    for s_idx in range(1, num_subjects + 1):
        subject_id = f"sub_{s_idx:03d}"

        # Each subject has individual physiological baselines
        baseline_ear = np.random.uniform(0.26, 0.33)
        baseline_blink_dur = np.random.uniform(0.16, 0.24)
        baseline_blink_rate = np.random.uniform(12.0, 22.0)

        for w_idx in range(windows_per_subject):
            timestamp = w_idx * 1.5
            # First 55% alert, 45% drowsy / fatigue scenarios
            is_drowsy = 1 if w_idx >= int(windows_per_subject * 0.55) else 0

            # Transition phase: intermediate fatigue right at the boundary
            is_transition = (w_idx >= int(windows_per_subject * 0.50)) and (w_idx < int(windows_per_subject * 0.60))

            if is_drowsy == 0:
                # Alert State Characteristics with natural human variance
                ear_mean = float(np.random.normal(baseline_ear, 0.022))
                ear_min = float(max(0.16, ear_mean - np.random.uniform(0.05, 0.10)))
                # Natural blinking rate with occasional conversational pauses
                blink_count = float(np.random.poisson(baseline_blink_rate / 4.0))
                blink_freq = float(blink_count * 4.0)
                avg_blink_dur = float(np.clip(np.random.normal(baseline_blink_dur, 0.035), 0.10, 0.32))
                max_closure = float(np.random.uniform(0.10, 0.42))
                # Alert PERCLOS typically under 10%
                perclos = float(np.clip(np.random.beta(1.6, 24.0), 0.01, 0.11))
                # Talking or smiling can occasionally elevate MAR
                mar_mean = float(np.random.normal(0.28, 0.05))
                yawn_count = float(1 if np.random.rand() < 0.03 else 0)  # rare isolated yawn/sigh
                head_nod_count = float(1 if np.random.rand() < 0.02 else 0) # rare mirror check nod
                head_pose_var = float(np.random.uniform(1.2, 3.8))
                gaze_stability = float(np.random.uniform(0.80, 0.99))
                fatigue_trend_val = float(np.random.normal(0.0, 0.12))
            else:
                # Drowsy State Characteristics (progressive sagging, micro-sleeps, yawning)
                severity = 0.65 if is_transition else np.random.uniform(0.75, 1.0)
                ear_mean = float(np.random.normal(baseline_ear * (1.0 - 0.28 * severity), 0.028))
                ear_min = float(np.random.uniform(0.07, 0.18))
                # Blink count can be either sluggishly low or rapid flutter
                blink_count = float(np.random.choice([np.random.poisson(2), np.random.poisson(9)]))
                blink_freq = float(blink_count * 4.0)
                avg_blink_dur = float(np.clip(np.random.normal(0.38 * severity + 0.08, 0.07), 0.24, 0.65))
                max_closure = float(np.random.uniform(0.55 * severity, 1.85))
                perclos = float(np.clip(np.random.beta(3.5, 12.0) + (0.10 * severity), 0.09, 0.58))
                mar_mean = float(np.random.normal(0.36 + (0.08 * severity), 0.10))
                yawn_count = float(np.random.choice([0, 1, 2], p=[0.42, 0.38, 0.20]))
                head_nod_count = float(np.random.choice([0, 1, 2, 3], p=[0.40, 0.35, 0.18, 0.07]))
                head_pose_var = float(np.random.uniform(3.2, 7.8))
                gaze_stability = float(np.random.uniform(0.28, 0.72))
                fatigue_trend_val = float(np.random.uniform(0.20, 1.15))

            # Non-linear interaction: low EAR with normal blink duration is just natural small eyes,
            # but low EAR with elevated blink duration indicates genuine fatigue
            ear_dev = (baseline_ear - ear_mean) / baseline_ear
            blink_ratio = avg_blink_dur / baseline_blink_dur

            records.append({
                "subject_id": subject_id,
                "video_id": f"{subject_id}_vid_{(w_idx // 50) + 1}",
                "timestamp": round(timestamp, 2),
                "EAR_mean": round(ear_mean, 4),
                "EAR_min": round(ear_min, 4),
                "blink_count": round(blink_count, 1),
                "blink_frequency": round(blink_freq, 2),
                "avg_blink_duration": round(avg_blink_dur, 4),
                "max_eye_closure_duration": round(max_closure, 4),
                "PERCLOS": round(perclos, 4),
                "MAR_mean": round(mar_mean, 4),
                "yawn_count": round(yawn_count, 1),
                "head_nod_count": round(head_nod_count, 1),
                "head_pose_variation": round(head_pose_var, 3),
                "gaze_stability": round(gaze_stability, 3),
                "EAR_deviation_from_baseline": round(float(np.clip(ear_dev, -0.5, 1.0)), 4),
                "blink_duration_ratio": round(float(np.clip(blink_ratio, 0.5, 4.0)), 3),
                "fatigue_trend_val": round(fatigue_trend_val, 4),
                "label": is_drowsy
            })

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Benchmark] Generated {len(df)} benchmark feature rows for {num_subjects} subjects at '{output_path}'.")
    return df


if __name__ == "__main__":
    print("[Self-Test] Running dataset extractor utility...")
    # Test benchmark dataset generator
    df_test = generate_benchmark_feature_dataset(num_subjects=3, windows_per_subject=20)
    print(f"Generated test dataset shape: {df_test.shape}")
    print(f"Class distribution:\n{df_test['label'].value_counts()}")
    assert "PERCLOS" in df_test.columns
    assert "EAR_mean" in df_test.columns
    print("[Self-Test] Dataset extractor test completed successfully.")
