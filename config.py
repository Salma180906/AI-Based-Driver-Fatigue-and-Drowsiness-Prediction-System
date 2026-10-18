"""
================================================================================
AI-Based Driver Fatigue and Drowsiness Prediction System
Module: config.py
Purpose: Central configuration file defining project paths, feature thresholds,
         temporal window sizes, calibration durations, and environment variables.
Team: College Lab Project (Team of 2 Students)
      Member 1: Computer Vision & Feature Extraction
      Member 2: Machine Learning, Application & UI
================================================================================
"""

import os
from pathlib import Path

# Try importing dotenv if available, otherwise continue gracefully
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ==============================================================================
# 0. PROJECT & TEAM INFORMATION
# ==============================================================================
PROJECT_TITLE = "AI-Based Driver Fatigue and Drowsiness Prediction Using Facial and Temporal Behavioral Features"
SYSTEM_INFO = {
    "system_name": "AI Driver Safety & Fatigue Monitoring System",
    "version": "2.0.0",
    "dataset": "NTHU Drowsy Driver Detection Dataset (NTHU-DDD)",
    "architecture": "3-Model Concurrent Ensemble (Random Forest + SVM + Logistic Regression)"
}
TEAM_MEMBERS = []

# ==============================================================================
# 1. DIRECTORY & FILE PATHS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_FEATURES_DIR = DATA_DIR / "features"

# Default path for extracted features CSV
FEATURES_CSV_PATH = DATA_FEATURES_DIR / "extracted_features.csv"

# Models and results
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "trained_model.pkl"
RF_MODEL_PATH = MODEL_DIR / "random_forest.pkl"
SVM_MODEL_PATH = MODEL_DIR / "svm.pkl"
LR_MODEL_PATH = MODEL_DIR / "logistic_regression.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"

RESULTS_DIR = BASE_DIR / "results"
METRICS_CSV_PATH = RESULTS_DIR / "metrics.csv"
MODEL_COMPARISON_CSV_PATH = RESULTS_DIR / "model_comparison.csv"
CONFUSION_MATRIX_PNG_PATH = RESULTS_DIR / "confusion_matrix.png"
ALL_CONFUSION_MATRICES_PNG_PATH = RESULTS_DIR / "all_confusion_matrices.png"
ROC_CURVES_PNG_PATH = RESULTS_DIR / "roc_curves.png"
FEATURE_IMPORTANCE_PNG_PATH = RESULTS_DIR / "feature_importance.png"

# Optional external NTHU Dataset directory configured via .env or environment variable
NTHU_DATASET_RAW_PATH = os.getenv("NTHU_DATASET_PATH", str(DATA_RAW_DIR / "NTHU-DDD"))

# Ensure essential directories exist
for folder in [DATA_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_FEATURES_DIR, MODEL_DIR, RESULTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. CAMERA & VIDEO STREAM SETTINGS
# ==============================================================================
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30.0

# ==============================================================================
# 3. FACIAL LANDMARK DETECTION SETTINGS (MediaPipe)
# ==============================================================================
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MAX_NUM_FACES = 1

# Landmark indices for MediaPipe Face Mesh (468/478 standard)
# Left Eye landmarks
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
# Right Eye landmarks
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Mouth landmarks for MAR (vertical lip boundaries and horizontal corners)
# 61: Left mouth corner, 291: Right mouth corner
# 81, 13, 311: Upper lip vertical landmarks
# 178, 14, 402: Lower lip vertical landmarks
MOUTH_OUTLINE_INDICES = [61, 81, 13, 311, 291, 402, 14, 178]
MOUTH_UPPER_LIP = 13
MOUTH_LOWER_LIP = 14
MOUTH_CORNER_LEFT = 61
MOUTH_CORNER_RIGHT = 291

# Iris landmark indices (if 478 landmarks available)
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# 3D Head Pose Landmark Points for solvePnP
# 1: Nose tip, 152: Chin, 33: Left eye outer corner, 263: Right eye outer corner,
# 61: Left mouth corner, 291: Right mouth corner
HEAD_POSE_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

# ==============================================================================
# 4. INSTANTANEOUS FEATURE THRESHOLDS (DEFAULT / INITIAL)
# ==============================================================================
# Eye Aspect Ratio (EAR)
DEFAULT_EAR_THRESHOLD = 0.16         # Below this, eye is considered closed (avoids false eye closure on open eyes)
BLINK_MIN_DURATION_SECONDS = 0.08   # Minimum duration for valid blink
BLINK_MAX_DURATION_SECONDS = 0.45   # Normal blinks last ~0.1 - 0.4s
PROLONGED_CLOSURE_SECONDS = 1.0     # Eye closure >= 1.0s indicates micro-sleep / drowsiness

# Mouth Aspect Ratio (MAR)
DEFAULT_MAR_YAWN_THRESHOLD = 0.48   # Above this, mouth is wide open
YAWN_MIN_DURATION_SECONDS = 1.0     # Differentiate talking from continuous yawning

# Head Pose (Degrees)
HEAD_PITCH_NOD_THRESHOLD = -14.0    # Negative pitch = downward tilt / nod
HEAD_NOD_DURATION_WINDOW = 2.0      # Window to detect upward recovery (nodding cycle)

# ==============================================================================
# 5. TEMPORAL ROLLING WINDOW & AGGREGATION
# ==============================================================================
# Rolling time window in seconds (10 - 30 seconds)
ROLLING_WINDOW_SECONDS = 15.0

# PERCLOS threshold over rolling window
PERCLOS_ALERT_THRESHOLD = 0.10      # <= 10% closed time is alert
PERCLOS_EARLY_FATIGUE_THRESHOLD = 0.18 # 10% - 18% closed time is early fatigue
PERCLOS_DROWSY_THRESHOLD = 0.25     # > 25% closed time is drowsy

# Blink frequency thresholds (blinks per minute)
NORMAL_BLINK_RATE_LOW = 10.0
NORMAL_BLINK_RATE_HIGH = 25.0
FATIGUE_BLINK_RATE_LOW = 6.0        # Staring / sluggish blinks
FATIGUE_BLINK_RATE_HIGH = 35.0      # Rapid flutter blinks due to fatigue

# ==============================================================================
# 6. PERSONALIZED BASELINE CALIBRATION
# ==============================================================================
CALIBRATION_DURATION_SECONDS = 30.0  # 30-60 second calibration at start

# ==============================================================================
# 7. FATIGUE TREND & PROGRESSION SCORING
# ==============================================================================
# Trend categories
TREND_STABLE = "Stable"
TREND_INCREASING = "Increasing"
TREND_DECREASING = "Decreasing"

# Fatigue score ranges (0 to 100)
FATIGUE_SCORE_ALERT_MAX = 40.0
FATIGUE_SCORE_EARLY_FATIGUE_MAX = 70.0

# 3-Level Prediction Categories: Alertness, Drowsiness, and Distraction
STATE_ALERT = "ALERT"
STATE_DROWSINESS = "DROWSINESS"
STATE_DISTRACTION = "DISTRACTION"
STATE_DROWSY = "DROWSINESS"         # Backward compatibility alias
STATE_EARLY_FATIGUE = "Early Fatigue"

# Colors for UI Status (Hex)
COLOR_ALERT = "#10b981"            # Emerald Green
COLOR_DROWSINESS = "#ef4444"       # Vibrant Red
COLOR_DROWSY = "#ef4444"           # Alias for COLOR_DROWSINESS
COLOR_DISTRACTION = "#f59e0b"      # Warning Amber / Yellow
COLOR_EARLY_FATIGUE = "#eab308"    # Early fatigue yellow

# Demo Videos Directory
DATA_DEMO_VIDEOS_DIR = DATA_DIR / "demo_videos"
DATA_DEMO_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 8. MACHINE LEARNING SETTINGS
# ==============================================================================
RANDOM_STATE = 42
TEST_SPLIT_RATIO = 0.20
CV_FOLDS = 5

FEATURE_COLUMNS = [
    "EAR_mean",
    "EAR_min",
    "blink_count",
    "blink_frequency",
    "avg_blink_duration",
    "max_eye_closure_duration",
    "PERCLOS",
    "MAR_mean",
    "yawn_count",
    "head_nod_count",
    "head_pose_variation",
    "gaze_stability",
    "EAR_deviation_from_baseline",
    "blink_duration_ratio",
    "fatigue_trend_val"
]

TARGET_COLUMN = "label"  # Ground truth binary: 0 = Alert/Normal, 1 = Drowsy

# Audio Alert Settings (Hardware buzzer disabled in favor of natural voice alerts)
AUDIO_ALARM_ENABLED = False
AUDIO_FREQUENCY_HZ = 1200
AUDIO_DURATION_MS = 600
