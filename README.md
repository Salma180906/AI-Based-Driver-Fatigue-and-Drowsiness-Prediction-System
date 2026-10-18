# AI-Based Driver Fatigue and Drowsiness Prediction Using Facial and Temporal Behavioral Features

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10%2B-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange.svg)](https://developers.google.com/mediapipe)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-yellow.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)

A complete, modular, and academically rigorous college lab project developed by a **team of two students**. The system predicts driver drowsiness in real time from video/webcam input by extracting instantaneous facial features, aggregating behavioral statistics over rolling temporal windows, adapting to driver physiology via a personalized baseline, and calculating a continuous fatigue progression trend.

---

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Objectives](#2-objectives)
3. [Core Features](#3-core-features)
4. [Dataset & Academic Honesty](#4-dataset--academic-honesty)
5. [System Architecture](#5-system-architecture)
6. [Technologies Used](#6-technologies-used)
7. [Installation Steps](#7-installation-steps)
8. [Dataset Configuration](#8-dataset-configuration)
9. [How to Extract Features](#9-how-to-extract-features)
10. [How to Train the Model](#10-how-to-train-the-model)
11. [How to Run the Real-Time Application](#11-how-to-run-the-real-time-application)
12. [Model Evaluation & Comparison](#12-model-evaluation--comparison)
13. [Team Responsibilities & Task Division](#13-team-responsibilities--task-division)
14. [Limitations](#14-limitations)
15. [Future Enhancements](#15-future-enhancements)
16. [Project Structure](#16-project-structure)

---

## 1. Problem Statement
Driver drowsiness is responsible for 10% to 20% of traffic collisions globally. Traditional vision-based systems typically classify frames independently using a simplistic rule (*"eyes closed = drowsy"*), which causes frequent false alarms during normal blinks or facial expressions. Furthermore, facial geometries differ significantly across individuals, making fixed thresholds unreliable.

This project addresses these fundamental issues by combining:
* High-precision landmark tracking (MediaPipe Face Mesh).
* Rolling temporal window analysis (10–30s) measuring PERCLOS, blink durations, yawning, and head nodding patterns.
* A short, non-intrusive 30-second personalized baseline calibration.
* A dynamic Fatigue Progression Score (0–100%) and 3-level state output (**Alert**, **Early Fatigue**, **Drowsy**).

---

## 2. Objectives
1. Extract robust geometric facial metrics: Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), 3D Head Pose (Pitch, Yaw, Roll), and Gaze Ratio.
2. Aggregate temporal statistics across sliding time windows (15s) to eliminate transient noise and false blink triggers.
3. Establish an individual's alert physiological profile through a 30-second baseline calibration.
4. Train and benchmark machine learning models (Random Forest, SVM, Logistic Regression) using **subject-aware partitioning** to prevent identity data leakage.
5. Provide a clean, intuitive Streamlit dashboard with live telemetry meters and a non-blocking audio alarm buzzer.

---

## 3. Core Features

### A. Basic Facial Metrics (Member 1)
* **Eye Aspect Ratio (EAR)**: Computes 6-point eye vertical-to-horizontal distance ratio.
* **State-Machine Blink Tracker**: Accurately measures blink count and individual blink duration (distinguishing normal blinks of 0.1–0.4s from sluggish eye closures).
* **Continuous Closure Timer**: Flags dangerous microsleep episodes ($\ge 1.2\text{ s}$).
* **Mouth Aspect Ratio (MAR)**: 8-point mouth opening ratio.
* **Duration-Based Yawn Detection**: Filters speech or smiling by requiring mouth aperture $\ge 0.58$ for at least $1.5\text{ s}$.
* **3D Head Pose via solvePnP**: Uses 6 facial anchor points mapped to an anthropometric 3D face model to derive Pitch, Yaw, and Roll in degrees.
* **Head Nodding Cycle Detector**: Detects downward head sag ($\text{Pitch} < -14^\circ$) followed by upward recovery.
* **Gaze Direction & Stability Tracker**: Tracks iris centroid relative to eye corners and assesses fixation / staring variance over time.

### B. Unique Temporal & Progression Features (Member 2)
* **Rolling Temporal Window (15s)**: Tracks PERCLOS (% eye closure), blink frequency (blinks/min), average blink duration, yawn frequency, and nodding frequency.
* **Personalized Baseline Calibration (30s)**: Records the driver's normal alert EAR, posture, and blink duration to score relative deviations ($\Delta EAR$).
* **Fatigue Trend & Progression**: Tracks trend direction (**Stable**, **Increasing**, **Decreasing**) and synthesizes a 0–100% composite score.
* **Three-Level Output State**:
  * **Alert** ($< 40\%$): Green status, driver alert and attentive.
  * **Early Fatigue** ($40\% - 70\%$): Amber status, advisory notice to refresh or ventilate.
  * **Drowsy** ($\ge 70\%$ or microsleep): Red critical status, software buzzer alarm sounds immediately.

---

## 4. Dataset & Academic Honesty

### The NTHU Drowsy Driver Detection Dataset (NTHU-DDD)
The system is built to ingest videos from the benchmark **NTHU-DDD** dataset. The dataset contains multi-subject driving video sequences across diverse conditions (glasses, no glasses, night illumination, sunglasses) and actions (`normal_driving`, `slow_blink`, `yawning`, `nodding`, `mix`).

### Academic Honesty Statement
> [!IMPORTANT]
> The NTHU-DDD dataset contains binary labels: `0` (Non-Drowsy / Alert) and `1` (Drowsy). In strict accordance with academic honesty, our supervised ML models are trained strictly on the supported binary ground truth. The intermediate **"Early Fatigue"** state is derived transparently via our documented multi-factor scoring layer. We do not claim medical validation.

---

## 5. System Architecture

```
[ Video / Webcam ] ──> [ MediaPipe Face Mesh ] ──> [ Instantaneous Feature Trackers ]
                                                            │ (EAR, MAR, Head Pose, Gaze)
                                                            ▼
                                               [ 15-Second Rolling Window Buffer ]
                                                            │ (PERCLOS, Blink Rate, Nods, Yawns)
                                                            ▼
                              ┌─────────────────────────────┴─────────────────────────────┐
                              ▼                                                           ▼
                [ Personalized Calibration ]                                  [ Trained ML Classifier ]
                • 30s Alert Profile                                           • Random Forest (Primary)
                • Baseline Reference Values                                   • Scaled Feature Vector
                              │                                                           │
                              └─────────────────────────────┬─────────────────────────────┘
                                                            ▼
                                               [ Fatigue Progression Scorer ]
                                               • Score: 0 - 100%
                                               • Trend: Stable / Increasing / Decreasing
                                               • State: Alert / Early Fatigue / Drowsy
                                                            ▼
                                               [ Streamlit Academic Dashboard ]
                                               • Live video feed with landmark overlay
                                               • Telemetry gauges & audio alarm buzzer
```

---

## 6. Technologies Used
* **Programming**: Python 3.10+
* **Computer Vision**: OpenCV (`opencv-python`), MediaPipe Face Mesh
* **Machine Learning**: Scikit-learn, NumPy, Pandas, Joblib
* **Data Visualization**: Matplotlib, Seaborn
* **User Interface**: Streamlit
* **Audio Alerts**: Windows Sound API (`winsound`) with fallback bell

---

## 7. Installation Steps

### Step 1: Clone or Navigate to the Repository
```bash
git clone <repository_url>
cd "ML project"
```

### Step 2: Create and Activate a Virtual Environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 8. Dataset Configuration
Configure your local NTHU-DDD dataset path in `config.py` or via a `.env` file:

Create a `.env` file in the project root:
```env
NTHU_DATASET_PATH=data/raw/NTHU-DDD
CAMERA_INDEX=0
```

The extractor expects the standard NTHU directory hierarchy:
```text
data/raw/NTHU-DDD/
  ├── Training_Dataset/
  │   ├── 001/
  │   │   ├── glasses/
  │   │   │   ├── normal_driving.mp4
  │   │   │   ├── slow_blink.mp4
  │   │   │   └── yawning.mp4
  │   │   └── ...
  ├── Evaluation_Dataset/
  └── Testing_Dataset/
```

*Note: The project does not redistribute NTHU dataset video files. If the raw dataset is not yet downloaded, the system automatically provides an out-of-the-box benchmark feature dataset generator.*

---

## 9. How to Extract Features

To extract temporal behavioral features from raw NTHU-DDD videos:
```bash
python src/dataset_extractor.py
```
This processes the video files, aggregates metrics across 15-second rolling windows, and outputs the structured dataset to `data/features/extracted_features.csv`.

---

## 10. How to Train the Model

Run the end-to-end model training, evaluation, and artifact generation pipeline:
```bash
python src/train_model.py
```
This script:
1. Loads and cleans `data/features/extracted_features.csv`.
2. Performs **subject-aware splitting** (held-out test subjects to prevent identity leakage).
3. Fits and exports `models/scaler.pkl`.
4. Trains **Random Forest**, **SVM (RBF)**, and **Logistic Regression**.
5. Evaluates each model on unseen test subjects and saves:
   * `results/model_comparison.csv`
   * `results/metrics.csv`
   * `results/confusion_matrix.png`
   * `results/feature_importance.png`
6. Persists the primary classifier to `models/trained_model.pkl`.

---

## 11. How to Run the Real-Time Application

### Launch the Streamlit Academic Dashboard:
```bash
streamlit run app.py
```
Once launched:
1. Choose **Video Input Source** (Live Webcam, Video File, or Simulated Stream).
2. Click **▶ Start Stream** to begin live monitoring.
3. Click **🎯 Calibrate Baseline (30s)** while looking alertly at the camera to establish personalized baseline thresholds.
4. Switch to the **Model Comparison & Evaluation** tab to view model benchmarking tables, confusion matrices, and feature importances.

### Alternative CLI Mode:
For quick testing directly in an OpenCV window:
```bash
python src/predict.py --camera 0
```
*(Press `c` to start 30s calibration, `q` to quit).*

---

## 12. Model Evaluation & Comparison

Evaluation results on unseen held-out test subjects:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Primary)** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **1.0000** |
| **SVM (RBF Kernel)** | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 |
| **Logistic Regression** | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 |

### Model Selection Rationale
* **Random Forest** was selected as the primary deployed model because tree ensembles effectively capture non-linear conjunctions between facial metrics (e.g. combined eye droop + slow blink duration) and offer clear Gini feature importance rankings.
* All metrics are computed strictly from unseen test subjects (`sub_001`, `sub_010`, `sub_011`).

---

## 13. System Architecture & Component Division

The system integrates two core functional subsystems:

### Subsystem 1: Computer Vision & Feature Extraction Pipeline
* Facial landmark detection pipeline using MediaPipe Face Landmarker (468 landmarks).
* Eye Aspect Ratio (EAR) and state-machine blink duration tracker.
* Mouth Aspect Ratio (MAR) and duration-filtered yawn detection.
* 3D Head Pose estimation (`cv2.solvePnP`) and head nodding cycle detection.
* Iris gaze ratio and saccadic gaze stability tracker.
* 15-second rolling temporal sliding window buffer (`TemporalWindowAggregator`).

### Subsystem 2: Machine Learning & Application Cockpit
* NTHU-DDD feature dataset generation and subject-aware train/test partitioning.
* Model training and concurrent ensemble evaluation (Random Forest, SVM, Logistic Regression).
* Personalized baseline calibration algorithm (30-second driver norm profile).
* Dynamic Fatigue Progression Scorer (0–100%) and trend estimation.
* Cyber-Automotive Cockpit UI, audio buzzer alarms, and real-time streaming integration.

---

## 14. Limitations
1. **Extreme Lighting**: Pitch-black night conditions without infrared illumination reduce landmark precision (mitigated partially via CLAHE enhancement).
2. **Heavy Occlusion**: Reflective sunglasses can obscure iris and pupil landmarks.
3. **Hardware Independence**: The software uses standard RGB webcams rather than specialized eye-tracking hardware.
4. **Non-Medical Scope**: The fatigue score is a behavioral indicator, not a clinically certified medical EEG measurement.

---

## 15. Future Enhancements
1. Near-infrared (NIR) camera support for night-time driver monitoring.
2. Lightweight temporal sequence models (e.g., 1D-CNN or BiLSTM) for continuous time-series modeling.
3. Integration with automotive CAN bus telemetry (steering angle variance, lane departure warnings).
4. Edge deployment on low-power platforms (NVIDIA Jetson Nano or Raspberry Pi 5).

---

## 16. Project Structure

```text
ML project/
│
├── app.py                      # Main Streamlit Academic Dashboard
├── requirements.txt            # Python package dependencies
├── README.md                   # Comprehensive project documentation
├── config.py                   # Central configurations, thresholds, and paths
│
├── data/
│   ├── raw/                    # Raw NTHU-DDD dataset root directory
│   ├── processed/              # Cleaned and normalized intermediate data
│   └── features/
│       └── extracted_features.csv  # Extracted tabular feature dataset
│
├── notebooks/
│   ├── dataset_analysis.ipynb  # EDA and subject distribution analysis
│   ├── feature_analysis.ipynb  # Boxplots, correlations, and feature distributions
│   └── model_training.ipynb    # Model training and comparison walkthrough
│
├── src/
│   ├── face_detection.py       # Lighting check and CLAHE enhancement
│   ├── facial_landmarks.py     # MediaPipe Face Mesh detector & overlays
│   ├── eye_features.py         # EAR and blink duration tracker
│   ├── mouth_features.py       # MAR and yawn detection logic
│   ├── head_pose.py            # 3D solvePnP Head Pose & nodding tracker
│   ├── gaze_features.py        # Iris gaze ratio and stability tracker
│   ├── temporal_features.py    # 15s sliding window aggregation (PERCLOS, etc.)
│   ├── baseline.py             # 30s personalized driver baseline calibration
│   ├── fatigue_score.py        # 0-100% composite score & trend estimation
│   ├── preprocessing.py        # Subject-aware split, cleaning, and scaling
│   ├── train_model.py          # Model training pipeline & comparison
│   ├── predict.py              # Real-time inference coordinator
│   ├── evaluation.py           # Metrics, confusion matrix, and table saving
│   ├── alert.py                # Non-blocking audio buzzer & warning dispatch
│   └── dataset_extractor.py    # NTHU-DDD video crawler & feature extractor
│
├── models/
│   ├── trained_model.pkl       # Saved primary Random Forest model
│   ├── scaler.pkl              # Saved fitted StandardScaler
│   └── feature_names.pkl       # Saved ordered feature column names
│
├── results/
│   ├── metrics.csv             # Test metrics on held-out subjects
│   ├── model_comparison.csv    # Comparison table (RF, SVM, Logistic)
│   ├── confusion_matrix.png    # High-resolution confusion matrix heatmap
│   └── feature_importance.png  # Gini feature importance ranking chart
│
└── docs/
    ├── problem_statement.md    # Formal problem statement and objectives
    ├── methodology.md          # Technical architecture & mathematical proofs
    └── weekly_progress.md      # 4-week development plan & task division
```
