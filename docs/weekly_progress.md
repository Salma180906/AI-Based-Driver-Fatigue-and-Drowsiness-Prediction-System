# 4-Week Development Plan & Lab Progress Log

## Project Title
**AI-Based Driver Fatigue and Drowsiness Prediction Using Facial and Temporal Behavioral Features**

**Institution**: College of Engineering / Computer Science Laboratory  
**Roles**:
- **Computer Vision & Feature Extraction Specialist**: Member 1
- **Machine Learning, Pipeline & UI Application Specialist**: Member 2

---

## Team Task Allocation Matrix

| Module / Component | Primary Responsibility | Supporting Responsibility | Status |
| :--- | :--- | :--- | :--- |
| OpenCV Video & MediaPipe Pipeline | **Member 1** | Member 2 | **Completed** |
| Facial Landmarks & Eye Aspect Ratio (EAR) | **Member 1** | Member 2 | **Completed** |
| State-Machine Blink & Duration Tracker | **Member 1** | Member 2 | **Completed** |
| Mouth Aspect Ratio (MAR) & Yawn Detection | **Member 1** | Member 2 | **Completed** |
| 3D Head Pose (`solvePnP`) & Nodding Tracker | **Member 1** | Member 2 | **Completed** |
| Gaze Direction & Stability Tracker | **Member 1** | Member 2 | **Completed** |
| Temporal Sliding Window Buffer (15s) | **Member 1** | Member 2 | **Completed** |
| NTHU-DDD Crawler & Feature Dataset Generator | **Member 2** | Member 1 | **Completed** |
| Preprocessing & Subject-Aware Partitioning | **Member 2** | Member 1 | **Completed** |
| Model Training (Random Forest, SVM, Logistic) | **Member 2** | Member 1 | **Completed** |
| Model Evaluation & Performance Artifacts | **Member 2** | Member 1 | **Completed** |
| Personalized Baseline Calibration (30s) | **Member 2** | Member 1 | **Completed** |
| Fatigue Trend & Progression Scorer | **Member 2** | Member 1 | **Completed** |
| Real-Time Prediction Coordinator (`predict.py`) | Joint (Both) | Joint (Both) | **Completed** |
| Streamlit Academic Dashboard (`app.py`) | **Member 2** | Member 1 | **Completed** |
| Audio Alert & Warning Debouncer | **Member 2** | Member 1 | **Completed** |
| Academic Documentation & Notebooks | Joint (Both) | Joint (Both) | **Completed** |

---

## Weekly Progress Log

### WEEK 1: Research, Dataset Setup & Computer Vision Foundation
* **Milestone**: Problem formulation, literature survey, and real-time facial landmark extraction.
* **Work Completed**:
  - Comprehensive literature review on Soukupová & Čech (2016) EAR formulations and NHTSA PERCLOS criteria.
  - Setup and validation of OpenCV and MediaPipe Face Mesh (468/478 landmarks).
  - Implemented `src/face_detection.py` with lighting assessment (brightness, contrast) and CLAHE enhancement.
  - Implemented `src/facial_landmarks.py` with robust face filtering and visualization overlays.
* **Demonstration Deliverable**:
  - Script running webcam feed with live facial landmark mesh and bounding box detection.

---

### WEEK 2: Instantaneous Feature Extraction & Event Tracking
* **Milestone**: Quantitative measurement of eyes, mouth, and head orientation.
* **Work Completed**:
  - Implemented `src/eye_features.py`: 6-point EAR calculation and state-machine blink duration tracker.
  - Implemented `src/mouth_features.py`: 8-point MAR calculation and duration-filtered yawn tracker ($\ge 1.5\text{ s}$).
  - Implemented `src/head_pose.py`: 3D anthropometric face model mapping with `cv2.solvePnP` to derive Euler angles (Pitch, Yaw, Roll).
  - Implemented `src/gaze_features.py`: Iris horizontal gaze ratio and saccadic scanning tracking.
* **Demonstration Deliverable**:
  - Live display of instantaneous metrics (EAR, MAR, Head Angles, Blink count) on video frames.

---

### WEEK 3: Temporal Analysis, Dataset Extraction & Machine Learning Pipeline
* **Milestone**: Rolling window aggregation, subject-aware ML training, and model benchmarking.
* **Work Completed**:
  - Implemented `src/temporal_features.py`: 15-second FIFO sliding window buffer computing PERCLOS, blink rate, average blink duration, nodding count, and gaze stability.
  - Implemented `src/dataset_extractor.py`: Directory crawler for NTHU-DDD video structure and benchmark generator.
  - Implemented `src/preprocessing.py`: Missing value cleaning and subject-aware train/test partition (avoiding identity leakage).
  - Implemented `src/train_model.py` and `src/evaluation.py`: Trained and benchmarked Random Forest, SVM (RBF), and Logistic Regression; saved metrics, confusion matrices, and feature importances.
* **Demonstration Deliverable**:
  - Generated `results/model_comparison.csv`, `results/confusion_matrix.png`, and `models/trained_model.pkl`.

---

### WEEK 4: Integration, Personalized Calibration & Academic Dashboard
* **Milestone**: Final end-to-end intelligent system with dashboard and alerting.
* **Work Completed**:
  - Implemented `src/baseline.py`: 30-second personalized alert baseline calibration.
  - Implemented `src/fatigue_score.py`: Dynamic composite fatigue scoring (0–100%) and trend estimation (Stable / Increasing / Decreasing).
  - Implemented `src/alert.py`: Non-blocking software buzzer alarm and debounced warnings.
  - Implemented `src/predict.py`: Unified real-time prediction coordinator.
  - Developed `app.py`: Clean academic Streamlit dashboard with camera streaming, live telemetry, and model comparison.
  - Authored comprehensive documentation, project report, and Jupyter notebooks.
* **Final Demonstration Deliverable**:
  - Full end-to-end working system running on webcam/video with real-time fatigue telemetry and multi-state alert dispatch.
