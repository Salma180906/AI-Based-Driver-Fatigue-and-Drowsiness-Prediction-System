# Problem Statement

## Project Title
**AI-Based Driver Fatigue and Drowsiness Prediction Using Facial and Temporal Behavioral Features**

---

## 1. Context & Motivation
Driver fatigue and drowsiness are leading contributors to vehicular accidents worldwide. According to global road safety statistics, driver sleepiness accounts for approximately 10% to 20% of commercial and private vehicular collisions. A driver experiencing microsleeps or sluggish cognitive response times suffers from impaired decision-making and delayed braking reflexes.

Traditional drowsiness detection systems suffer from severe real-world limitations:
1. **Instantaneous Frame Flaws**: Many naive computer vision solutions rely solely on instantaneous eye closure (e.g., if eyes are closed in frame $i$, flag drowsiness). This produces high false-alarm rates during natural eye blinks, facial expressions, or lighting flickers.
2. **Lack of Temporal Context**: True physiological fatigue manifests as behavioral escalation over time—progressive PERCLOS increase, longer average blink duration, repeated yawning episodes, downward head nodding cycles, and fixed or erratic staring.
3. **Subject-Agnostic Bias**: Facial morphology varies dramatically across human populations. A person with naturally narrower eye palpebral fissures will trigger false positives in a static-threshold system, while a driver with wide eyes may fall asleep before a generic threshold is breached.
4. **Data Leakage in Research**: Many academic studies improperly evaluate their models by performing random frame-level train/test splits. Consequently, frames of the same subject appear in both training and test partitions, causing models to memorize facial identities rather than genuine fatigue indicators.

---

## 2. Project Objectives
This project, developed as a 4-week college lab project by a team of two students, develops a practical, modular, and academically honest AI system with the following core objectives:

1. **Facial Landmark Behavioral Extraction**: Accurately track Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), 3D Head Pose (Pitch, Yaw, Roll via solvePnP), and Gaze Ratio using MediaPipe Face Mesh.
2. **Temporal Windowing**: Aggregate instantaneous signals across a rolling sliding window (10–30 seconds) to compute statistically robust temporal metrics (PERCLOS, blink frequency, mean blink duration, yawn frequency, nodding frequency, gaze stability).
3. **Personalized Baseline Calibration**: Implement a non-intrusive 30-second initial calibration routine that measures the driver's normal alert values, enabling relative deviation scoring.
4. **Multi-Factor Fatigue Progression**: Replace hard instantaneous binary switches with a transparent, documented Fatigue Trend Score (0–100%) and 3-level progression state (**Alert**, **Early Fatigue**, **Drowsy**).
5. **Subject-Aware Validation**: Guarantee rigorous academic evaluation by ensuring that test subjects are completely separated from training subjects, avoiding identity data leakage.
6. **Deployable Real-Time Dashboard**: Build a lightweight, accessible Streamlit user interface featuring a live camera stream, telemetry meters, and an audible software alarm buzzer.
