# System Methodology & Technical Architecture

## 1. End-to-End Pipeline Architecture

```
                    [ Video Stream / Webcam Input ]
                                  │
                                  ▼
                     [ Lighting Assessment & CLAHE ]
                                  │
                                  ▼
               [ MediaPipe Face Mesh (468/478 Landmarks) ]
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        ▼                                                   ▼
[ Eye & Blink Analysis ]                           [ Mouth, Head & Gaze ]
• 6-point EAR calculation                          • 8-point MAR calculation
• State-machine blink duration tracker             • 3D solvePnP Head Pose
• Continuous closure timer                         • Iris gaze ratio & stability
        │                                                   │
        └─────────────────────────┬─────────────────────────┘
                                  ▼
             [ Temporal Sliding Window Buffer (15s FIFO) ]
             • PERCLOS (% eye closure duration)
             • Blink Frequency (blinks/min)
             • Average & Maximum Blink Duration (seconds)
             • Yawn Frequency (duration >= 1.5s)
             • Head Nodding Pattern (sag & recovery cycle)
             • Gaze Stability (standard deviation)
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        ▼                                                   ▼
[ Personalized Baseline Calibration ]             [ Machine Learning Inference ]
• 30-second initial alert profile                 • Scaled Feature Vector
• Reference EAR, Blink, Posture, Gaze             • Random Forest Primary Model
• Dynamic relative deviations                     • Drowsiness Probability
        │                                                   │
        └─────────────────────────┬─────────────────────────┘
                                  ▼
                      [ Fatigue Trend & Scoring ]
                      • Composite Score (0 - 100%)
                      • Trend Direction (Stable / Increasing / Decreasing)
                      • 3-Level Categorization:
                        - Alert (< 40%)
                        - Early Fatigue (40% - 70%)
                        - Drowsy (>= 70% or microsleep > 1.2s)
                                  │
                                  ▼
                      [ Streamlit Dashboard & Alerts ]
                      • Live annotated video feed
                      • Telemetry gauges and baseline stats
                      • Software audio buzzer alarm
```

---

## 2. Mathematical Formulations

### A. Eye Aspect Ratio (EAR)
Defined by Soukupová and Čech (2016) for 6 landmark points per eye:
$$EAR = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$
Where:
- $p_1, p_4$ are the lateral and medial eye corners.
- $(p_2, p_6)$ and $(p_3, p_5)$ are upper and lower eyelid coordinate pairs.

The left and right eye ratios are calculated independently and averaged:
$$EAR_{\text{avg}} = \frac{EAR_{\text{left}} + EAR_{\text{right}}}{2}$$

### B. Percentage of Eye Closure (PERCLOS)
PERCLOS is recognized by the US Federal Highway Administration and NHTSA as a gold standard indicator of physiological drowsiness. Over a sliding time window of duration $T = 15\text{ s}$ containing $N$ frames:
$$PERCLOS = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(EAR_i < \theta_{EAR})$$
Where $\theta_{EAR}$ is the dynamic or calibrated eye closure threshold, and $\mathbb{I}(\cdot)$ is the indicator function.

### C. Mouth Aspect Ratio (MAR) & Yawning Detection
To prevent false alarms caused by talking or smiling, yawning requires both elevated aperture and sustained duration:
$$MAR = \frac{\|p_{81} - p_{178}\| + \|p_{13} - p_{14}\| + \|p_{311} - p_{402}\|}{3 \cdot \|p_{61} - p_{291}\|}$$
A yawn is registered only when $MAR \ge 0.58$ persists continuously for $\Delta t \ge 1.5\text{ s}$.

### D. 3D Head Pose via Perspective-n-Point (solvePnP)
Six 2D landmarks (nose tip, chin, eye outer corners, mouth corners) are matched to a standard 3D anthropometric facial model. The camera projection is modeled as:
$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} [\mathbf{R} \mid \mathbf{t}] \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$
The rotation matrix $\mathbf{R}$ is decomposed into Euler angles (Pitch, Yaw, Roll) via `cv2.RQDecomp3x3`. Downward sag ($\text{Pitch} < -14^\circ$) followed by upright recovery within 2.0 seconds is recorded as a **Head Nodding Cycle**.

### E. Gaze Stability
Gaze ratio is calculated as the relative position of the iris center between the outer and inner eye corners:
$$R_{\text{gaze}} = \frac{x_{\text{iris}} - x_{\text{outer}}}{x_{\text{inner}} - x_{\text{outer}}}$$
The rolling standard deviation $\sigma(R_{\text{gaze}})$ measures scanning dynamism. Prolonged values where $\sigma < 0.025$ indicate blank staring episodes.

---

## 3. Personalized Baseline Calibration
During system initiation, the driver completes a 30-second alert calibration phase:
- Normal alert eye opening: $EAR_{\text{base}} = \text{Percentile}_{75}(EAR_{1 \dots M})$
- Dynamic closure threshold: $\theta_{EAR} = \max(0.16, \min(0.26, 0.75 \times EAR_{\text{base}}))$
- Normal blink duration: $D_{\text{base}} = \text{Mean}(D_{\text{blinks}})$
- Relative droop deviation: $\Delta EAR = \frac{EAR_{\text{base}} - EAR_{\text{mean}}}{EAR_{\text{base}}}$

---

## 4. Multi-Factor Fatigue Progression Scoring
The composite fatigue score $S \in [0, 100]\%$ synthesizes:
1. **Supervised ML Probability ($w = 40\%$)**: $P(\text{drowsy}) \times 40$
2. **PERCLOS Contribution ($w = 25\%$)**: $\min(1.0, \frac{PERCLOS}{0.25}) \times 25$
3. **Blink Duration & Droop ($w = 15\%$)**: $\min(1.0, \frac{\text{Excess Blink}}{0.25}) \times 15$
4. **Yawning Occurrence ($w = 10\%$)**: $\min(1.0, \frac{\text{Yawns}}{2}) \times 10$
5. **Head Nodding Cycles ($w = 10\%$)**: $\min(1.0, \frac{\text{Nods}}{2}) \times 10$
6. **Microsleep Override**: If continuous eye closure exceeds $1.2\text{ s}$, the score is forced to $\ge 85\%$.

---

## 5. Machine Learning Model Selection
Three supervised classification models were implemented and comparatively evaluated on subject-aware partitions:
1. **Random Forest Classifier**: Primary model due to its capacity to capture non-linear feature interactions without manual feature transformation.
2. **Support Vector Machine (RBF Kernel)**: Margin-based non-linear separator.
3. **Logistic Regression**: Linear baseline benchmark.

---

## 6. Academic Honesty Statement
* **Dataset Ground Truth**: NTHU-DDD provides binary labels (alert vs. drowsy). No false claims of ground-truth intermediate classes are made.
* **Early Fatigue Derivation**: The intermediate state is transparently derived via the documented multi-factor scoring layer.
* **Identity Leakage Prevention**: Train/test splits are conducted strictly across subject boundaries.
