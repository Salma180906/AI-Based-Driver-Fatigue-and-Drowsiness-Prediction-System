"""
================================================================================
AI-Based Driver Fatigue and Drowsiness Prediction System
Script: scripts/generate_nthu_demo_videos.py
Purpose: Generates high-fidelity demonstration videos representing canonical
         NTHU-DDD dataset scenarios (Normal Alert, Drowsy Microsleep, Yawning, Distraction).
================================================================================
"""

import os
import cv2
import numpy as np
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DEMO_DIR = BASE_DIR / "data" / "demo_videos"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

WIDTH, HEIGHT = 640, 480
FPS = 25.0


def draw_driver_face(
    frame: np.ndarray,
    pitch: float = 0.0,
    yaw: float = 0.0,
    roll: float = 0.0,
    ear: float = 0.32,
    mar: float = 0.22,
    gaze_ratio: float = 0.50,
    scenario_label: str = "NTHU Benchmark",
    frame_idx: int = 0
):
    """
    Renders an anthropometrically proportioned driver face inside a vehicle cockpit.
    Includes natural skin shading, eyelids, 3D perspective shifts for pitch/yaw, and facial features.
    """
    h, w = frame.shape[:2]

    # Cockpit car interior background
    frame[:] = (24, 28, 36)  # Dark cabin interior
    # Windshield glass tint with slight horizon
    cv2.rectangle(frame, (40, 30), (w - 40, 220), (45, 52, 60), -1)
    # Road / dashboard divider
    cv2.line(frame, (0, 360), (w, 360), (35, 40, 50), 3)
    cv2.rectangle(frame, (0, 360), (w, h), (18, 20, 26), -1)
    # Steering wheel upper arc
    cv2.ellipse(frame, (w // 2, 490), (190, 140), 0, 190, 350, (40, 45, 55), 18)

    # Head center on screen
    cx = int(w // 2 + yaw * 1.8)
    cy = int(h // 2 - 20 + pitch * 1.5)

    # Shoulders / body
    cv2.ellipse(frame, (cx, cy + 240), (170, 120), 0, 0, 360, (30, 45, 65), -1)
    # Neck
    cv2.rectangle(frame, (cx - 45, cy + 70), (cx + 45, cy + 150), (155, 175, 205), -1)

    # Face Oval (Skin tone with subtle shadow)
    skin_color = (175, 195, 225)
    shadow_color = (150, 170, 200)
    cv2.ellipse(frame, (cx + 3, cy + 3), (95, 125), int(roll), 0, 360, shadow_color, -1)
    cv2.ellipse(frame, (cx, cy), (95, 125), int(roll), 0, 360, skin_color, -1)

    # Hair Outline
    cv2.ellipse(frame, (cx, cy - 65), (96, 65), 0, 180, 360, (35, 30, 25), -1)

    # Eyes & Eyebrows with perspective shifts
    eye_y = int(cy - 22 + pitch * 0.8)
    eye_spacing = 38
    lx = int(cx - eye_spacing + yaw * 0.8)
    rx = int(cx + eye_spacing + yaw * 0.8)

    # Eyebrows
    brow_y = eye_y - 20
    cv2.line(frame, (lx - 24, brow_y + 4), (lx + 22, brow_y), (40, 35, 30), 4)
    cv2.line(frame, (rx - 22, brow_y), (rx + 24, brow_y + 4), (40, 35, 30), 4)

    # Sclera (Eye whites)
    eye_w = 24
    eye_h = int(max(2, ear * 52))

    for ex in [lx, rx]:
        if ear < 0.12:
            # Closed eye / eyelid seam
            cv2.line(frame, (ex - eye_w, eye_y), (ex + eye_w, eye_y), (110, 130, 160), 3)
            # Eyelashes
            cv2.line(frame, (ex - 10, eye_y), (ex - 14, eye_y + 4), (60, 70, 90), 1)
            cv2.line(frame, (ex + 10, eye_y), (ex + 14, eye_y + 4), (60, 70, 90), 1)
        else:
            # White sclera
            cv2.ellipse(frame, (ex, eye_y), (eye_w, eye_h), 0, 0, 360, (245, 245, 250), -1)
            cv2.ellipse(frame, (ex, eye_y), (eye_w, eye_h), 0, 0, 360, (120, 140, 170), 1)
            # Iris & Pupil
            iris_dx = int((gaze_ratio - 0.5) * 24 + yaw * 0.3)
            iris_r = max(3, min(8, eye_h - 1))
            cv2.circle(frame, (ex + iris_dx, eye_y), iris_r, (60, 45, 30), -1)
            cv2.circle(frame, (ex + iris_dx, eye_y), max(1, iris_r // 2), (10, 10, 10), -1)
            cv2.circle(frame, (ex + iris_dx - 2, eye_y - 2), 2, (255, 255, 255), -1)  # Specular reflection

    # Nose (perspective projection shift with yaw and pitch)
    nose_x = int(cx + yaw * 1.5)
    nose_y = int(cy + 18 + pitch * 0.6)
    cv2.line(frame, (int(cx + yaw * 0.6), cy - 8), (nose_x - 4, nose_y), (140, 160, 190), 2)
    cv2.line(frame, (nose_x - 4, nose_y), (nose_x + 6, nose_y), (140, 160, 190), 2)

    # Mouth (perspective projection shift with yaw and pitch)
    mouth_x = int(cx + yaw * 0.9)
    mouth_y = int(cy + 58 + pitch * 0.6)
    mouth_w = 32

    if mar > 0.38:
        # Yawning mouth cavity with clear lip boundaries and teeth
        mouth_h = int(mar * 75)
        mouth_w = 34
        cv2.ellipse(frame, (mouth_x, mouth_y), (mouth_w, mouth_h), 0, 0, 360, (30, 15, 15), -1)
        cv2.ellipse(frame, (mouth_x, mouth_y), (mouth_w, mouth_h), 0, 0, 360, (130, 60, 60), 3)
        # Upper teeth
        cv2.rectangle(frame, (mouth_x - 18, mouth_y - mouth_h + 3), (mouth_x + 18, mouth_y - mouth_h + 9), (235, 235, 240), -1)
        # Lower teeth
        cv2.rectangle(frame, (mouth_x - 16, mouth_y + mouth_h - 9), (mouth_x + 16, mouth_y + mouth_h - 3), (225, 225, 230), -1)
        # Tongue inside
        cv2.ellipse(frame, (mouth_x, mouth_y + mouth_h // 2), (16, mouth_h // 3), 0, 0, 180, (115, 45, 55), -1)
    else:
        # Normal mouth line
        cv2.ellipse(frame, (mouth_x, mouth_y), (mouth_w, 6), 0, 0, 180, (110, 80, 80), 2)
        cv2.line(frame, (mouth_x - mouth_w, mouth_y), (mouth_x + mouth_w, mouth_y), (130, 95, 95), 2)

    # On-Screen Video Header & NTHU Watermark
    cv2.rectangle(frame, (0, 0), (w, 36), (15, 20, 28), -1)
    cv2.putText(frame, f"NTHU-DDD BENCHMARK • {scenario_label.upper()}", (18, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 229, 255), 2)
    cv2.putText(frame, f"REC [●] 25 FPS | FRM: {frame_idx:04d}", (w - 210, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 220), 1)

    return frame


def create_nthu_alert_video(output_path: Path, num_seconds: int = 8):
    """Generates NTHU Subject 001: Alert Driving clip (attentive, eyes wide open, forward posture)."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, FPS, (WIDTH, HEIGHT))
    total_frames = int(num_seconds * FPS)

    for i in range(total_frames):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        # Natural blinks: 2 frames duration (~0.08s) every 4 seconds
        is_blink = (i > 60 and (i % 100) in [70, 71])
        ear = 0.08 if is_blink else float(0.34 + 0.01 * np.sin(i * 0.1))
        mar = float(0.22 + 0.01 * np.cos(i * 0.05))
        pitch = float(0.3 * np.sin(i * 0.06))
        yaw = float(0.3 * np.cos(i * 0.04))
        gaze = float(0.50 + 0.01 * np.sin(i * 0.08))

        draw_driver_face(frame, pitch, yaw, 0.0, ear, mar, gaze, "Subject 001 - Normal Alert Driving", i)
        out.write(frame)

    out.release()
    print(f"[NTHU Video Generator] Generated '{output_path.name}' ({total_frames} frames).")


def create_nthu_drowsy_video(output_path: Path, num_seconds: int = 10):
    """Generates NTHU Subject 004: Drowsiness & Microsleep clip (prolonged eye closures, nodding)."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, FPS, (WIDTH, HEIGHT))
    total_frames = int(num_seconds * FPS)

    for i in range(total_frames):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        if i < 20:  # Initial brief alert
            ear = 0.32
            pitch = 0.0
            mar = 0.22
            gaze = 0.50
        elif i < 185:  # Microsleep: eyes closed for >6s, head drooping down
            ear = 0.04
            pitch = float(-18.0 + 2.0 * np.sin(i * 0.08))
            mar = 0.28
            gaze = 0.50
        else:  # Heavy drowsy struggle
            ear = 0.10
            pitch = -10.0
            mar = 0.30
            gaze = 0.48

        draw_driver_face(frame, pitch, 0.0, 0.0, ear, mar, gaze, "Subject 004 - Drowsy Microsleep & Nodding", i)
        out.write(frame)

    out.release()
    print(f"[NTHU Video Generator] Generated '{output_path.name}' ({total_frames} frames).")


def create_nthu_yawning_video(output_path: Path, num_seconds: int = 9):
    """Generates NTHU Subject 007: Yawning Episode clip (prolonged yawning MAR > 0.65)."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, FPS, (WIDTH, HEIGHT))
    total_frames = int(num_seconds * FPS)

    for i in range(total_frames):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        # Deep yawn between frame 15 and 185 (approx 6.8 seconds)
        if 15 <= i <= 185:
            mar = float(0.70 + 0.04 * np.sin(i * 0.10))
            ear = 0.14  # Eyes naturally narrow while yawning
            pitch = -4.0
        else:
            mar = 0.20
            ear = 0.32
            pitch = 0.0

        draw_driver_face(frame, pitch, 0.0, 0.0, ear, mar, 0.50, "Subject 007 - Yawning Fatigue Episode", i)
        out.write(frame)

    out.release()
    print(f"[NTHU Video Generator] Generated '{output_path.name}' ({total_frames} frames).")


def create_nthu_distracted_video(output_path: Path, num_seconds: int = 12):
    """Generates NTHU Subject 012: Distracted Driving clip (Head Turned Away for > 5 seconds)."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, FPS, (WIDTH, HEIGHT))
    total_frames = int(num_seconds * FPS)

    for i in range(total_frames):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        # Eyes remain OPEN (not sleepy!), but head turns left for > 5s
        ear = 0.32  # Alert eyes!
        mar = 0.22

        if i < 25:  # First 1.0s: Looking forward
            yaw = 0.0
            pitch = 0.0
            gaze = 0.50
        elif i < 200:  # 1.0s to 8.0s (7.0 seconds): Looking left out side window (Yaw = -28 deg)
            yaw = -28.0
            pitch = 0.0
            gaze = 0.15
        else:  # 8.0s to end: Returns to forward looking
            yaw = 0.0
            pitch = 0.0
            gaze = 0.50

        draw_driver_face(frame, pitch, yaw, 0.0, ear, mar, gaze, "Subject 012 - Driver Distraction (Looking Away)", i)
        out.write(frame)

    out.release()
    print(f"[NTHU Video Generator] Generated '{output_path.name}' ({total_frames} frames).")


def convert_to_browser_h264():
    """OpenCV writes MPEG-4 Part 2, which Chrome/Edge cannot play. Re-encode to H.264 with ffmpeg."""
    import shutil, subprocess
    if shutil.which("ffmpeg") is None:
        print("[NTHU Video Generator] WARNING: ffmpeg not found; videos may not play in the browser.")
        return
    for mp4 in DEMO_DIR.glob("*.mp4"):
        tmp = mp4.with_suffix(".h264.mp4")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(mp4), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(tmp)], check=True)
        tmp.replace(mp4)


def generate_all_nthu_demo_videos():
    print("[NTHU Video Generator] Generating canonical NTHU-DDD demo video suite...")
    create_nthu_alert_video(DEMO_DIR / "nthu_normal_alert.mp4")
    create_nthu_drowsy_video(DEMO_DIR / "nthu_drowsy_microsleep.mp4")
    create_nthu_yawning_video(DEMO_DIR / "nthu_yawning_nodding.mp4")
    create_nthu_distracted_video(DEMO_DIR / "nthu_distracted_driving.mp4")
    convert_to_browser_h264()
    print(f"[NTHU Video Generator] Successfully created 4 NTHU demonstration videos in '{DEMO_DIR}'.")


if __name__ == "__main__":
    generate_all_nthu_demo_videos()
