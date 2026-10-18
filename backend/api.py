"""
================================================================================
AI-Based Driver Fatigue and Drowsiness Prediction System
Module: api.py
Purpose: High-performance FastAPI REST API & Static Server for the React Frontend.
         Executes all 3 classical ML models concurrently (Random Forest, SVM, LR)
         on real streaming camera data and NTHU benchmark videos.
================================================================================
"""

import os
import sys
import cv2
import time
import base64
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure project root is in path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    PROJECT_TITLE,
    SYSTEM_INFO,
    MODEL_COMPARISON_CSV_PATH,
    METRICS_CSV_PATH,
    ROC_CURVES_PNG_PATH,
    ALL_CONFUSION_MATRICES_PNG_PATH,
    CONFUSION_MATRIX_PNG_PATH,
    FEATURE_IMPORTANCE_PNG_PATH,
    DATA_DEMO_VIDEOS_DIR,
    STATE_ALERT,
    STATE_DROWSINESS,
    STATE_DISTRACTION
)
from src.predict import RealTimePredictor

app = FastAPI(
    title="Driver Fatigue & Distraction Prediction API",
    description="Real-Time Computer Vision & Multi-Model Inference API (NTHU-DDD)",
    version="2.0.0"
)

# Enable CORS for React frontend (Vite dev server or production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Predictor Instance (loads all 3 models in memory)
predictor = RealTimePredictor()


class FramePayload(BaseModel):
    image: Optional[str] = None  # Base64 data URL
    source: Optional[str] = "webcam"


@app.get("/api/status")
def get_system_status():
    """Returns system status, active models, and system information."""
    return {
        "status": "online",
        "project_title": PROJECT_TITLE,
        "system_info": SYSTEM_INFO,
        "models": list(predictor.models.keys()),
        "dataset": "NTHU Drowsy Driver Detection Dataset (NTHU-DDD)",
        "features_tracked": [
            "Eye Aspect Ratio (EAR)",
            "Mouth Aspect Ratio (MAR)",
            "3D Head Pose (Pitch, Yaw, Roll)",
            "PERCLOS (15s Window)",
            "Blink Frequency & Duration",
            "Head Nodding Frequency",
            "Gaze Direction & Stability",
            "Personalized Baseline Deviations"
        ]
    }


@app.post("/api/process-frame")
def process_frame(payload: FramePayload):
    """
    Decodes a live camera or video frame, extracts facial landmarks and temporal metrics,
    evaluates all 3 ML models simultaneously, and classifies driver state into:
      • ALERT
      • DROWSINESS
      • DISTRACTION
    """
    if not payload.image:
        return predictor._empty_telemetry("No image frame payload received")

    try:
        # Strip header (e.g., 'data:image/jpeg;base64,')
        raw_b64 = payload.image
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]

        img_bytes = base64.b64decode(raw_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame_bgr is None or frame_bgr.size == 0:
            return predictor._empty_telemetry("Corrupted frame decode")

        # Run full computer vision + temporal aggregation + 3-model inference
        annotated_frame, telemetry = predictor.process_frame(frame_bgr, draw_annotations=False)
        return telemetry

    except Exception as e:
        return predictor._empty_telemetry(f"Processing error: {str(e)}")


@app.post("/api/calibrate")
def trigger_calibration():
    """Initiates 30-second personalized driver baseline calibration."""
    predictor.start_calibration()
    return {
        "status": "calibration_started",
        "duration_seconds": 30.0,
        "message": "Personalized baseline calibration initialized. Please look forward with normal alert posture."
    }


@app.get("/api/demo-videos")
def list_nthu_demo_videos():
    """Returns the list of available NTHU benchmark video scenarios."""
    scenarios = [
        {
            "id": "nthu_normal_alert",
            "title": "NTHU Sub 001 - Normal Alert Driving",
            "scenario": "Alert Driving",
            "description": "Driver focused on the road, alert eyes, natural blinking cycles (EAR ~0.30), forward head posture.",
            "expected_state": "ALERT",
            "filename": "nthu_normal_alert.mp4",
            "stream_url": "/api/demo-video/nthu_normal_alert.mp4"
        },
        {
            "id": "nthu_drowsy_microsleep",
            "title": "NTHU Sub 004 - Drowsy Microsleep & Nodding",
            "scenario": "Drowsiness & Microsleep",
            "description": "Progressive fatigue, drooping eyes, slow blinks, downward head nodding (-18° pitch), and microsleep closures.",
            "expected_state": "DROWSINESS",
            "filename": "nthu_drowsy_microsleep.mp4",
            "stream_url": "/api/demo-video/nthu_drowsy_microsleep.mp4"
        },
        {
            "id": "nthu_yawning_nodding",
            "title": "NTHU Sub 007 - Yawning Fatigue Episode",
            "scenario": "Continuous Yawning",
            "description": "Repeated prolonged mouth openings (MAR > 0.65 for > 2s) and sleepy drooping eyes.",
            "expected_state": "DROWSINESS",
            "filename": "nthu_yawning_nodding.mp4",
            "stream_url": "/api/demo-video/nthu_yawning_nodding.mp4"
        },
        {
            "id": "nthu_distracted_driving",
            "title": "NTHU Sub 012 - Driver Distraction (Looking Away)",
            "scenario": "Distracted Driving",
            "description": "Driver keeps eyes open but turns head away (Yaw > 25°), glances down at phone (Pitch < -20°), diverting attention off-road.",
            "expected_state": "DISTRACTION",
            "filename": "nthu_distracted_driving.mp4",
            "stream_url": "/api/demo-video/nthu_distracted_driving.mp4"
        }
    ]
    return scenarios


@app.post("/api/reset-state")
def reset_predictor_state():
    """Resets temporal sliding window aggregators and trackers for a fresh session."""
    predictor.reset()
    return {"status": "ok", "message": "Driver monitoring state and temporal aggregators reset successfully."}


@app.get("/api/demo-video/{filename}")
def stream_demo_video(filename: str):
    """Streams an NTHU benchmark video file with CORS headers."""
    video_path = DATA_DEMO_VIDEOS_DIR / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Demo video file not found")
    return FileResponse(
        str(video_path),
        media_type="video/mp4",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Accept-Ranges": "bytes"
        }
    )


@app.get("/api/model-comparison")
def get_model_comparison_metrics():
    """Returns actual NTHU-DDD experimental evaluation metrics across all 3 models."""
    if not MODEL_COMPARISON_CSV_PATH.exists():
        raise HTTPException(status_code=404, detail="Model comparison metrics not found")

    df_comp = pd.read_csv(MODEL_COMPARISON_CSV_PATH)
    metrics_list = df_comp.to_dict(orient="records")

    summary = {
        "primary_model": "Random Forest",
        "comparison_model": "Support Vector Machine (SVM - RBF)",
        "baseline_model": "Logistic Regression",
        "comparison_table": metrics_list,
        "evaluation_strategy": "Subject-Aware Partition (Zero Identity Leakage across Held-Out NTHU Subjects)",
        "feature_count": len(predictor.feature_names)
    }
    return summary


@app.get("/api/results/{filename}")
def get_result_image(filename: str):
    """Serves high-resolution evaluation graphs (ROC, confusion matrices, feature importance)."""
    img_path = BASE_DIR / "results" / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Result asset not found")
    return FileResponse(str(img_path), media_type="image/png")


AUDIO_DIR = BASE_DIR / "data" / "audio"

@app.get("/api/voice-alert-audio/{filename}")
def stream_voice_audio(filename: str):
    """Streams authentic natural voice speech alerts (Native Tamil and English)."""
    audio_path = AUDIO_DIR / filename
    if not audio_path.exists():
        alt_path = BASE_DIR.parent / "frontend" / "public" / "audio" / filename
        if alt_path.exists():
            audio_path = alt_path
        else:
            raise HTTPException(status_code=404, detail="Voice audio file not found")
    return FileResponse(
        str(audio_path),
        media_type="audio/mpeg",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Accept-Ranges": "bytes"
        }
    )


# Mount built React frontend if distribution exists
FRONTEND_DIST = BASE_DIR.parent / "frontend" / "dist"
if not FRONTEND_DIST.exists():
    FRONTEND_DIST = BASE_DIR / "frontend" / "dist"

if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    print("[API] Starting Driver Fatigue & Distraction Monitoring Server on port 8765...")
    uvicorn.run("api:app", host="127.0.0.1", port=8765, reload=False)
