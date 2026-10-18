"""
================================================================================
AI-Based Driver Fatigue and Drowsiness Prediction System
Module: app.py
Purpose: Advanced Cyber-Automotive Cockpit HUD Dashboard for real-time driver
         monitoring, multi-model classical ML comparison (Random Forest, SVM, LR),
         personalized baseline calibration, fatigue trend scoring, and audio alerts.
ARCHITECTURE:
  • 3-Model Classical Ensemble (Random Forest + SVM + Logistic Regression)
================================================================================
"""

import os
import sys
import cv2
import time
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
import streamlit as st
import streamlit.components.v1 as components

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    PROJECT_TITLE,
    TEAM_MEMBERS,
    MODEL_PATH,
    RF_MODEL_PATH,
    SVM_MODEL_PATH,
    LR_MODEL_PATH,
    SCALER_PATH,
    FEATURES_CSV_PATH,
    MODEL_COMPARISON_CSV_PATH,
    CONFUSION_MATRIX_PNG_PATH,
    ALL_CONFUSION_MATRICES_PNG_PATH,
    ROC_CURVES_PNG_PATH,
    FEATURE_IMPORTANCE_PNG_PATH,
    METRICS_CSV_PATH,
    STATE_ALERT,
    STATE_EARLY_FATIGUE,
    STATE_DROWSY,
    COLOR_ALERT,
    COLOR_EARLY_FATIGUE,
    COLOR_DROWSY,
    CAMERA_INDEX
)
from src.predict import RealTimePredictor
from src.alert import AlertManager
from src.dataset_extractor import generate_benchmark_feature_dataset

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="AI Driver Fatigue Monitoring Cockpit",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# FUTURISTIC CYBER-AUTOMOTIVE HUD CSS STYLING & ANIMATIONS
# ==============================================================================
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    /* Global Dark Automotive Background */
    .stApp {
        background: radial-gradient(circle at 50% 10%, #111827 0%, #0b0f19 60%, #030712 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }

    /* Animated Cyber HUD Header */
    .hud-title-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.7) 100%);
        border: 1px solid rgba(0, 229, 255, 0.3);
        border-radius: 14px;
        padding: 18px 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5), inset 0 0 15px rgba(0, 229, 255, 0.08);
        position: relative;
        overflow: hidden;
    }

    .hud-title-container::before {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, #00e5ff, transparent);
        animation: scanline 4s linear infinite;
    }

    @keyframes scanline {
        0% { left: -100%; }
        100% { left: 100%; }
    }

    .hud-title {
        font-family: 'Orbitron', monospace;
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: 1px;
        color: #ffffff;
        text-shadow: 0 0 15px rgba(0, 229, 255, 0.6);
        margin: 0;
    }

    .hud-subtitle {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: #94a3b8;
        letter-spacing: 0.5px;
        margin-top: 4px;
        margin-bottom: 12px;
    }

    .team-badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 8px;
    }

    .team-badge {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 0.85rem;
        color: #f1f5f9;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        transition: all 0.25s ease;
    }

    .team-badge:hover {
        border-color: #00e5ff;
        box-shadow: 0 0 12px rgba(0, 229, 255, 0.3);
        transform: translateY(-2px);
    }

    /* Status Banners with Animated Radar Pulses */
    .status-banner {
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        margin-bottom: 15px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }

    .status-alert {
        background: linear-gradient(135deg, rgba(5, 46, 22, 0.85) 0%, rgba(20, 83, 45, 0.85) 100%);
        border: 2px solid #10b981;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.35);
        color: #ecfdf5;
    }

    .status-early {
        background: linear-gradient(135deg, rgba(69, 26, 3, 0.85) 0%, rgba(120, 53, 15, 0.85) 100%);
        border: 2px solid #f59e0b;
        box-shadow: 0 0 25px rgba(245, 158, 11, 0.35);
        color: #fffbeb;
        animation: pulse-amber 2s infinite;
    }

    .status-drowsy {
        background: linear-gradient(135deg, rgba(76, 5, 25, 0.9) 0%, rgba(136, 19, 55, 0.9) 100%);
        border: 2px solid #ef4444;
        box-shadow: 0 0 35px rgba(239, 68, 68, 0.65);
        color: #fef2f2;
        animation: pulse-critical 1.2s infinite;
    }

    @keyframes pulse-amber {
        0% { box-shadow: 0 0 15px rgba(245, 158, 11, 0.25); }
        50% { box-shadow: 0 0 30px rgba(245, 158, 11, 0.6); }
        100% { box-shadow: 0 0 15px rgba(245, 158, 11, 0.25); }
    }

    @keyframes pulse-critical {
        0% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.4); transform: scale(1); }
        50% { box-shadow: 0 0 45px rgba(239, 68, 68, 0.85); transform: scale(1.01); }
        100% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.4); transform: scale(1); }
    }

    .status-title {
        font-family: 'Orbitron', monospace;
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        margin: 0;
    }

    .status-desc {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 4px;
        opacity: 0.9;
    }

    /* Glassmorphic Metric Cards */
    .hud-card {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .hud-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 229, 255, 0.4);
    }

    .hud-card-label {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 0.8px;
    }

    .hud-card-value {
        font-family: 'Orbitron', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
        margin: 4px 0;
    }

    .hud-card-sub {
        font-size: 0.8rem;
        color: #64748b;
    }

    /* Animated Trend Indicator */
    .trend-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .trend-stable { background: rgba(14, 165, 233, 0.2); color: #38bdf8; border: 1px solid #0284c7; }
    .trend-increasing { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #dc2626; animation: bounce-trend 1s infinite alternate; }
    .trend-decreasing { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #059669; }

    @keyframes bounce-trend {
        0% { transform: translateY(0); }
        100% { transform: translateY(-2px); }
    }

    /* Model Probability Comparison Bars */
    .prob-bar-container {
        margin: 8px 0;
    }
    .prob-bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 2px;
    }
    .prob-bar-bg {
        background: #1e293b;
        border-radius: 6px;
        height: 8px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.4s ease;
    }

    /* Streamlit Widget tweaks */
    div[data-testid="stMetricValue"] {
        font-family: 'Orbitron', monospace !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# AUDIO GENERATOR COMPONENT (Web Audio API for in-browser sound + backend)
# ==============================================================================
def render_browser_audio_alarm(trigger: bool = False):
    """
    Renders an in-browser Web Audio API synthesizer that plays an authentic
    pulsed 1200 Hz tone directly through the client's speakers when trigger is True.
    """
    if trigger:
        components.html("""
        <script>
            try {
                const AudioContext = window.AudioContext || window.webkitAudioContext;
                if (AudioContext) {
                    const ctx = new AudioContext();
                    const osc = ctx.createOscillator();
                    const gain = ctx.createGain();

                    osc.type = "sawtooth";
                    osc.frequency.setValueAtTime(1200, ctx.currentTime);
                    // Pulsed alarm modulation
                    osc.frequency.exponentialRampToValueAtTime(750, ctx.currentTime + 0.4);

                    gain.gain.setValueAtTime(0.3, ctx.currentTime);
                    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.45);

                    osc.connect(gain);
                    gain.connect(ctx.destination);

                    osc.start();
                    osc.stop(ctx.currentTime + 0.45);
                }
            } catch (e) {
                console.log("Web Audio suppressed or not supported:", e);
            }
        </script>
        """, height=0, width=0)


# ==============================================================================
# SVG CIRCULAR RADIAL FATIGUE GAUGE COMPONENT
# ==============================================================================
def render_radial_fatigue_gauge(score: float, trend: str, state: str):
    """Renders a responsive, animated SVG radial gauge displaying the 0-100% score."""
    clamped_score = max(0.0, min(100.0, float(score)))
    # SVG circle circumference for r=54 is 2 * PI * 54 = 339.29
    circumference = 339.29
    offset = circumference - (clamped_score / 100.0) * circumference

    if state == STATE_ALERT:
        color = "#10b981"
        glow = "rgba(16, 185, 129, 0.6)"
        badge_class = "trend-stable"
        trend_icon = "➡️"
    elif state == STATE_EARLY_FATIGUE:
        color = "#f59e0b"
        glow = "rgba(245, 158, 11, 0.6)"
        badge_class = "trend-decreasing" if trend == "Decreasing" else "trend-increasing"
        trend_icon = "⚠️"
    else:
        color = "#ef4444"
        glow = "rgba(239, 68, 68, 0.75)"
        badge_class = "trend-increasing"
        trend_icon = "🚨"

    gauge_html = f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 10px;">
        <svg width="150" height="150" viewBox="0 0 140 140">
            <!-- Background track -->
            <circle cx="70" cy="70" r="54" fill="none" stroke="#1e293b" stroke-width="10"/>
            <!-- Animated progress circle -->
            <circle cx="70" cy="70" r="54" fill="none" stroke="{color}" stroke-width="10"
                    stroke-dasharray="{circumference}" stroke-dashoffset="{offset:.2f}"
                    stroke-linecap="round" transform="rotate(-90 70 70)"
                    style="filter: drop-shadow(0 0 8px {glow}); transition: stroke-dashoffset 0.5s ease;"/>
            <!-- Center score text -->
            <text x="70" y="65" text-anchor="middle" font-family="'Orbitron', monospace" font-size="24" font-weight="800" fill="#ffffff">
                {clamped_score:.0f}%
            </text>
            <text x="70" y="85" text-anchor="middle" font-family="'Rajdhani', sans-serif" font-size="12" font-weight="600" fill="#94a3b8" letter-spacing="1">
                FATIGUE
            </text>
        </svg>
        <div style="margin-top: 6px;">
            <span class="trend-pill {badge_class}">
                {trend_icon} Trend: {trend}
            </span>
        </div>
    </div>
    """
    st.markdown(gauge_html, unsafe_allow_html=True)


# ==============================================================================
# SINGLETON REAL-TIME PREDICTOR
# ==============================================================================
@st.cache_resource
def get_predictor():
    """Initializes and caches the RealTimePredictor with all models loaded."""
    return RealTimePredictor()


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
def render_sidebar(predictor: RealTimePredictor):
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 15px;">
        <h2 style="font-family: 'Orbitron', monospace; font-size: 1.3rem; color: #00e5ff; margin-bottom: 2px;">
            ⚙️ SYSTEM HUD
        </h2>
        <div style="font-size: 0.8rem; color: #64748b;">NTHU-DDD Feature Engine</div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Model Selector
    st.sidebar.subheader("Active Inference Model")
    available_models = list(predictor.models.keys()) if predictor.models else ["Random Forest"]
    model_choice = st.sidebar.selectbox(
        "Classifier Architecture",
        available_models,
        index=available_models.index("Random Forest") if "Random Forest" in available_models else 0,
        help="Switch dynamically between Random Forest (Primary), SVM (Comparison), and Logistic Regression (Baseline)."
    )
    predictor.set_active_model(model_choice)

    st.sidebar.markdown("---")

    # 2. Input Stream Source
    st.sidebar.subheader("Video Input Source")
    input_mode = st.sidebar.radio(
        "Select Stream Feed",
        ["Simulated Driver Stream (Interactive)", "Live Webcam", "Upload Video File"],
        index=0
    )

    cam_id = CAMERA_INDEX
    uploaded_video = None
    sim_scenario = "Normal Alert Driving"
    custom_params = {}

    if input_mode == "Live Webcam":
        cam_id = st.sidebar.number_input("Camera Index", min_value=0, max_value=5, value=CAMERA_INDEX, step=1)
    elif input_mode == "Upload Video File":
        uploaded_video = st.sidebar.file_uploader("Select Video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
    else:
        # Simulation Controls
        st.sidebar.markdown("### 🎮 Driver Behavior Simulator")
        sim_scenario = st.sidebar.selectbox(
            "Simulation Behavior Preset",
            [
                "Normal Alert Driving",
                "Early Fatigue Warning",
                "Severe Drowsiness & Microsleep",
                "Continuous Yawning Episode",
                "Head Nodding / Saccadic Cycles",
                "Custom Behavioral Sliders"
            ]
        )

        if sim_scenario == "Custom Behavioral Sliders":
            st.sidebar.caption("Fine-tune individual features to test model split boundaries:")
            c_ear = st.sidebar.slider("Eye Aspect Ratio (EAR)", 0.05, 0.38, 0.28, 0.01)
            c_mar = st.sidebar.slider("Mouth Aspect Ratio (MAR)", 0.15, 0.85, 0.26, 0.02)
            c_pitch = st.sidebar.slider("Head Pitch Angle (°)", -30.0, 20.0, 0.0, 1.0)
            c_blink = st.sidebar.slider("Avg Blink Duration (s)", 0.10, 1.80, 0.20, 0.05)
            custom_params = {
                "ear": c_ear,
                "mar": c_mar,
                "pitch": c_pitch,
                "blink_duration": c_blink
            }

    st.sidebar.markdown("---")
    st.sidebar.subheader("Telemetry & Alarm")
    draw_mesh = st.sidebar.checkbox("Facial Landmark Mesh", value=True)
    draw_axes = st.sidebar.checkbox("3D Head Pose Axes", value=True)
    enable_audio = st.sidebar.checkbox("Sound Alarm Buzzer", value=True)

    if st.sidebar.button("🔊 Test Alarm Sound"):
        mgr = AlertManager()
        mgr.trigger_audio_alarm(force=True)
        render_browser_audio_alarm(trigger=True)
        st.sidebar.success("Alarm triggered! (Speaker + Web Synth)")

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06); font-size: 0.8rem;">
        <strong style="color: #00e5ff;">🛡️ ADAS Safety Architecture</strong><br>
        <strong>3-Model Concurrent Ensemble</strong><br>
        <span style="color: #94a3b8;">NTHU Drowsy Driver Detection Benchmark</span>
    </div>
    """, unsafe_allow_html=True)

    return input_mode, cam_id, uploaded_video, sim_scenario, custom_params, draw_mesh, draw_axes, enable_audio


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================
def main():
    predictor = get_predictor()

    # Futuristic Header Banner
    st.markdown("""
    <div class="hud-title-container">
        <h1 class="hud-title">🚗 AI-BASED DRIVER FATIGUE & DROWSINESS PREDICTION SYSTEM</h1>
        <div class="hud-subtitle">
            Facial Landmark Dynamics • Temporal Sliding Windows (10–30s) • Personalized Baseline • Multi-Model Benchmark (RF / SVM / LR)
        </div>
        <div class="team-badge-row">
            <div class="team-badge">
                <span>🛡️</span> <strong>Real-Time ADAS</strong> • Driver Safety Monitoring
            </div>
            <div class="team-badge">
                <span>⚡</span> <strong>Multi-Model Ensemble</strong> • RF + SVM + Logistic Regression
            </div>
            <div class="team-badge" style="border-color: #10b981; color: #10b981;">
                <span>●</span> System Ready • Active
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    input_mode, cam_id, uploaded_video, sim_scenario, custom_params, draw_mesh, draw_axes, enable_audio = render_sidebar(predictor)

    tabs = st.tabs([
        "📹 Real-Time Monitoring Cockpit",
        "📊 Model Comparison & Evaluation",
        "📂 Feature Dataset & Extraction",
        "📘 Project Documentation & Architecture"
    ])

    # ==========================================================================
    # TAB 1: REAL-TIME MONITORING COCKPIT
    # ==========================================================================
    with tabs[0]:
        col_video, col_telemetry = st.columns([7, 5])

        with col_video:
            st.markdown("### 📡 Live Driver Telemetry Feed")
            video_placeholder = st.empty()

            c_btn1, c_btn2, c_btn3 = st.columns([1, 1, 1.4])
            with c_btn1:
                start_stream = st.button("▶ START FEED", key="start_btn", type="primary", use_container_width=True)
            with c_btn2:
                stop_stream = st.button("⏹ STOP FEED", key="stop_btn", use_container_width=True)
            with c_btn3:
                calibrate_btn = st.button("🎯 CALIBRATE BASELINE (30s)", key="calib_btn", use_container_width=True)

            calib_bar_placeholder = st.empty()
            calib_text_placeholder = st.empty()

        with col_telemetry:
            st.markdown("### 🎯 Fatigue Status & Progression")
            status_banner_placeholder = st.empty()

            # Gauge & Active Model side-by-side
            g_col1, g_col2 = st.columns([1, 1])
            with g_col1:
                gauge_placeholder = st.empty()
            with g_col2:
                model_card_placeholder = st.empty()

            st.markdown("---")
            st.markdown("### ⚡ Live Behavioral Features (15s Window)")

            m1, m2 = st.columns(2)
            with m1:
                card_ear = st.empty()
                card_blink = st.empty()
                card_nod = st.empty()
            with m2:
                card_perclos = st.empty()
                card_mar = st.empty()
                card_gaze = st.empty()

            st.markdown("---")
            baseline_card_placeholder = st.empty()

        # Telemetry history chart container
        st.markdown("### 📈 Real-Time Behavioral Feature Telemetry (Rolling 60 Seconds)")
        chart_placeholder = st.empty()

        # Session state for stream and chart history
        if "is_running" not in st.session_state:
            st.session_state.is_running = False
        if "history" not in st.session_state:
            st.session_state.history = deque(maxlen=60)

        if calibrate_btn:
            predictor.start_calibration()
            st.toast("🎯 Personalized baseline calibration started! Look forward and blink naturally for 30s.", icon="🎯")

        if start_stream:
            st.session_state.is_running = True
        if stop_stream:
            st.session_state.is_running = False

        if st.session_state.is_running:
            cap = None
            if input_mode == "Live Webcam":
                cap = cv2.VideoCapture(cam_id)
                if not cap.isOpened():
                    st.error(f"Could not open camera device {cam_id}. Please verify camera connection.")
                    st.session_state.is_running = False
            elif input_mode == "Upload Video File" and uploaded_video is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_video.read())
                tfile.close()
                cap = cv2.VideoCapture(tfile.name)

            frame_idx = 0
            while st.session_state.is_running:
                frame_idx += 1

                if input_mode == "Simulated Driver Stream (Interactive)":
                    time.sleep(0.04)  # ~25-30 FPS pacing
                    annotated, telem = predictor.process_simulated_frame(
                        scenario=sim_scenario,
                        frame_idx=frame_idx,
                        custom_params=custom_params
                    )
                else:
                    if cap is not None:
                        ret, frame = cap.read()
                        if not ret:
                            st.info("Video stream completed or disconnected.")
                            break
                    else:
                        break
                    annotated, telem = predictor.process_frame(frame, draw_annotations=draw_mesh)

                # Render video frame
                frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                # Extract telemetry fields
                st_state = telem["state"]
                st_score = telem["fatigue_score"]
                st_trend = telem["fatigue_trend"]
                st_active_model = telem.get("active_model", predictor.active_model_name)
                st_model_probs = telem.get("model_probs", {})
                st_active_prob = telem["ml_drowsy_prob"]

                # Trigger browser audio alarm if Drowsy
                if st_state == STATE_DROWSY and enable_audio:
                    render_browser_audio_alarm(trigger=True)

                # Update Status Banner
                if st_state == STATE_ALERT:
                    status_banner_placeholder.markdown(f"""
                    <div class="status-banner status-alert">
                        <div class="status-title">🟢 STATUS: {st_state.upper()}</div>
                        <div class="status-desc">Driver is attentive, eyes open, head upright. Normal cognitive state.</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif st_state == STATE_EARLY_FATIGUE:
                    status_banner_placeholder.markdown(f"""
                    <div class="status-banner status-early">
                        <div class="status-title">🟠 STATUS: {st_state.upper()}</div>
                        <div class="status-desc">Early fatigue detected (sluggish blinks / drooping eyes). Advisory warning.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    status_banner_placeholder.markdown(f"""
                    <div class="status-banner status-drowsy">
                        <div class="status-title">🔴 CRITICAL: {st_state.upper()}</div>
                        <div class="status-desc">HIGH DROWSINESS / MICROSLEEP DETECTED! PULL OVER IMMEDIATELY!</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Update Radial Gauge
                with gauge_placeholder.container():
                    render_radial_fatigue_gauge(st_score, st_trend, st_state)

                # Update Model Comparison Card
                prob_rf = st_model_probs.get("Random Forest", st_active_prob)
                prob_svm = st_model_probs.get("SVM (RBF Kernel)", st_active_prob)
                prob_lr = st_model_probs.get("Logistic Regression", st_active_prob)

                model_card_placeholder.markdown(f"""
                <div class="hud-card">
                    <div class="hud-card-label">Active Model: <span style="color: #00e5ff;">{st_active_model}</span></div>
                    <div class="hud-card-value" style="font-size: 1.15rem; color: #38bdf8;">
                        Drowsy Prob: {st_active_prob * 100:.1f}%
                    </div>
                    <div class="prob-bar-container">
                        <div class="prob-bar-label"><span>Random Forest (Primary)</span><span>{prob_rf*100:.0f}%</span></div>
                        <div class="prob-bar-bg"><div class="prob-bar-fill" style="width: {prob_rf*100}%; background: #00e5ff;"></div></div>
                    </div>
                    <div class="prob-bar-container">
                        <div class="prob-bar-label"><span>SVM (RBF Kernel)</span><span>{prob_svm*100:.0f}%</span></div>
                        <div class="prob-bar-bg"><div class="prob-bar-fill" style="width: {prob_svm*100}%; background: #10b981;"></div></div>
                    </div>
                    <div class="prob-bar-container">
                        <div class="prob-bar-label"><span>Logistic Regression</span><span>{prob_lr*100:.0f}%</span></div>
                        <div class="prob-bar-bg"><div class="prob-bar-fill" style="width: {prob_lr*100}%; background: #f59e0b;"></div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Update Live Telemetry Cards
                card_ear.markdown(f"""
                <div class="hud-card">
                    <div class="hud-card-label">👁️ Eye Aspect Ratio (EAR)</div>
                    <div class="hud-card-value">{telem['ear']:.2f}</div>
                    <div class="hud-card-sub">Mean: {telem['EAR_mean']:.2f} | Min: {telem['EAR_min']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

                card_perclos.markdown(f"""
                <div class="hud-card">
                    <div class="hud-card-label">⏱️ PERCLOS (% Closed Eye)</div>
                    <div class="hud-card-value" style="color: {'#ef4444' if telem['PERCLOS'] > 0.20 else '#10b981'};">
                        {telem['PERCLOS'] * 100:.1f}%
                    </div>
                    <div class="hud-card-sub">Threshold: 18.0% | Window: 15s</div>
                </div>
                """, unsafe_allow_html=True)

                card_blink.markdown(f"""
                <div class="hud-card">
                    <div class="hud-card-label">💓 Blink Rate & Duration</div>
                    <div class="hud-card-value">{telem['blink_frequency']:.1f} <span style="font-size: 0.8rem;">/min</span></div>
                    <div class="hud-card-sub">Avg Duration: {telem['avg_blink_duration']:.2f}s | Count: {telem['blink_count']}</div>
                </div>
                """, unsafe_allow_html=True)

                card_mar.markdown(f"""
                <div class="hud-card">
                    <div class="hud-card-label">👄 Mouth Aspect Ratio (MAR)</div>
                    <div class="hud-card-value">{telem['mar']:.2f}</div>
                    <div class="hud-card-sub">Yawns in Window: <strong style="color: {'#ef4444' if telem['yawn_count'] > 0 else '#94a3b8'};">{telem['yawn_count']}</strong></div>
                </div>
                """, unsafe_allow_html=True)

                card_nod.markdown(f"""
                <div class="hud-card">
                    <div class="hud-card-label">🧭 3D Head Pose & Nods</div>
                    <div class="hud-card-value">{telem['pitch']:.1f}° <span style="font-size: 0.8rem;">Pitch</span></div>
                    <div class="hud-card-sub">Head Nods: <strong style="color: {'#ef4444' if telem['head_nod_count'] > 0 else '#94a3b8'};">{telem['head_nod_count']}</strong> | Yaw: {telem['yaw']:.1f}°</div>
                </div>
                """, unsafe_allow_html=True)

                card_gaze.markdown(f"""
                <div class="hud-card">
                    <div class="hud-card-label">🎯 Gaze Direction & Stability</div>
                    <div class="hud-card-value" style="font-size: 1.15rem;">{telem['gaze_direction']}</div>
                    <div class="hud-card-sub">Stability Index: {telem['gaze_stability'] * 100:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)

                # Calibration Feedback
                cal = telem["calibration"]
                if cal["is_calibrating"]:
                    calib_bar_placeholder.progress(cal["progress"])
                    calib_text_placeholder.info(f"🎯 Calibrating driver baseline... {cal['remaining_seconds']}s remaining")
                elif cal["is_calibrated"]:
                    calib_bar_placeholder.progress(1.0)
                    calib_text_placeholder.success("✅ Personalized baseline active!")
                    b_sum = telem["baseline_summary"]
                    baseline_card_placeholder.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 8px; padding: 10px 14px; font-size: 0.85rem;">
                        🎯 <strong>Active Driver Baseline:</strong>
                        Normal EAR: <code>{b_sum['baseline_ear']}</code> |
                        Dynamic Threshold: <code>{b_sum['dynamic_ear_threshold']}</code> |
                        Blink Norm: <code>{b_sum['baseline_blink_duration']}s</code> |
                        EAR Deviation: <code>{telem['deviations']['EAR_deviation_from_baseline']:+.2f}</code>
                    </div>
                    """, unsafe_allow_html=True)

                # Telemetry history chart
                st.session_state.history.append({
                    "Fatigue Score (%)": st_score,
                    "PERCLOS (%)": telem["PERCLOS"] * 100,
                    "EAR (x100)": telem["ear"] * 100
                })
                df_hist = pd.DataFrame(st.session_state.history)
                chart_placeholder.line_chart(df_hist, use_container_width=True)

            if cap is not None:
                cap.release()

    # ==========================================================================
    # TAB 2: MODEL COMPARISON & EVALUATION
    # ==========================================================================
    with tabs[1]:
        st.markdown("## 📊 Machine Learning Model Comparison & Evaluation")
        st.markdown("""
        In accordance with the project abstract, three classical machine-learning algorithms
        (**Random Forest**, **Support Vector Machine (SVM)**, and **Logistic Regression**)
        were trained and rigorously evaluated on held-out test subjects using **Subject-Aware Partitioning**
        (guaranteeing zero identity data leakage between train and test subjects).
        """)

        # Model Comparison Table
        if MODEL_COMPARISON_CSV_PATH.exists():
            df_comp = pd.read_csv(MODEL_COMPARISON_CSV_PATH)
            st.markdown("### 🏆 Comprehensive Benchmark Comparison Table")
            st.dataframe(df_comp, use_container_width=True)
        else:
            st.warning("Model comparison table not found. Run `python src/train_model.py` to generate.")

        st.markdown("---")

        col_roc, col_cm = st.columns([1, 1])

        with col_roc:
            st.markdown("### 📈 Multi-Model ROC Curves Comparison")
            if ROC_CURVES_PNG_PATH.exists():
                st.image(str(ROC_CURVES_PNG_PATH), use_container_width=True, caption="Receiver Operating Characteristic (ROC) Comparison")
            else:
                st.info("ROC Curves image not found.")

        with col_cm:
            st.markdown("### 🎯 Confusion Matrices (Test Subjects)")
            if ALL_CONFUSION_MATRICES_PNG_PATH.exists():
                st.image(str(ALL_CONFUSION_MATRICES_PNG_PATH), use_container_width=True, caption="Multi-Model Confusion Matrices on Test Subjects")
            elif CONFUSION_MATRIX_PNG_PATH.exists():
                st.image(str(CONFUSION_MATRIX_PNG_PATH), use_container_width=True, caption="Primary Model Confusion Matrix")

        st.markdown("---")
        st.markdown("### 🌲 Random Forest Feature Importance Analysis (Gini Metric)")
        if FEATURE_IMPORTANCE_PNG_PATH.exists():
            st.image(str(FEATURE_IMPORTANCE_PNG_PATH), use_container_width=True, caption="Gini Feature Importance Ranking for Random Forest Classifier")

        st.markdown("---")
        st.markdown("### 🧠 Academic Algorithm Selection Rationale")
        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            st.markdown("""
            <div class="hud-card" style="border-top: 3px solid #00e5ff;">
                <h4 style="color: #00e5ff; margin-top: 0;">🌲 Random Forest (Primary)</h4>
                <p style="font-size: 0.88rem; color: #cbd5e1;">
                    <strong>Role:</strong> Primary Model.<br>
                    <strong>Rationale:</strong> Drowsiness features exhibit strong non-linear interactions
                    (e.g., slight EAR drop paired with prolonged blink duration indicates critical micro-sleep,
                    whereas EAR drop with short blink is just squinting). Decision tree ensembles split naturally
                    on these multi-variable boundaries.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_r2:
            st.markdown("""
            <div class="hud-card" style="border-top: 3px solid #10b981;">
                <h4 style="color: #10b981; margin-top: 0;">⚡ SVM - RBF (Comparison)</h4>
                <p style="font-size: 0.88rem; color: #cbd5e1;">
                    <strong>Role:</strong> Comparison Model.<br>
                    <strong>Rationale:</strong> Support Vector Machine with Radial Basis Function kernel maps
                    behavioral features into high-dimensional space, providing a strong classical non-linear
                    margin classification benchmark.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_r3:
            st.markdown("""
            <div class="hud-card" style="border-top: 3px solid #f59e0b;">
                <h4 style="color: #f59e0b; margin-top: 0;">📈 Logistic Regression (Baseline)</h4>
                <p style="font-size: 0.88rem; color: #cbd5e1;">
                    <strong>Role:</strong> Baseline Model.<br>
                    <strong>Rationale:</strong> Provides an interpretable, linear probabilistic benchmark.
                    Demonstrates empirical performance gain achieved when transitioning from linear hyper-planes
                    to non-linear ensemble models.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ==========================================================================
    # TAB 3: DATASET & FEATURE EXTRACTION
    # ==========================================================================
    with tabs[2]:
        st.markdown("## 📂 NTHU Drowsy Driver Detection Dataset (NTHU-DDD)")
        st.markdown("""
        The system is developed and validated against the public **NTHU-DDD** benchmark,
        extracting facial landmarks and 15 aggregated temporal behavioral features across rolling windows.
        """)

        d_col1, d_col2 = st.columns([1, 2])

        with d_col1:
            st.markdown("### Dataset Tools & Actions")
            dataset_path = st.text_input("Local Dataset Path", value="data/raw/NTHU-DDD")
            st.caption("Point this path to your local copy of NTHU-DDD containing subject video folders.")

            if st.button("🔄 Generate / Refresh Benchmark Feature CSV", type="primary"):
                with st.spinner("Generating benchmark features across 12 subjects (1,800 windows)..."):
                    generate_benchmark_feature_dataset(windows_per_subject=150)
                st.success("Extracted features dataset saved to 'data/features/extracted_features.csv'!")
                st.rerun()

        with d_col2:
            st.markdown("### Extracted Features Dataset Preview")
            if FEATURES_CSV_PATH.exists():
                df_feat = pd.read_csv(FEATURES_CSV_PATH)
                st.markdown(f"**Total Feature Windows:** `{len(df_feat)}` | **Unique Subjects:** `{df_feat['subject_id'].nunique()}` | **Features:** `{df_feat.shape[1] - 3}`")
                st.dataframe(df_feat.head(12), use_container_width=True)
            else:
                st.info("Feature dataset not yet found. Click 'Generate Benchmark Feature CSV'.")

    # ==========================================================================
    # TAB 4: PROJECT DOCUMENTATION & ARCHITECTURE
    # ==========================================================================
    with tabs[3]:
        st.markdown("## 📘 Project Abstract & Technical Methodology")

        st.markdown("""
        ### Abstract
        Driver fatigue and drowsiness are major safety concerns that can reduce a driver's attention and reaction ability.
        This project proposes an AI-based system that uses webcam or video input to analyze facial and behavioral features
        and predict the driver's fatigue level. Instead of relying only on single-frame eye closure, the system considers
        changes in facial behavior over time.

        The system extracts features such as **Eye Aspect Ratio (EAR)**, **blink frequency and duration**, **PERCLOS**,
        **Mouth Aspect Ratio (MAR)**, **yawning**, **head pose**, **head-nodding patterns**, and **gaze stability**.
        These features are aggregated over a temporal window and used with machine-learning algorithms to identify fatigue patterns.
        A simple personalized baseline is also introduced to compare the driver's current behavior with their normal behavior.
        The final system provides an **Alert**, **Early Fatigue**, or **Drowsy** status along with a fatigue trend and warning.
        """)

        st.markdown("---")
        st.markdown("### 👥 System Subsystems & Responsibilities")
        st.markdown("""
        * **Subsystem 1: Computer Vision & Feature Extraction**
          * MediaPipe Face Landmarker integration, EAR formula, state-machine blink duration tracker,
            MAR calculation, yawn detection, 3D solvePnP head pose estimation, nodding cycle tracker, gaze ratio & stability,
            15s temporal sliding window aggregation.
        * **Subsystem 2: Machine Learning, Application & UI**
          * NTHU-DDD dataset extraction & subject-aware partition, model training & cross-validation
            (Random Forest, SVM, Logistic Regression), personalized baseline calibration, multi-factor fatigue trend scoring,
            software & browser audio buzzer alarms, and Cyber Cockpit UI.
        """)

        st.markdown("---")
        st.markdown("### 📐 Core Mathematical Formulations")

        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.markdown(r"""
            **1. Eye Aspect Ratio (EAR)**
            $$EAR = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$
            Where $p_1, \dots, p_6$ are the 2D facial landmark coordinates of the eye corners and eyelids.

            **2. Mouth Aspect Ratio (MAR)**
            $$MAR = \frac{\|p_{81} - p_{178}\| + \|p_{13} - p_{14}\| + \|p_{311} - p_{402}\|}{3 \cdot \|p_{61} - p_{291}\|}$$
            A threshold $> 0.58$ persisting for $\ge 1.5$ seconds indicates a genuine yawn event.
            """)

        with f_col2:
            st.markdown(r"""
            **3. Percentage of Eye Closure (PERCLOS)**
            $$PERCLOS = \frac{\sum_{t \in W} \mathbb{I}(EAR_t < EAR_{thresh})}{|W|} \times 100\%$$
            Computed over a sliding $W = 15$-second temporal window.

            **4. 3D Head Pose (Perspective-n-Point)**
            Maps 3D anthropometric face model coordinates $\mathbf{P}_i$ to 2D image coordinates $\mathbf{p}_i$:
            $$\mathbf{p}_i = \mathbf{K} \cdot [\mathbf{R} | \mathbf{t}] \cdot \mathbf{P}_i$$
            Decomposed into Pitch (nodding), Yaw (turning), and Roll (lateral tilt).
            """)


if __name__ == "__main__":
    main()
