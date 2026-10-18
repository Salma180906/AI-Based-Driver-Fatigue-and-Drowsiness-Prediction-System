"""
================================================================================
AI-Based Driver Fatigue and Drowsiness Prediction System
Module: src/face_detection.py
Purpose: Face presence verification, lighting assessment, and frame preprocessing.
         Member 1: Computer Vision & Feature Extraction
================================================================================
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any


def assess_lighting_condition(frame_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Computes brightness and contrast of a frame to warn if camera lighting
    is insufficient for reliable facial landmark detection.

    Returns:
        Dict with 'mean_brightness', 'contrast_std', and 'lighting_status'.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return {"mean_brightness": 0.0, "contrast_std": 0.0, "lighting_status": "No Frame"}

    # Convert to grayscale for illumination analysis
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    contrast_std = float(np.std(gray))

    if mean_brightness < 40.0:
        status = "Poor (Too Dark)"
    elif mean_brightness > 220.0:
        status = "Poor (Overexposed)"
    elif contrast_std < 20.0:
        status = "Low Contrast"
    else:
        status = "Good"

    return {
        "mean_brightness": round(mean_brightness, 2),
        "contrast_std": round(contrast_std, 2),
        "lighting_status": status
    }


def enhance_frame_if_needed(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the
    luminance channel if the frame is under-illuminated.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr

    lighting = assess_lighting_condition(frame_bgr)
    if lighting["mean_brightness"] < 55.0:
        # Convert to LAB color space
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)

        # Merge channels and convert back
        merged = cv2.merge((cl, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return frame_bgr


if __name__ == "__main__":
    print("[Self-Test] Testing face_detection lighting utilities...")
    test_img = np.full((100, 100, 3), 30, dtype=np.uint8)
    lighting = assess_lighting_condition(test_img)
    print(f"Dark image assessment: {lighting}")
    enhanced = enhance_frame_if_needed(test_img)
    print(f"Enhanced mean brightness: {np.mean(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)):.2f}")
    print("[Self-Test] Completed successfully.")
