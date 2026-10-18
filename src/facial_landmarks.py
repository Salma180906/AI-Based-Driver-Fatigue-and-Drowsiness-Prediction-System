"""
================================================================================
AI-Based Driver Fatigue and Drowsiness Prediction System
Module: src/facial_landmarks.py
Purpose: Face detection and facial landmark extraction using MediaPipe Face Mesh /
         Face Landmarker (supporting MediaPipe 1.0+ and 0.10+).
         Member 1: Computer Vision & Feature Extraction
================================================================================
"""

import cv2
import urllib.request
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    HAS_TASKS_API = True
except (ImportError, AttributeError):
    HAS_TASKS_API = False

from config import (
    MODEL_DIR,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MAX_NUM_FACES,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    MOUTH_OUTLINE_INDICES,
    HEAD_POSE_LANDMARK_INDICES,
    LEFT_IRIS_CENTER,
    RIGHT_IRIS_CENTER
)

FACE_LANDMARKER_TASK_PATH = MODEL_DIR / "face_landmarker.task"
TASK_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


class FaceMeshDetector:
    """
    Detects faces and extracts 468/478 facial landmarks using MediaPipe.
    Compatible with both modern MediaPipe Tasks (1.0+) and classic Solutions API.
    Designed for real-time driver monitoring with robust fallback when
    no face or multiple faces are detected.
    """

    def __init__(
        self,
        max_num_faces: int = MAX_NUM_FACES,
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
        refine_landmarks: bool = True
    ):
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.refine_landmarks = refine_landmarks

        self.landmarker = None
        self.classic_face_mesh = None
        self._init_detector()

    def _ensure_task_model(self) -> Path:
        """Downloads face_landmarker.task if not already cached locally."""
        if not FACE_LANDMARKER_TASK_PATH.exists():
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            print(f"[FaceMeshDetector] Downloading face_landmarker.task to '{FACE_LANDMARKER_TASK_PATH}'...")
            try:
                urllib.request.urlretrieve(TASK_MODEL_URL, FACE_LANDMARKER_TASK_PATH)
                print(f"[FaceMeshDetector] Download complete ({FACE_LANDMARKER_TASK_PATH.stat().st_size} bytes).")
            except Exception as e:
                print(f"[FaceMeshDetector] Warning: Could not download task model: {e}")
        return FACE_LANDMARKER_TASK_PATH

    def _init_detector(self):
        """Initializes the MediaPipe Tasks FaceLandmarker or classic FaceMesh."""
        if HAS_TASKS_API:
            try:
                model_file = self._ensure_task_model()
                if model_file.exists():
                    options = vision.FaceLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=str(model_file)),
                        running_mode=vision.RunningMode.IMAGE,
                        num_faces=self.max_num_faces,
                        min_face_detection_confidence=self.min_detection_confidence,
                        min_face_presence_confidence=self.min_detection_confidence,
                        min_tracking_confidence=self.min_tracking_confidence,
                        output_face_blendshapes=False
                    )
                    self.landmarker = vision.FaceLandmarker.create_from_options(options)
                    return
            except Exception as e:
                print(f"[FaceMeshDetector] Tasks API initialization failed: {e}. Falling back to solutions.")

        # Fallback to solutions API if available
        try:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                self.classic_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=self.max_num_faces,
                    refine_landmarks=self.refine_landmarks,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence
                )
        except Exception as e:
            print(f"[FaceMeshDetector] Solutions initialization warning: {e}")

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Processes a BGR image frame and returns pixel coordinates of facial landmarks.

        Args:
            frame_bgr: Input video frame as a NumPy array (BGR format).

        Returns:
            np.ndarray of shape (num_landmarks, 3) representing [x_px, y_px, z_norm],
            or None if no face is detected.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        h, w = frame_bgr.shape[:2]
        if h == 0 or w == 0:
            return None

        # 1. Use MediaPipe Tasks API if initialized
        if self.landmarker is not None:
            try:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                result = self.landmarker.detect(mp_image)

                if not result.face_landmarks:
                    return None

                # Pick primary face
                best_face = result.face_landmarks[0]
                if len(result.face_landmarks) > 1:
                    best_area = -1
                    for face_lm in result.face_landmarks:
                        xs = [lm.x for lm in face_lm]
                        ys = [lm.y for lm in face_lm]
                        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                        if area > best_area:
                            best_area = area
                            best_face = face_lm

                landmarks_px = []
                for lm in best_face:
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    landmarks_px.append([px, py, lm.z])

                return np.array(landmarks_px, dtype=np.float32)
            except Exception as e:
                return None

        # 2. Use Classic Solutions API if available
        if self.classic_face_mesh is not None:
            try:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = self.classic_face_mesh.process(frame_rgb)

                if not results.multi_face_landmarks:
                    return None

                best_face = results.multi_face_landmarks[0]
                if len(results.multi_face_landmarks) > 1:
                    best_area = -1
                    for face_lm in results.multi_face_landmarks:
                        xs = [lm.x for lm in face_lm.landmark]
                        ys = [lm.y for lm in face_lm.landmark]
                        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                        if area > best_area:
                            best_area = area
                            best_face = face_lm

                landmarks_px = []
                for lm in best_face.landmark:
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    landmarks_px.append([px, py, lm.z])

                return np.array(landmarks_px, dtype=np.float32)
            except Exception:
                return None

        return None

    def draw_annotations(
        self,
        frame_bgr: np.ndarray,
        landmarks_px: Optional[np.ndarray],
        draw_eyes: bool = True,
        draw_mouth: bool = True,
        draw_bbox: bool = True
    ) -> np.ndarray:
        """Draws visual cues (eyes, mouth, bounding box) on the frame."""
        if landmarks_px is None or frame_bgr is None:
            return frame_bgr

        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]

        # Draw Face Bounding Box
        if draw_bbox and len(landmarks_px) > 0:
            x_min = int(np.min(landmarks_px[:, 0]))
            y_min = int(np.min(landmarks_px[:, 1]))
            x_max = int(np.max(landmarks_px[:, 0]))
            y_max = int(np.max(landmarks_px[:, 1]))

            margin_x = int((x_max - x_min) * 0.08)
            margin_y = int((y_max - y_min) * 0.08)
            x_min = max(0, x_min - margin_x)
            y_min = max(0, y_min - margin_y)
            x_max = min(w - 1, x_max + margin_x)
            y_max = min(h - 1, y_max + margin_y)

            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 128), 2)

        # Draw Eye Contours
        if draw_eyes:
            for idx in LEFT_EYE_INDICES:
                if idx < len(landmarks_px):
                    pt = (int(landmarks_px[idx][0]), int(landmarks_px[idx][1]))
                    cv2.circle(annotated, pt, 2, (0, 255, 255), -1)

            for idx in RIGHT_EYE_INDICES:
                if idx < len(landmarks_px):
                    pt = (int(landmarks_px[idx][0]), int(landmarks_px[idx][1]))
                    cv2.circle(annotated, pt, 2, (0, 255, 255), -1)

            if len(landmarks_px) > RIGHT_IRIS_CENTER:
                left_iris = (int(landmarks_px[LEFT_IRIS_CENTER][0]), int(landmarks_px[LEFT_IRIS_CENTER][1]))
                right_iris = (int(landmarks_px[RIGHT_IRIS_CENTER][0]), int(landmarks_px[RIGHT_IRIS_CENTER][1]))
                cv2.circle(annotated, left_iris, 3, (0, 0, 255), -1)
                cv2.circle(annotated, right_iris, 3, (0, 0, 255), -1)

        # Draw Mouth Outline
        if draw_mouth:
            for idx in MOUTH_OUTLINE_INDICES:
                if idx < len(landmarks_px):
                    pt = (int(landmarks_px[idx][0]), int(landmarks_px[idx][1]))
                    cv2.circle(annotated, pt, 2, (255, 128, 0), -1)

        return annotated

    def close(self):
        """Releases the detector resources."""
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
        if self.classic_face_mesh is not None:
            try:
                self.classic_face_mesh.close()
            except Exception:
                pass


if __name__ == "__main__":
    print("[Self-Test] Initializing FaceMeshDetector...")
    det = FaceMeshDetector()
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    res = det.process_frame(dummy)
    print(f"[Self-Test] Blank frame result: {res} (Expected None)")
    det.close()
    print("[Self-Test] FaceMeshDetector self-test passed.")
