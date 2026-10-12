"""
core/liveness.py
Anti-spoofing module using MediaPipe 1.x FaceLandmarker (with Haar eye detector fallback)
to detect natural blinks via Eye Aspect Ratio (EAR) or blendshapes.
"""

import os
import sys
import time
import cv2
import numpy as np

# Path to task model and haarcascades
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "data", "face_landmarker.task")
EYE_CASCADE_PATH = os.path.join(BASE_DIR, "data", "haarcascade_eye.xml")

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

class LivenessDetector:
    def __init__(self, ear_threshold=0.22, required_blinks=1):
        self.ear_threshold = ear_threshold
        self.required_blinks = required_blinks
        self.landmarker = None
        self.eye_cascade = None
        self.use_mediapipe = False

        self._init_detector()
        self.reset()

    def _init_detector(self):
        # Attempt to load MediaPipe 1.x FaceLandmarker
        if os.path.exists(MODEL_PATH):
            try:
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=True,
                    num_faces=1
                )
                self.landmarker = vision.FaceLandmarker.create_from_options(options)
                self.use_mediapipe = True
            except Exception as e:
                self.use_mediapipe = False

        # Fallback to OpenCV Haar Cascade if MediaPipe is not active
        if not self.use_mediapipe:
            if os.path.exists(EYE_CASCADE_PATH):
                self.eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
            else:
                self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    def reset(self):
        self.blink_count = 0
        self.is_eye_closed = False
        self.start_time = time.time()

    @staticmethod
    def calculate_ear(landmarks, eye_indices):
        p = [landmarks[i] for i in eye_indices]
        a = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
        b = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
        c = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
        if c == 0:
            return 0.0
        return float((a + b) / (2.0 * c))

    def process_frame(self, frame):
        """
        Process a single OpenCV BGR frame.
        Returns:
            passed (bool): True if required blinks met
            ear (float): current average EAR
            blinks (int): total blinks observed since reset
            annotated_frame (ndarray): frame with visual indicators
        """
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        avg_ear = 0.0
        face_detected = False

        if self.use_mediapipe and self.landmarker is not None:
            try:
                import mediapipe as mp
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = self.landmarker.detect(mp_img)

                if results.face_landmarks and len(results.face_landmarks) > 0:
                    face_detected = True
                    mesh = results.face_landmarks[0]
                    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in mesh]

                    left_ear = self.calculate_ear(landmarks, LEFT_EYE)
                    right_ear = self.calculate_ear(landmarks, RIGHT_EYE)
                    avg_ear = (left_ear + right_ear) / 2.0

                    # Draw eye keypoints
                    for idx in LEFT_EYE + RIGHT_EYE:
                        if idx < len(landmarks):
                            cv2.circle(annotated_frame, landmarks[idx], 1, (0, 255, 255), -1)

                    # State machine for blink
                    if avg_ear < self.ear_threshold:
                        if not self.is_eye_closed:
                            self.is_eye_closed = True
                    else:
                        if self.is_eye_closed:
                            self.blink_count += 1
                            self.is_eye_closed = False
            except Exception:
                face_detected = False

        # Fallback: Haar Cascade Eye Detector
        if not self.use_mediapipe and self.eye_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eyes = self.eye_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
            if len(eyes) >= 2:
                face_detected = True
                avg_ear = 0.30
                if self.is_eye_closed:
                    self.blink_count += 1
                    self.is_eye_closed = False
                for (ex, ey, ew, eh) in eyes[:2]:
                    cv2.rectangle(annotated_frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 1)
            elif len(eyes) == 0 and face_detected:
                avg_ear = 0.15
                self.is_eye_closed = True

        status_text = f"Blinks: {self.blink_count}/{self.required_blinks} | EAR: {avg_ear:.2f}"
        color = (0, 255, 0) if self.blink_count >= self.required_blinks else (0, 165, 255)
        cv2.putText(annotated_frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if not face_detected:
            cv2.putText(annotated_frame, "Face Not Detected", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        passed = self.blink_count >= self.required_blinks
        return passed, avg_ear, self.blink_count, annotated_frame

    def close(self):
        if self.landmarker:
            try:
                self.landmarker.close()
            except Exception:
                pass
