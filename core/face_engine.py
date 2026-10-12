"""
core/face_engine.py
Advanced Face Recognition Engine supporting DeepFace neural embeddings (FaceNet)
with cosine similarity verification against database profiles.
"""

import sys
import os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
import cv2
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine

LOCAL_HAAR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "haarcascade_frontalface_default.xml")
HAAR_CASCADE_PATH = LOCAL_HAAR if os.path.exists(LOCAL_HAAR) else (cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class FaceEngine:
    def __init__(self, model_name="Facenet", detector_backend="opencv", threshold=0.55):
        """
        model_name: "Facenet", "Facenet512", "VGG-Face", etc.
        detector_backend: "opencv" for real-time speed.
        threshold: minimum cosine similarity (0.0 to 1.0) to declare a match.
        """
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.threshold = threshold
        self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    def detect_face_bbox(self, frame):
        """Finds the most prominent face bounding box (x, y, w, h)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return None
        # Return largest face
        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        return faces[0]

    def get_embedding(self, face_bgr):
        """
        Extracts 128-d or 512-d normalized embedding vector from cropped face BGR image.
        Returns: 1D numpy array or None.
        """
        try:
            results = DeepFace.represent(
                img_path=face_bgr,
                model_name=self.model_name,
                detector_backend="skip", # already cropped
                enforce_detection=False,
                align=True
            )
            if results and len(results) > 0:
                emb = np.array(results[0]["embedding"], dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                return emb
        except Exception as e:
            # Fallback or error
            pass
        return None

    def match_embedding(self, query_emb, enrolled_dict):
        """
        Compares query embedding with enrolled embeddings in database.
        enrolled_dict: {user_name: [list of np.ndarray embeddings]}
        Returns: (best_name, best_similarity)
        """
        if query_emb is None or not enrolled_dict:
            return None, 0.0

        best_name = None
        best_sim = -1.0

        for name, emb_list in enrolled_dict.items():
            for stored_emb in emb_list:
                # Cosine similarity = 1 - cosine distance
                dist = cosine(query_emb, stored_emb)
                sim = 1.0 - dist
                if sim > best_sim:
                    best_sim = sim
                    best_name = name

        if best_sim >= self.threshold:
            return best_name, float(best_sim)
        return "Unknown", float(best_sim)

    def process_frame(self, frame, enrolled_dict):
        """
        Processes frame for real-time display and recognition.
        Returns: (predicted_name, similarity, bbox, annotated_frame)
        """
        annotated = frame.copy()
        bbox = self.detect_face_bbox(frame)

        predicted_name = None
        similarity = 0.0

        if bbox is not None:
            x, y, w, h = bbox
            face_roi = frame[y:y+h, x:x+w]
            query_emb = self.get_embedding(face_roi)

            if query_emb is not None:
                predicted_name, similarity = self.match_embedding(query_emb, enrolled_dict)
                color = (0, 255, 0) if (predicted_name and predicted_name != "Unknown") else (0, 0, 255)
                label = f"{predicted_name} ({similarity*100:.1f}%)" if predicted_name else "Scanning..."
                cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
                cv2.putText(annotated, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return predicted_name, similarity, bbox, annotated
