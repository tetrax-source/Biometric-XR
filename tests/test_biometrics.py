"""
tests/test_biometrics.py
Unit tests for biometric algorithms and calculations:
- Cosine similarity mathematical behavior
- Eye Aspect Ratio (EAR) calculation on geometric dummy coordinates
- Threshold matching rules
"""

import pytest
import numpy as np
from scipy.spatial.distance import cosine
from core.liveness import LivenessDetector
from core.face_engine import FaceEngine
from core.voice_engine import VoiceEngine

def test_cosine_similarity_properties():
    """Verifies cosine similarity returns 1.0 for identical and 0.0 for orthogonal vectors."""
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_orth = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    sim_identical = 1.0 - cosine(v1, v2)
    sim_orthogonal = 1.0 - cosine(v1, v_orth)

    assert pytest.approx(sim_identical, 1e-5) == 1.0
    assert pytest.approx(sim_orthogonal, 1e-5) == 0.0

def test_eye_aspect_ratio_open_eye():
    """Verifies EAR on simulated open eye coordinates exceeds blink threshold."""
    # Dummy eye indices: [33, 160, 158, 133, 153, 144]
    indices = [0, 1, 2, 3, 4, 5]
    # Wide open eye geometry: height ~ 10px, width ~ 30px
    landmarks = [
        (0, 10),   # 0: left corner
        (10, 15),  # 1: top left
        (20, 15),  # 2: top right
        (30, 10),  # 3: right corner
        (20, 5),   # 4: bottom right
        (10, 5),   # 5: bottom left
    ]
    ear = LivenessDetector.calculate_ear(landmarks, indices)
    # Vertical |p1-p5| = 10, |p2-p4| = 10, Horizontal |p0-p3| = 30
    # EAR = (10 + 10) / (2 * 30) = 20 / 60 = 0.333
    assert pytest.approx(ear, 1e-2) == 0.33
    assert ear >= 0.22  # Open eye passes threshold

def test_eye_aspect_ratio_closed_eye():
    """Verifies EAR on simulated closed eye coordinates falls below blink threshold."""
    indices = [0, 1, 2, 3, 4, 5]
    # Closed eye geometry: height ~ 1px, width ~ 30px
    landmarks = [
        (0, 10),
        (10, 10),
        (20, 10),
        (30, 10),
        (20, 9),
        (10, 9),
    ]
    ear = LivenessDetector.calculate_ear(landmarks, indices)
    # Vertical distances ~ 1, Horizontal ~ 30
    # EAR = (1 + 1) / (2 * 30) = 2 / 60 = 0.033
    assert ear < 0.22  # Blink detected!

def test_face_engine_threshold_matching():
    """Tests FaceEngine match_embedding against enrolled user dictionary."""
    engine = FaceEngine(threshold=0.55)
    target_emb = np.random.randn(128).astype(np.float32)
    target_emb /= np.linalg.norm(target_emb)

    enrolled = {
        "ASWIN": [target_emb]
    }

    # Query with exact vector -> should match ASWIN
    name, sim = engine.match_embedding(target_emb, enrolled)
    assert name == "ASWIN"
    assert sim >= 0.99

    # Query with random orthogonal vector -> should return Unknown
    random_emb = np.random.randn(128).astype(np.float32)
    random_emb /= np.linalg.norm(random_emb)
    # Ensure it's orthogonal
    random_emb -= np.dot(random_emb, target_emb) * target_emb
    random_emb /= np.linalg.norm(random_emb)

    name_unknown, sim_low = engine.match_embedding(random_emb, enrolled)
    assert name_unknown == "Unknown"
    assert sim_low < 0.55

def test_voice_engine_threshold_matching():
    """Tests VoiceEngine verify against enrolled voice profile."""
    engine = VoiceEngine(threshold=0.50)
    enrolled_v = np.random.randn(20).astype(np.float32)
    enrolled_v /= np.linalg.norm(enrolled_v)

    passed, score = engine.verify(enrolled_v, enrolled_v)
    assert passed is True
    assert pytest.approx(score, 1e-4) == 1.0
