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


# ----------------- SPOKEN 4-DIGIT CHALLENGE & SESSION TESTS -----------------

def test_challenge_session_lifecycle():
    """Tests ChallengeSession single-use code creation, expiration, and retry limits."""
    from core.voice_engine import ChallengeSession

    session = ChallengeSession(length=4)
    assert len(session.code) == 4
    assert session.code.isdigit()
    assert session.attempts == 0
    assert session.has_attempts_left() is True
    assert session.is_expired() is False

    # Simulate attempts
    session.record_attempt()
    assert session.attempts == 1
    session.record_attempt()
    session.record_attempt()
    assert session.attempts == 3
    assert session.has_attempts_left() is False

    # Expiration simulation
    session.created_at = 0.0  # Far past timestamp
    assert session.is_expired() is True


def test_spoken_code_digit_grammar_parsing(monkeypatch):
    """
    Tests digit parsing logic for:
    - Exact match
    - Wrong order
    - Missing digit
    - Extra digit
    - 'oh' mapped to '0'
    - Empty speech / noise
    """
    from core.voice_engine import VoiceEngine, WORD_TO_DIGIT
    import core.voice_engine as ve

    # Test dictionary mappings directly
    assert WORD_TO_DIGIT["zero"] == "0"
    assert WORD_TO_DIGIT["oh"] == "0"
    assert WORD_TO_DIGIT["seven"] == "7"

    engine = VoiceEngine()

    # Mock Vosk recognizer results to test verification logic thoroughly
    class MockVoskResult:
        def __init__(self, text):
            self.text = text

    class MockRecognizer:
        def __init__(self, text_to_return):
            self.text_to_return = text_to_return
        def AcceptWaveform(self, pcm):
            return True
        def FinalResult(self):
            import json
            return json.dumps({"text": self.text_to_return})

    dummy_audio = np.zeros(16000, dtype=np.float32)

    # 1. Exact match with standard digits ("four two nine one" -> "4291")
    monkeypatch.setattr(ve, "get_vosk_model", lambda: object())
    import vosk
    monkeypatch.setattr(vosk, "KaldiRecognizer", lambda m, sr, g: MockRecognizer("four two nine one"))
    passed, heard = engine.verify_spoken_code(dummy_audio, "4291")
    assert passed is True
    assert heard == "4291"

    # 2. 'oh' and 'zero' mapped to 0 ("oh zero five three" -> "0053")
    monkeypatch.setattr(vosk, "KaldiRecognizer", lambda m, sr, g: MockRecognizer("oh zero five three"))
    passed, heard = engine.verify_spoken_code(dummy_audio, "0053")
    assert passed is True
    assert heard == "0053"

    # 3. Wrong order ("one two three four" vs expected "4321")
    monkeypatch.setattr(vosk, "KaldiRecognizer", lambda m, sr, g: MockRecognizer("one two three four"))
    passed, heard = engine.verify_spoken_code(dummy_audio, "4321")
    assert passed is False
    assert heard == "1234"

    # 4. Missing digit ("one two three" vs expected "1234")
    monkeypatch.setattr(vosk, "KaldiRecognizer", lambda m, sr, g: MockRecognizer("one two three"))
    passed, heard = engine.verify_spoken_code(dummy_audio, "1234")
    assert passed is False
    assert heard == "123"

    # 5. Extra digit ("one two three four five" vs expected "1234")
    monkeypatch.setattr(vosk, "KaldiRecognizer", lambda m, sr, g: MockRecognizer("one two three four five"))
    passed, heard = engine.verify_spoken_code(dummy_audio, "1234")
    assert passed is False
    assert heard == "12345"

    # 6. Empty speech / silence ("[unk]" or "")
    monkeypatch.setattr(vosk, "KaldiRecognizer", lambda m, sr, g: MockRecognizer("[unk]"))
    passed, heard = engine.verify_spoken_code(dummy_audio, "1234")
    assert passed is False
    assert heard == ""


def test_spoken_code_disabled_flag(tmp_path, monkeypatch):
    """Verifies that if SPOKEN_CODE_CHECK is False, verification passes gracefully."""
    import config
    import core.database as db
    import core.verification_service as vs

    test_db = str(tmp_path / "test_flag.db")
    monkeypatch.setattr(db, "DB_FILE", test_db)
    db.init_db()

    monkeypatch.setattr(vs, "SPOKEN_CODE_CHECK", False)
    # When disabled, process_voice_challenge should succeed with dummy audible audio
    dummy_audio = np.ones(16000, dtype=np.float32) * 0.1
    res = vs.process_voice_challenge(
        user_name="FLAG_STUDENT_01",
        face_conf=0.95,
        audio=dummy_audio,
        expected_code="7890",
        method="Test Unit"
    )
    assert res.success is True
    assert res.heard_digits == "7890"


def test_face_only_enrollment_does_not_require_voice(tmp_path, monkeypatch):
    """Verifies enrollment creates user and face embeddings without needing any voice record."""
    import core.database as db
    test_db = str(tmp_path / "test_enroll.db")
    monkeypatch.setattr(db, "DB_FILE", test_db)
    db.init_db()

    # Enroll face only
    user_name = "FACE_ONLY_STUDENT"
    emb = np.random.randn(128).astype(np.float32)
    db.save_face_embedding(user_name, emb)

    # User and face must exist
    all_faces = db.get_all_face_embeddings()
    assert user_name in all_faces
    assert len(all_faces[user_name]) == 1

    # Voice embedding must be None (not required)
    voice_emb = db.get_voice_embedding(user_name)
    assert voice_emb is None

