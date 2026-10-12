"""
tests/test_database.py
Unit tests for core/database.py:
- Database creation & table schema
- Embedding serialization / deserialization roundtrip
- User CRUD
- Duplicate attendance rules
- System users (teachers/admins) and audit logging
"""

import os
import pytest
import numpy as np
import core.database as db

@pytest.fixture(autouse=True)
def setup_temp_db(tmp_path, monkeypatch):
    """Isolates each test in a clean temporary SQLite database."""
    test_db = str(tmp_path / "test_attendance.db")
    monkeypatch.setattr(db, "DB_FILE", test_db)
    db.init_db()
    yield test_db

def test_database_initialization(setup_temp_db):
    """Verifies all required tables and indexes exist in the database."""
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    assert "users" in tables
    assert "face_embeddings" in tables
    assert "voice_embeddings" in tables
    assert "attendance_logs" in tables
    assert "system_users" in tables
    assert "audit_log" in tables
    conn.close()

def test_embedding_serialization_roundtrip():
    """Verifies float32 embeddings convert to raw bytes and reconstruct bit-for-bit."""
    original = np.random.randn(128).astype(np.float32)
    original /= np.linalg.norm(original)

    blob = db.serialize_embedding(original)
    assert isinstance(blob, bytes)
    assert len(blob) == 128 * 4  # 512 bytes for 128 float32 values

    reconstructed = db.deserialize_embedding(blob)
    assert reconstructed.shape == original.shape
    assert reconstructed.dtype == np.float32
    assert np.allclose(original, reconstructed, atol=1e-7)

def test_user_and_face_embedding_storage():
    """Tests saving and retrieving face embeddings."""
    name = "TEST_STUDENT"
    emb1 = np.random.randn(128).astype(np.float32)
    emb2 = np.random.randn(128).astype(np.float32)

    db.save_face_embedding(name, emb1)
    db.save_face_embedding(name, emb2)

    all_faces = db.get_all_face_embeddings()
    assert name in all_faces
    assert len(all_faces[name]) == 2
    assert np.allclose(all_faces[name][0], emb1, atol=1e-7)

def test_voice_embedding_storage_upsert():
    """Tests saving and updating voice embeddings."""
    name = "TEST_STUDENT"
    v1 = np.random.randn(20).astype(np.float32)
    db.save_voice_embedding(name, v1)

    loaded = db.get_voice_embedding(name)
    assert loaded is not None
    assert np.allclose(loaded, v1, atol=1e-7)

    # Upsert with new voice vector
    v2 = np.random.randn(20).astype(np.float32)
    db.save_voice_embedding(name, v2)
    loaded_v2 = db.get_voice_embedding(name)
    assert np.allclose(loaded_v2, v2, atol=1e-7)

def test_duplicate_attendance_rule():
    """Tests that attendance can only be logged ONCE per user per day."""
    name = "ALICE"
    ok1, msg1 = db.record_attendance(name, method="Test", face_conf=0.9, voice_sim=0.85)
    assert ok1 is True
    assert "Attendance marked" in msg1

    # Second attempt on the same day must be rejected
    ok2, msg2 = db.record_attendance(name, method="Test", face_conf=0.9, voice_sim=0.85)
    assert ok2 is False
    assert "already logged" in msg2

def test_system_users_and_audit_logging():
    """Tests teacher/admin creation and audit logging."""
    from werkzeug.security import generate_password_hash, check_password_hash
    pwd_hash = generate_password_hash("teacher123")
    
    uid = db.create_system_user("math_teacher", pwd_hash, role="teacher")
    assert uid > 0

    user = db.get_system_user("math_teacher")
    assert user is not None
    assert user["role"] == "teacher"
    assert check_password_hash(user["password_hash"], "teacher123")

    # Audit log
    db.log_audit("math_teacher", "ENROLL_STUDENT", "Enrolled Alice")
    audits = db.get_audit_logs()
    assert len(audits) >= 1
    assert audits[0]["action"] == "ENROLL_STUDENT"
    assert audits[0]["username"] == "math_teacher"
