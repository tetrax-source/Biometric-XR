"""
tests/test_auth_routes.py
Integration tests for Flask authentication, RBAC, sessions, and CSRF protection.
"""

import pytest
from werkzeug.security import generate_password_hash, check_password_hash
from app_web import app
import core.database as db

@pytest.fixture
def client(tmp_path, monkeypatch):
    """Sets up a test client with a clean isolated database."""
    test_db = str(tmp_path / "test_auth.db")
    monkeypatch.setattr(db, "DB_FILE", test_db)
    db.init_db()

    # Pre-seed an admin and a teacher
    db.create_system_user("admin_user", generate_password_hash("adminpass123"), role="admin")
    db.create_system_user("teacher_user", generate_password_hash("teacherpass123"), role="teacher")

    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_password_hashing():
    """Verifies werkzeug password hashing and verification security."""
    pwd = "SecretPassword!2026"
    pwd_hash = generate_password_hash(pwd)
    assert pwd_hash != pwd
    assert check_password_hash(pwd_hash, pwd) is True
    assert check_password_hash(pwd_hash, "WrongPassword") is False

def test_kiosk_public_access(client):
    """Verifies kiosk landing and status endpoints are accessible without login."""
    r_kiosk = client.get("/")
    assert r_kiosk.status_code == 200
    assert b"BIO-SECURE" in r_kiosk.data

    r_status = client.get("/api/status")
    assert r_status.status_code == 200
    json_data = r_status.get_json()
    assert "step" in json_data

def test_protected_routes_unauthenticated(client):
    """Verifies management endpoints return 401 Unauthorized for unauthenticated requests."""
    assert client.get("/api/logs").status_code == 401
    assert client.get("/api/users").status_code == 401
    assert client.get("/api/export_csv").status_code == 401
    assert client.post("/api/enroll", json={"name": "HACKER"}).status_code == 401

def test_login_flow(client):
    """Tests login with invalid and valid credentials."""
    client.get("/login")
    with client.session_transaction() as sess:
        csrf = sess["csrf_token"]

    # 1. Invalid password
    r_bad = client.post("/login", data={
        "username": "admin_user",
        "password": "WrongPassword",
        "csrf_token": csrf
    })
    assert r_bad.status_code == 200
    assert b"Invalid username or password" in r_bad.data

    # 2. Valid credentials -> redirect to /
    r_ok = client.post("/login", data={
        "username": "admin_user",
        "password": "adminpass123",
        "csrf_token": csrf
    }, follow_redirects=False)
    assert r_ok.status_code == 302
    assert r_ok.headers["Location"] == "/"

    # 3. Access protected route now succeeds
    r_logs = client.get("/api/logs")
    assert r_logs.status_code == 200

def test_role_based_access_control(client):
    """Verifies Teacher role cannot access Admin-only faculty endpoints."""
    client.get("/login")
    with client.session_transaction() as sess:
        csrf = sess["csrf_token"]

    # Login as Teacher
    client.post("/login", data={
        "username": "teacher_user",
        "password": "teacherpass123",
        "csrf_token": csrf
    })

    # Teacher CAN view logs
    assert client.get("/api/logs").status_code == 200

    # Teacher CANNOT view faculty management or audit logs (403 Forbidden)
    assert client.get("/api/teachers").status_code == 403
    assert client.get("/api/audit_logs").status_code == 403

def test_csrf_protection_on_mutations(client):
    """Verifies POST endpoints reject requests missing valid CSRF tokens."""
    client.get("/login")
    with client.session_transaction() as sess:
        csrf = sess["csrf_token"]

    # Login as Admin
    client.post("/login", data={
        "username": "admin_user",
        "password": "adminpass123",
        "csrf_token": csrf
    })

    # Mutation with missing CSRF token -> 400 Bad Request
    r_no_csrf = client.post("/api/create_teacher", json={
        "username": "new_teacher",
        "password": "password123",
        "role": "teacher"
    })
    assert r_no_csrf.status_code == 400

    # Mutation with valid CSRF header -> 200 OK
    r_with_csrf = client.post("/api/create_teacher", json={
        "username": "new_teacher",
        "password": "password123",
        "role": "teacher"
    }, headers={"X-CSRF-Token": csrf})
    assert r_with_csrf.status_code == 200
    assert r_with_csrf.get_json()["success"] is True
