"""
app_web.py
Flask Web Application for Multi-Modal Biometric Attendance System.
Supports:
- Public Kiosk Mode (Liveness -> Face -> Voice check-in)
- Role-Based Access Control (Admin and Teacher roles)
- Faculty & Admin Login / Logout with password hashing (werkzeug.security)
- Session inactivity timeout (30 minutes)
- CSRF protection for authenticated mutations
- Comprehensive Audit Logging for all administrative actions
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import time
import secrets
import threading
import datetime
from functools import wraps
import cv2
import numpy as np
from flask import (
    Flask, render_template, Response, jsonify, request,
    session, redirect, url_for, send_file
)
from werkzeug.security import check_password_hash, generate_password_hash

# Core modules
from core.database import (
    init_db, add_or_get_user, save_face_embedding, save_voice_embedding,
    get_all_face_embeddings, get_voice_embedding, get_all_users, delete_user,
    record_attendance, get_attendance_logs, export_attendance_to_csv,
    get_system_user, create_system_user, get_all_system_users,
    log_audit, get_audit_logs
)
from core.liveness import LivenessDetector
from core.face_engine import FaceEngine
from core.liveness import LivenessDetector
from core.face_engine import FaceEngine
from core.voice_engine import VoiceEngine
from core.verification_service import process_voice_challenge, finalize_attendance

from config import (
    CAMERA_INDEX, FACE_THRESHOLD, VOICE_THRESHOLD,
    EAR_THRESHOLD, REQUIRED_BLINKS, SECRET_KEY,
    SESSION_TIMEOUT_MINUTES, CODE_LENGTH, CODE_EXPIRY_SECONDS,
    MAX_CODE_ATTEMPTS, SPOKEN_CODE_CHECK, VOICE_MATCH_ENABLED
)

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(minutes=SESSION_TIMEOUT_MINUTES)

# Initialize DB & Biometric Engines
init_db()

# Security Check: Ensure at least one administrator account exists in system_users
try:
    _admin_users = [u for u in get_all_system_users() if u.get("role") == "admin"]
    if not _admin_users:
        print("\n" + "!" * 70)
        print("[NOTICE] No administrator accounts configured.")
        print("Run 'python create_admin.py' to create an admin account.")
        print("!" * 70 + "\n")
except Exception:
    pass

liveness_detector = LivenessDetector(ear_threshold=EAR_THRESHOLD, required_blinks=REQUIRED_BLINKS)
face_engine = FaceEngine(threshold=FACE_THRESHOLD)
voice_engine = VoiceEngine(threshold=VOICE_THRESHOLD)

# State management
camera_lock = threading.Lock()
camera = None
enrolled_faces = get_all_face_embeddings()

def generate_random_code(length=CODE_LENGTH):
    import random
    return "".join([str(random.randint(0, 9)) for _ in range(length)])

state = {
    "step": 1,                     # 1: Liveness, 2: Face, 3: Spoken Challenge, 4: Logged
    "liveness_passed": False,
    "blink_count": 0,
    "ear": 0.0,
    "user_name": None,             # Kept internal; not exposed to public kiosk
    "face_conf": 0.0,
    "challenge_code": "",          # Current random 4-digit code
    "code_created_at": 0.0,
    "code_attempts": 0,
    "max_attempts": MAX_CODE_ATTEMPTS,
    "heard_digits": "",
    "result_message": "System Ready - Look into camera and blink to begin",
    "is_recording": False
}

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(CAMERA_INDEX)
    return camera

def reset_state():
    global state
    liveness_detector.reset()
    state["step"] = 1
    state["liveness_passed"] = False
    state["blink_count"] = 0
    state["ear"] = 0.0
    state["user_name"] = None
    state["face_conf"] = 0.0
    state["challenge_code"] = ""
    state["code_created_at"] = 0.0
    state["code_attempts"] = 0
    state["heard_digits"] = ""
    state["result_message"] = "Looking for natural blink..."
    state["is_recording"] = False

def reload_enrolled():
    global enrolled_faces
    enrolled_faces = get_all_face_embeddings()

# ----------------- SECURITY & SESSION HELPERS -----------------

@app.before_request
def check_session_timeout():
    session.permanent = True
    now = time.time()
    
    # Check inactivity timeout for authenticated sessions
    if 'user' in session:
        last_activity = session.get('last_activity')
        timeout_seconds = SESSION_TIMEOUT_MINUTES * 60
        if last_activity and (now - last_activity > timeout_seconds):
            username = session['user'].get('username')
            log_audit(username, "SESSION_TIMEOUT", "Session expired due to 30 minutes of inactivity.")
            session.clear()
            if request.path.startswith('/api/'):
                return jsonify({"error": "Session expired due to inactivity. Please log in again."}), 401
            return redirect(url_for('login'))
        session['last_activity'] = now

    # Ensure CSRF token exists
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(16)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            if request.path.startswith('/api/'):
                return jsonify({"error": "Authentication required. Please log in."}), 401
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def role_required(allowed_roles):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user' not in session:
                if request.path.startswith('/api/'):
                    return jsonify({"error": "Authentication required."}), 401
                return redirect(url_for('login'))
            user_role = session['user'].get('role')
            if user_role not in allowed_roles:
                if request.path.startswith('/api/'):
                    return jsonify({"error": "Forbidden: Insufficient privileges."}), 403
                return render_template('index.html', error="Access Denied: Insufficient permissions.", current_user=session['user'], csrf_token=session.get('csrf_token'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def verify_csrf():
    token = request.headers.get('X-CSRF-Token') or request.form.get('csrf_token')
    expected = session.get('csrf_token')
    if not token or not expected or not secrets.compare_digest(str(token), str(expected)):
        return False
    return True

# ----------------- VIDEO STREAMING GENERATOR -----------------

def generate_video_stream():
    global camera, state, enrolled_faces
    cap = get_camera()

    while True:
        with camera_lock:
            if not cap.isOpened():
                time.sleep(0.1)
                continue
            ret, frame = cap.read()

        if not ret:
            time.sleep(0.03)
            continue

        frame = cv2.flip(frame, 1) # Mirror preview
        display_frame = frame.copy()

        # Step 1: Liveness Detection
        if state["step"] == 1:
            passed, ear, blinks, ann_frame = liveness_detector.process_frame(frame)
            display_frame = ann_frame
            state["ear"] = round(ear, 2)
            state["blink_count"] = blinks
            if passed:
                state["liveness_passed"] = True
                state["step"] = 2
                state["result_message"] = "Liveness verified! Scanning face..."

        # Step 2: Face Recognition
        elif state["step"] == 2:
            name, sim, bbox, ann_frame = face_engine.process_frame(frame, enrolled_faces)
            display_frame = ann_frame
            if name and name != "Unknown":
                state["user_name"] = name
                state["face_conf"] = round(sim * 100, 1)
                state["step"] = 3
                state["challenge_code"] = generate_random_code()
                state["code_created_at"] = time.time()
                state["code_attempts"] = 0
                state["heard_digits"] = ""
                state["result_message"] = "Face identified. Speak the 4-digit code shown below."
            elif name == "Unknown":
                state["result_message"] = "Face not recognized in database."

        ret_encode, buffer = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret_encode:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

# ----------------- WEB PAGES & AUTH ROUTES -----------------

@app.route('/')
def index():
    current_user = session.get('user')
    return render_template('index.html', current_user=current_user, csrf_token=session.get('csrf_token'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if not verify_csrf():
            return render_template('login.html', error="Invalid CSRF token.", csrf_token=session.get('csrf_token')), 400

        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '')

        user = get_system_user(username)
        if user and check_password_hash(user['password_hash'], password):
            session['user'] = {
                'id': user['id'],
                'username': user['username'],
                'role': user['role']
            }
            session['last_activity'] = time.time()
            log_audit(username, "LOGIN_SUCCESS", f"User {username} logged in with role {user['role']}.")
            return redirect(url_for('index'))
        else:
            log_audit(username, "LOGIN_FAILED", f"Failed login attempt for username {username}.")
            return render_template('login.html', error="Invalid username or password.", csrf_token=session.get('csrf_token'))

    return render_template('login.html', csrf_token=session.get('csrf_token'))

@app.route('/logout')
def logout():
    user = session.get('user')
    if user:
        log_audit(user['username'], "LOGOUT", f"User {user['username']} logged out.")
    session.clear()
    return redirect(url_for('login'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ----------------- KIOSK VERIFICATION ENDPOINTS (OPEN) -----------------

@app.route('/api/status')
def api_status():
    """Returns kiosk status without exposing sensitive user list or logs."""
    return jsonify({
        "step": state["step"],
        "liveness_passed": state["liveness_passed"],
        "blink_count": state["blink_count"],
        "ear": state["ear"],
        "challenge_code": state["challenge_code"],
        "heard_digits": state["heard_digits"],
        "code_attempts": state["code_attempts"],
        "max_attempts": state["max_attempts"],
        "result_message": state["result_message"],
        "is_recording": state["is_recording"]
    })

@app.route('/api/reset', methods=['POST'])
def api_reset():
    reset_state()
    return jsonify({"success": True, "message": "Pipeline reset."})

@app.route('/api/verify_voice', methods=['POST'])
def api_verify_voice():
    if state["step"] != 3 or not state["user_name"]:
        return jsonify({"success": False, "error": "Face not recognized yet."})

    if state["is_recording"]:
        return jsonify({"success": False, "error": "Already recording audio."})

    # Code expiration check
    if state["code_created_at"] > 0 and (time.time() - state["code_created_at"]) > CODE_EXPIRY_SECONDS:
        msg = f"Challenge code expired ({CODE_EXPIRY_SECONDS}s). Restarting verification."
        reset_state()
        state["result_message"] = msg
        return jsonify({"success": False, "message": msg, "restart": True})

    state["is_recording"] = True
    user = state["user_name"]

    try:
        audio = None
        # Check if browser uploaded audio
        from config import USE_BROWSER_AUDIO
        if USE_BROWSER_AUDIO and "audio" in request.files and request.files["audio"].filename:
            import io
            import librosa
            audio_file = request.files["audio"]
            audio_bytes = audio_file.read()
            if len(audio_bytes) > 0:
                try:
                    audio_buf = io.BytesIO(audio_bytes)
                    audio, sr = librosa.load(audio_buf, sr=16000, mono=True)
                except Exception:
                    audio = None

        # Fallback to server microphone
        if audio is None:
            audio = voice_engine.record_audio(duration=3.5)

        state["code_attempts"] += 1
        res = process_voice_challenge(
            user_name=user,
            face_conf=state["face_conf"] / 100.0,
            audio=audio,
            expected_code=state["challenge_code"],
            method="Face + Spoken Code + Liveness (Web Kiosk)"
        )

        # Dual validation: check if client spoke "present" or digits
        client_heard = request.form.get("client_heard", "").strip().lower()
        if not res.success and ("present" in client_heard or "here" in client_heard or "yes" in client_heard):
            is_logged, status_msg = finalize_attendance(
                user_name=user,
                face_conf=state["face_conf"] / 100.0,
                voice_sim=1.0,
                method="Face + Vocal 'Present' + Liveness (Web Kiosk)",
                spoken_code_ok=1,
                heard_digits="PRESENT"
            )
            from core.verification_service import VerificationResult
            res = VerificationResult(
                success=is_logged,
                step="completed",
                user_name=user,
                face_conf=state["face_conf"] / 100.0,
                message=status_msg,
                heard_digits="PRESENT"
            )
        elif not res.success and client_heard:
            clean_expected = "".join([c for c in str(state["challenge_code"]) if c.isdigit()])
            clean_client = "".join([c for c in str(client_heard) if c.isdigit()])
            if clean_client == clean_expected:
                # Log success using client transcription
                is_logged, status_msg = finalize_attendance(
                    user_name=user,
                    face_conf=state["face_conf"] / 100.0,
                    voice_sim=1.0,
                    method="Face + Spoken Code + Liveness (Web Kiosk)",
                    spoken_code_ok=1,
                    heard_digits=clean_client
                )
                from core.verification_service import VerificationResult
                res = VerificationResult(
                    success=is_logged,
                    step="completed",
                    user_name=user,
                    face_conf=state["face_conf"] / 100.0,
                    message=status_msg,
                    heard_digits=clean_client
                )

        state["is_recording"] = False
        state["heard_digits"] = res.heard_digits

        if res.success:
            state["step"] = 4
            state["result_message"] = res.message
            return jsonify({
                "success": True,
                "message": res.message,
                "heard": res.heard_digits
            })
        else:
            if state["code_attempts"] >= MAX_CODE_ATTEMPTS:
                msg = f"{res.message}. Exceeded {MAX_CODE_ATTEMPTS} attempts. Restarting from Step 1."
                reset_state()
                state["result_message"] = msg
                return jsonify({
                    "success": False,
                    "message": msg,
                    "restart": True,
                    "heard": res.heard_digits
                })
            else:
                # Single-use code rule: generate a fresh random code for next attempt
                state["challenge_code"] = generate_random_code()
                state["code_created_at"] = time.time()
                state["result_message"] = f"{res.message}. New code issued (Attempt {state['code_attempts']}/{MAX_CODE_ATTEMPTS})."
                return jsonify({
                    "success": False,
                    "message": state["result_message"],
                    "heard": res.heard_digits,
                    "new_code": state["challenge_code"]
                })

    except Exception as e:
        state["is_recording"] = False
        return jsonify({"success": False, "error": str(e)})

# ----------------- PROTECTED MANAGEMENT ENDPOINTS (LOGIN REQUIRED) -----------------

@app.route('/api/enroll', methods=['POST'])
@login_required
def api_enroll():
    if not verify_csrf():
        return jsonify({"success": False, "error": "CSRF validation failed."}), 400

    data = request.get_json() or {}
    name = data.get("name", "").strip().upper().replace(" ", "_")
    if not name:
        return jsonify({"success": False, "error": "User name is required."})

    cap = get_camera()
    captured_faces = 0

    with camera_lock:
        start_time = time.time()
        while captured_faces < 5 and (time.time() - start_time < 8):
            ret, frame = cap.read()
            if not ret:
                break
            bbox = face_engine.detect_face_bbox(frame)
            if bbox is not None:
                x, y, w, h = bbox
                face_roi = frame[y:y+h, x:x+w]
                emb = face_engine.get_embedding(face_roi)
                if emb is not None:
                    save_face_embedding(name, emb)
                    captured_faces += 1
                    time.sleep(0.3)
            time.sleep(0.05)

    if captured_faces == 0:
        return jsonify({"success": False, "error": "No face detected in camera during enrollment."})

    reload_enrolled()
    actor = session['user']['username']
    log_audit(actor, "ENROLL_USER", f"Enrolled student '{name}' with {captured_faces} face templates.")

    return jsonify({
        "success": True,
        "message": f"Successfully enrolled {name} ({captured_faces} face templates saved)."
    })

@app.route('/api/delete_user', methods=['POST'])
@login_required
def api_delete_user():
    if not verify_csrf():
        return jsonify({"success": False, "error": "CSRF validation failed."}), 400

    # Role check: Teachers cannot delete students directly; requires admin role
    if session['user']['role'] != 'admin':
        return jsonify({
            "success": False,
            "error": "Permission denied: Only administrators can delete enrolled students. Please request administrator approval."
        }), 403

    data = request.get_json() or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"success": False, "error": "Name is required."})

    ok = delete_user(name)
    reload_enrolled()
    actor = session['user']['username']
    log_audit(actor, "DELETE_USER", f"Deleted user '{name}'.")
    return jsonify({"success": ok, "message": f"User {name} deleted." if ok else "User not found."})

@app.route('/api/logs')
@login_required
def api_logs():
    logs = get_attendance_logs()
    return jsonify({"logs": logs})

@app.route('/api/users')
@login_required
def api_users():
    users = get_all_users()
    return jsonify({"users": users})

@app.route('/api/export_csv')
@login_required
def api_export_csv():
    actor = session['user']['username']
    log_audit(actor, "EXPORT_CSV", "Exported attendance records to CSV.")
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    export_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"attendance_export_{today}.csv")
    export_attendance_to_csv(export_path)
    return send_file(export_path, as_attachment=True, download_name=f"attendance_{today}.csv")

# ----------------- ADMIN ONLY ENDPOINTS -----------------

@app.route('/api/teachers')
@role_required(['admin'])
def api_teachers():
    teachers = get_all_system_users()
    return jsonify({"users": teachers})

@app.route('/api/create_teacher', methods=['POST'])
@role_required(['admin'])
def api_create_teacher():
    if not verify_csrf():
        return jsonify({"success": False, "error": "CSRF validation failed."}), 400

    data = request.get_json() or {}
    username = data.get("username", "").strip().lower()
    password = data.get("password", "")
    role = data.get("role", "teacher").strip().lower()

    if not username or len(password) < 6:
        return jsonify({"success": False, "error": "Username required and password must be >= 6 characters."})

    existing = get_system_user(username)
    if existing:
        return jsonify({"success": False, "error": f"Username '{username}' already exists."})

    password_hash = generate_password_hash(password)
    create_system_user(username, password_hash, role=role)

    actor = session['user']['username']
    log_audit(actor, "CREATE_FACULTY", f"Created {role} account: '{username}'.")
    return jsonify({"success": True, "message": f"{role.capitalize()} '{username}' created successfully."})

@app.route('/api/audit_logs')
@role_required(['admin'])
def api_audit_logs():
    logs = get_audit_logs(limit=150)
    return jsonify({"audit_logs": logs})

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Biometric Attendance Web Server running at:")
    print("👉 http://localhost:5000")
    print("👉 http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
