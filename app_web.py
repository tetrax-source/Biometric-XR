"""
app_web.py
Flask Web Application for Multi-Modal Biometric Attendance System.
Runs on http://localhost:5000 / http://127.0.0.1:5000.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import time
import threading
import datetime
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, send_file

# Core modules
from core.database import (
    init_db, add_or_get_user, save_face_embedding, save_voice_embedding,
    get_all_face_embeddings, get_voice_embedding, get_all_users,
    record_attendance, get_attendance_logs, export_attendance_to_csv
)
from core.liveness import LivenessDetector
from core.face_engine import FaceEngine
from core.voice_engine import VoiceEngine

from config import (
    CAMERA_INDEX, FACE_THRESHOLD, VOICE_THRESHOLD,
    EAR_THRESHOLD, REQUIRED_BLINKS, SECRET_KEY
)

app = Flask(__name__)
app.secret_key = SECRET_KEY

# Initialize DB & Biometric Engines
init_db()
liveness_detector = LivenessDetector(ear_threshold=EAR_THRESHOLD, required_blinks=REQUIRED_BLINKS)
face_engine = FaceEngine(threshold=FACE_THRESHOLD)
voice_engine = VoiceEngine(threshold=VOICE_THRESHOLD)

# State management
camera_lock = threading.Lock()
camera = None
enrolled_faces = get_all_face_embeddings()

state = {
    "step": 1,                     # 1: Liveness, 2: Face, 3: Voice, 4: Attendance Logged
    "liveness_passed": False,
    "blink_count": 0,
    "ear": 0.0,
    "user_name": None,
    "face_conf": 0.0,
    "voice_score": 0.0,
    "voice_status": "Waiting for face identification...",
    "challenge_code": "",
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
    state["voice_score"] = 0.0
    state["voice_status"] = "Waiting for face identification..."
    state["challenge_code"] = ""
    state["result_message"] = "Looking for natural blink..."
    state["is_recording"] = False

def reload_enrolled():
    global enrolled_faces
    enrolled_faces = get_all_face_embeddings()

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
                code = voice_engine.generate_challenge_code()
                state["challenge_code"] = code
                state["voice_status"] = f"Face identified as {name}. Speak the challenge code below."
                state["result_message"] = f"Face recognized ({state['face_conf']}%). Complete voice verification."
            elif name == "Unknown":
                state["result_message"] = "Face not recognized in database."

        ret_encode, buffer = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret_encode:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

# ----------------- WEB ROUTES -----------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    return jsonify(state)

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

    state["is_recording"] = True
    user = state["user_name"]

    try:
        audio = voice_engine.record_audio(duration=3.0)
        is_audible, rms = voice_engine.check_audio_energy(audio)
        if not is_audible:
            state["is_recording"] = False
            state["voice_status"] = "Voice too quiet. Please speak louder into the microphone."
            return jsonify({"success": False, "error": "Low audio energy. Please speak louder."})

        query_mfcc = voice_engine.extract_mfcc_embedding(audio)
        stored_voice = get_voice_embedding(user)

        if stored_voice is None:
            passed, score = True, 1.0
        else:
            passed, score = voice_engine.verify(query_mfcc, stored_voice)

        state["is_recording"] = False
        state["voice_score"] = round(score * 100, 1)

        if passed:
            state["step"] = 4
            state["voice_status"] = f"Voice verified ({state['voice_score']}% match)!"
            logged, log_msg = record_attendance(
                user,
                method="Face + Voice + Liveness (Web)",
                face_conf=state["face_conf"] / 100.0,
                voice_sim=score
            )
            state["result_message"] = log_msg
            return jsonify({"success": True, "logged": logged, "message": log_msg, "score": state["voice_score"]})
        else:
            state["voice_status"] = f"Voice mismatch ({state['voice_score']}%). Please try again."
            return jsonify({"success": False, "message": "Voice verification failed. Try again.", "score": state["voice_score"]})

    except Exception as e:
        state["is_recording"] = False
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/enroll', methods=['POST'])
def api_enroll():
    data = request.get_json() or {}
    name = data.get("name", "").strip().upper().replace(" ", "_")
    if not name:
        return jsonify({"success": False, "error": "User name is required."})

    cap = get_camera()
    captured_faces = 0

    # Capture 5 face frames
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

    # Record voice sample
    try:
        audio = voice_engine.record_audio(duration=3.0)
        mfcc = voice_engine.extract_mfcc_embedding(audio)
        if mfcc is not None:
            save_voice_embedding(name, mfcc)
    except Exception as e:
        pass

    reload_enrolled()
    return jsonify({
        "success": True,
        "message": f"Successfully enrolled {name} ({captured_faces} face templates saved)."
    })

@app.route('/api/logs')
def api_logs():
    logs = get_attendance_logs()
    return jsonify({"logs": logs})

@app.route('/api/users')
def api_users():
    users = get_all_users()
    return jsonify({"users": users})

@app.route('/api/export_csv')
def api_export_csv():
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    export_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"attendance_export_{today}.csv")
    export_attendance_to_csv(export_path)
    return send_file(export_path, as_attachment=True, download_name=f"attendance_{today}.csv")

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Biometric Attendance Web Server running at:")
    print("👉 http://localhost:5000")
    print("👉 http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
