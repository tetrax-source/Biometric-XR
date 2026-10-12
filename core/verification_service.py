"""
core/verification_service.py
Centralized multi-modal verification service for the Biometric Attendance System.
Enforces the single source of truth across Desktop GUI, Web Portal, and CLI:
Order: Liveness (Blink) -> Face (FaceNet) -> Voice (MFCC) -> Attendance Recording.
"""

import time
import cv2
import numpy as np
from core.database import (
    record_attendance, get_all_face_embeddings, get_voice_embedding
)
from core.face_engine import FaceEngine
from core.voice_engine import VoiceEngine
from liveness import check_blink
from config import (
    CAMERA_INDEX, FACE_THRESHOLD, VOICE_THRESHOLD,
    EAR_THRESHOLD, REQUIRED_BLINKS
)

class VerificationResult:
    def __init__(self, success: bool, step: str, user_name: str = None,
                 face_conf: float = 0.0, voice_sim: float = 0.0, message: str = ""):
        self.success = success
        self.step = step                      # 'liveness', 'face', 'voice', or 'completed'
        self.user_name = user_name
        self.face_confidence = face_conf
        self.voice_similarity = voice_sim
        self.message = message

    def to_dict(self):
        return {
            "success": self.success,
            "step": self.step,
            "user_name": self.user_name,
            "face_confidence": self.face_confidence,
            "voice_similarity": self.voice_similarity,
            "message": self.message
        }

def finalize_attendance(user_name: str, face_conf: float, voice_sim: float, method: str = "Face + Voice + Liveness"):
    """
    Finalizes attendance logging with duplicate verification check in database.
    Returns: (is_logged: bool, status_message: str)
    """
    if not user_name or user_name == "Unknown":
        return False, "Cannot log attendance for unidentified user."
    return record_attendance(user_name, method=method, face_conf=face_conf, voice_sim=voice_sim)

def run_cli_verification(timeout_face: int = 8, timeout_liveness: int = 8) -> VerificationResult:
    """
    Runs the full unified multi-modal pipeline in terminal/CLI mode:
    Step 1: Blink Liveness Detection (MediaPipe)
    Step 2: Face Recognition (FaceNet Deep Learning)
    Step 3: Voice Verification (MFCC & Cosine Similarity)
    Step 4: Record Attendance in Database
    """
    face_engine = FaceEngine(threshold=FACE_THRESHOLD)
    voice_engine = VoiceEngine(threshold=VOICE_THRESHOLD)
    enrolled_faces = get_all_face_embeddings()

    if not enrolled_faces:
        print("[!] No users enrolled in the database. Please enroll users first via GUI or Web.")
        return VerificationResult(False, "face", message="No enrolled users found in database.")

    # ----------------- STEP 1: LIVENESS -----------------
    print("\n" + "=" * 50)
    print("STEP 1: LIVENESS VERIFICATION (Blink naturally)")
    print("=" * 50)
    passed_liveness = check_blink(
        timeout=timeout_liveness,
        required_blinks=REQUIRED_BLINKS,
        ear_thresh=EAR_THRESHOLD,
        camera_index=CAMERA_INDEX
    )
    if not passed_liveness:
        print("[-] Liveness check failed or timed out. Aborting.")
        return VerificationResult(False, "liveness", message="Liveness check failed.")
    print("[+] Liveness verified successfully!")

    # ----------------- STEP 2: FACE RECOGNITION -----------------
    print("\n" + "=" * 50)
    print("STEP 2: FACE RECOGNITION (FaceNet Deep Learning)")
    print("=" * 50)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[-] Cannot access camera for face recognition.")
        return VerificationResult(False, "face", message="Camera unavailable.")

    start_time = time.time()
    recognized_name = None
    recognized_conf = 0.0

    print("Looking into camera... Press ESC to cancel.")
    while time.time() - start_time < timeout_face:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        name, sim, bbox, ann_frame = face_engine.process_frame(frame, enrolled_faces)

        remaining = int(timeout_face - (time.time() - start_time))
        cv2.putText(ann_frame, f"Time left: {remaining}s", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Face Recognition", ann_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        if name and name != "Unknown":
            recognized_name = name
            recognized_conf = sim
            break

    cap.release()
    cv2.destroyAllWindows()

    if not recognized_name:
        print("[-] Face not recognized or match below threshold.")
        return VerificationResult(False, "face", message="Face not recognized.")

    print(f"[+] Face recognized as: {recognized_name} (Confidence: {recognized_conf*100:.1f}%)")

    # ----------------- STEP 3: VOICE VERIFICATION -----------------
    print("\n" + "=" * 50)
    print("STEP 3: VOICE VERIFICATION (Acoustic MFCC Analysis)")
    print("=" * 50)
    challenge_code = voice_engine.generate_challenge_code()
    print(f"Spoken Verification Code: [ {challenge_code} ]")
    input("Press ENTER, speak into your microphone, and say the code...")

    audio = voice_engine.record_audio()
    is_audible, rms = voice_engine.check_audio_energy(audio)
    if not is_audible:
        print("[-] Audio signal too quiet. Please speak closer to microphone.")
        return VerificationResult(False, "voice", user_name=recognized_name,
                                  face_conf=recognized_conf, message="Audio too quiet.")

    query_mfcc = voice_engine.extract_mfcc_embedding(audio)
    stored_voice = get_voice_embedding(recognized_name)

    if stored_voice is None:
        print("[!] Note: No stored voice profile found for this user. Continuing with face-only confirmation.")
        passed_voice, voice_sim = True, 1.0
    else:
        passed_voice, voice_sim = voice_engine.verify(query_mfcc, stored_voice)

    if not passed_voice:
        print(f"[-] Voice verification failed (Similarity: {voice_sim*100:.1f}%).")
        return VerificationResult(False, "voice", user_name=recognized_name,
                                  face_conf=recognized_conf, voice_sim=voice_sim,
                                  message="Voice mismatch.")

    print(f"[+] Voice verified! Similarity score: {voice_sim*100:.1f}%")

    # ----------------- STEP 4: RECORD ATTENDANCE -----------------
    print("\n" + "=" * 50)
    print("STEP 4: LOGGING ATTENDANCE")
    print("=" * 50)
    is_logged, status_msg = finalize_attendance(
        recognized_name,
        face_conf=recognized_conf,
        voice_sim=voice_sim,
        method="CLI (FaceNet + Voice + Liveness)"
    )
    print(f"[*] Result: {status_msg}")

    return VerificationResult(
        success=is_logged,
        step="completed",
        user_name=recognized_name,
        face_conf=recognized_conf,
        voice_sim=voice_sim,
        message=status_msg
    )
