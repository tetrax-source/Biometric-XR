"""
core/verification_service.py
Centralized multi-modal verification service for the Biometric Attendance System.
Enforces the single source of truth across Desktop GUI, Web Portal, and CLI:
Order:
  (1) Blink Liveness Detection (MediaPipe)
  (2) Face Recognition (FaceNet Deep Embeddings)
  (3) Spoken 4-Digit Challenge (Vosk ASR restricted digit grammar)
      [Optional] Speaker Acoustic Matching (MFCC if VOICE_MATCH_ENABLED is True)
  (4) Attendance Database Logging with daily duplicate check
"""

import time
import cv2
import numpy as np
from core.database import (
    record_attendance, get_all_face_embeddings, get_voice_embedding, log_audit
)
from core.face_engine import FaceEngine
from core.voice_engine import VoiceEngine, ChallengeSession
from liveness import check_blink
from config import (
    CAMERA_INDEX, FACE_THRESHOLD, VOICE_THRESHOLD,
    EAR_THRESHOLD, REQUIRED_BLINKS, SPOKEN_CODE_CHECK,
    VOICE_MATCH_ENABLED, CODE_LENGTH, CODE_EXPIRY_SECONDS,
    MAX_CODE_ATTEMPTS
)

class VerificationResult:
    def __init__(self, success: bool, step: str, user_name: str = None,
                 face_conf: float = 0.0, voice_sim: float = 0.0,
                 message: str = "", heard_digits: str = ""):
        self.success = success
        self.step = step                      # 'liveness', 'face', 'voice', or 'completed'
        self.user_name = user_name
        self.face_confidence = face_conf
        self.voice_similarity = voice_sim
        self.message = message
        self.heard_digits = heard_digits

    def to_dict(self):
        return {
            "success": self.success,
            "step": self.step,
            "user_name": self.user_name,
            "face_confidence": self.face_confidence,
            "voice_similarity": self.voice_similarity,
            "message": self.message,
            "heard_digits": self.heard_digits
        }

def finalize_attendance(user_name: str, face_conf: float, voice_sim: float,
                        method: str = "Face + Spoken Code + Liveness",
                        spoken_code_ok: int = 1, heard_digits: str = ""):
    """
    Finalizes attendance logging with duplicate verification check in database.
    Writes audit trail on both success and duplicate attempts.
    Returns: (is_logged: bool, status_message: str)
    """
    if not user_name or user_name == "Unknown":
        return False, "Cannot log attendance for unidentified user."

    is_logged, status_msg = record_attendance(
        user_name,
        method=method,
        face_conf=face_conf,
        voice_sim=voice_sim,
        spoken_code_ok=spoken_code_ok,
        heard_digits=heard_digits
    )

    action = "ATTENDANCE_SUCCESS" if is_logged else "ATTENDANCE_DUPLICATE"
    log_audit(
        username=user_name,
        action=action,
        details=f"Method: {method} | Spoken digits: '{heard_digits}' (OK={spoken_code_ok}) | {status_msg}"
    )

    return is_logged, status_msg

def process_voice_challenge(user_name: str, face_conf: float, audio,
                            expected_code: str, method: str = "Web Kiosk") -> VerificationResult:
    """
    Unified voice stage logic:
    1. Audio energy check (rejects silence/room noise)
    2. Spoken digit recognition via Vosk (must match expected code exactly in order)
    3. Optional speaker verification (MFCC) if VOICE_MATCH_ENABLED is True
    4. Attendance logging upon verification
    """
    voice_engine = VoiceEngine(threshold=VOICE_THRESHOLD)

    # 1. Energy check
    is_audible, rms = voice_engine.check_audio_energy(audio)
    if not is_audible:
        return VerificationResult(
            success=False,
            step="voice",
            user_name=user_name,
            face_conf=face_conf,
            message="Audio too quiet. Please speak closer to the microphone.",
            heard_digits=""
        )

    clean_expected = "".join([c for c in str(expected_code) if c.isdigit()])
    heard_digits = ""
    passed_code = True

    # 2. Spoken 4-Digit Challenge Check
    if SPOKEN_CODE_CHECK:
        passed_code, heard_digits = voice_engine.verify_spoken_code(audio, clean_expected)
        if not passed_code:
            expected_formatted = " ".join(list(clean_expected))
            heard_formatted = " ".join(list(heard_digits)) if heard_digits else "(none)"
            fail_msg = f"Heard {heard_formatted}, expected {expected_formatted}"
            log_audit(user_name, "SPOKEN_CODE_FAILED", fail_msg)
            return VerificationResult(
                success=False,
                step="voice",
                user_name=user_name,
                face_conf=face_conf,
                message=fail_msg,
                heard_digits=heard_digits
            )
    else:
        heard_digits = clean_expected

    # 3. Optional Speaker Verification (MFCC)
    voice_sim = 1.0
    if VOICE_MATCH_ENABLED:
        stored_voice = get_voice_embedding(user_name)
        if stored_voice is not None:
            query_mfcc = voice_engine.extract_mfcc_embedding(audio)
            passed_speaker, voice_sim = voice_engine.verify(query_mfcc, stored_voice)
            if not passed_speaker:
                fail_msg = f"Speaker voice mismatch ({voice_sim*100:.1f}%)"
                log_audit(user_name, "SPEAKER_MATCH_FAILED", fail_msg)
                return VerificationResult(
                    success=False,
                    step="voice",
                    user_name=user_name,
                    face_conf=face_conf,
                    voice_sim=voice_sim,
                    message=fail_msg,
                    heard_digits=heard_digits
                )

    # 4. Finalize attendance
    is_logged, status_msg = finalize_attendance(
        user_name=user_name,
        face_conf=face_conf,
        voice_sim=voice_sim,
        method=method,
        spoken_code_ok=int(passed_code),
        heard_digits=heard_digits
    )

    return VerificationResult(
        success=is_logged,
        step="completed",
        user_name=user_name,
        face_conf=face_conf,
        voice_sim=voice_sim,
        message=status_msg,
        heard_digits=heard_digits
    )

def run_cli_verification(timeout_face: int = 8, timeout_liveness: int = 8) -> VerificationResult:
    """Runs the unified verification pipeline in terminal/CLI mode."""
    face_engine = FaceEngine(threshold=FACE_THRESHOLD)
    voice_engine = VoiceEngine(threshold=VOICE_THRESHOLD)
    enrolled_faces = get_all_face_embeddings()

    if not enrolled_faces:
        print("[!] No users enrolled in the database. Please enroll users first via GUI or Web.")
        return VerificationResult(False, "face", message="No enrolled users found in database.")

    # ----------------- STEP 1: LIVENESS -----------------
    print("\n" + "=" * 55)
    print("STEP 1: LIVENESS VERIFICATION (Blink naturally)")
    print("=" * 55)
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
    print("\n" + "=" * 55)
    print("STEP 2: FACE RECOGNITION (FaceNet Deep Learning)")
    print("=" * 55)
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

    # ----------------- STEP 3: SPOKEN CODE VERIFICATION -----------------
    print("\n" + "=" * 55)
    print("STEP 3: SPOKEN 4-DIGIT CHALLENGE (Speech Recognition)")
    print("=" * 55)

    session = ChallengeSession(length=CODE_LENGTH)
    while session.has_attempts_left():
        session.record_attempt()
        print(f"\nVerification Code to Speak: [ {' '.join(list(session.code))} ]")
        print(f"Attempt {session.attempts}/{session.max_attempts} (Expires in {session.expiry_seconds}s)")
        input("Press ENTER, speak clearly into your microphone, and pronounce the digits...")

        if session.is_expired():
            print("[-] Challenge code expired. Restarting from Step 1.")
            return VerificationResult(False, "voice", user_name=recognized_name,
                                      message="Challenge code expired.")

        audio = voice_engine.record_audio()
        res = process_voice_challenge(
            user_name=recognized_name,
            face_conf=recognized_conf,
            audio=audio,
            expected_code=session.code,
            method="CLI (FaceNet + Spoken Code + Liveness)"
        )

        if res.success:
            print(f"[+] Verification passed! Heard: {res.heard_digits}")
            print(f"[*] Result: {res.message}")
            return res
        else:
            print(f"[-] {res.message}")
            # Generate new code for retry
            session.code = "".join([str(np.random.randint(0, 10)) for _ in range(CODE_LENGTH)])

    print(f"[-] Max attempts ({MAX_CODE_ATTEMPTS}) exceeded. Restarting verification from Step 1.")
    return VerificationResult(False, "voice", user_name=recognized_name,
                              message=f"Exceeded {MAX_CODE_ATTEMPTS} attempts.")
