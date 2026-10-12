# Project Upgrade Progress

This document tracks all changes made during the system upgrade. Each phase records the status, files changed, and what to test.

---

## Phase 1: Cleanup & Centralized Configuration
- **Status**: DONE
- **Files Changed**:
  - `requirements.txt`: Added missing dependencies (`Flask>=3.0.0`, `Werkzeug>=3.0.0`, `pytest>=8.0.0`, `matplotlib>=3.8.0`).
  - `archive/`: Created folder and moved redundant/legacy scripts (`01.py`, `02.py`, `03.py`, `auto_enroll.py`, `auto_face_enroll.py`, `auto_voice_enroll.py`, `face_collect.py`, `face_enroll.py`, `face_train.py`, `emotion_only.py`, and legacy prototypes).
  - `config.py`: Created centralized configuration defining camera index, DB path, thresholds (face 0.55, voice 0.50, ear 0.22, blinks 1), session timeout, and secret key.
  - `core/database.py`: Updated to use `DB_PATH` from `config.py`.
  - `app_web.py`: Updated to use thresholds, secret key, and camera index from `config.py`.
  - `app_gui.py`: Updated to use thresholds and camera index from `config.py`.
  - `secure_verify.py`: Updated to use thresholds and camera index from `config.py`.
- **What to Test**:
  - Run `python -c "import config, app_web, app_gui, secure_verify"` to verify all entry modules import and configure properly.
  - Check that no archived files are imported by active modules.

## Phase 2: Database Safety & Schema Upgrades
- **Status**: DONE
- **Files Changed**:
  - `core/database.py`:
    - Enabled `PRAGMA foreign_keys = ON;` in `get_connection()`.
    - Replaced `pickle` with `serialize_embedding` (`tobytes()`) and `deserialize_embedding` (`np.frombuffer()`) using raw `float32` binary buffers.
    - Added one-time migration function `migrate_pickled_embeddings()`, converting all existing records in `attendance_system.db`.
    - Verified bit-for-bit equivalence ($0.0$ error diff) between backup database and migrated database.
    - Added composite index `idx_attendance_user_date` on `attendance_logs(user_name, date)`.
    - Added `system_users` table for teachers and administrators (`id`, `username UNIQUE`, `password_hash`, `role CHECK ('admin', 'teacher')`, `created_at`).
    - Added `audit_log` table (`id`, `username`, `action`, `details`, `timestamp`).
- **What to Test**:
  - Verified embedding shapes (`(128,) float32`, `(20,) float32`) and numerical equivalence (`Face diff: 0.0`, `Voice diff: 0.0`).
  - Checked `sqlite_master` to ensure tables `system_users`, `audit_log`, and index `idx_attendance_user_date` exist.

## Phase 3: Unified CLI and Shared Decision Pipeline
- **Status**: DONE
- **Files Changed**:
  - `core/verification_service.py`: Created shared verification service providing `finalize_attendance()` and `run_cli_verification()`. Enforces identical multi-modal rules across Desktop GUI, Web, and CLI:
    1. Liveness (MediaPipe blink detection)
    2. Face recognition (FaceNet deep embeddings, threshold `0.55`)
    3. Voice verification (MFCC acoustic similarity, threshold `0.50`)
    4. Attendance recording with daily duplicate check
  - `secure_verify.py`: Updated to use `core.verification_service` (FaceNet + MFCC + Liveness) instead of legacy LBPH `face_verify.py`.
  - `app_web.py`: Updated to call `finalize_attendance()` for shared decision logic.
  - `app_gui.py`: Updated to call `finalize_attendance()` for shared decision logic.
- **What to Test**:
  - Run `python -c "import core.verification_service, secure_verify, app_web, app_gui"` to ensure unified service functions seamlessly across all environments.
  - Ensure CLI pipeline no longer references `trainer.yml` or `face_verify.py`.

## Phase 4: Role-Based Access Control, Authentication, and Audit Trail
- **Status**: DONE
- **Files Changed**:
  - `create_admin.py`: Created interactive CLI tool to securely create or update administrator credentials without hardcoded passwords.
  - `templates/login.html`: Created dedicated faculty/admin login page with Bootstrap 5 Dark theme and CSRF validation.
  - `templates/index.html`: Updated UI to distinguish public Kiosk mode from authenticated Teacher/Admin portal, adding faculty management and audit log views.
  - `app_web.py`:
    - Added `@login_required` and `@role_required` decorators protecting sensitive endpoints (`/api/enroll`, `/api/delete_user`, `/api/export_csv`, `/api/logs`, `/api/teachers`, `/api/create_teacher`, `/api/audit_logs`).
    - Added CSRF token validation on mutating actions (`verify_csrf()`).
    - Added 30-minute inactivity session expiration in `@app.before_request`.
    - Integrated audit logging (`log_audit`) tracking every login, logout, enrollment, user deletion, teacher creation, and report export.
    - Preserved public attendance check-in as open "Kiosk mode" on `/`.
- **What to Test**:
  - Unauthenticated access to `/api/logs` returns `401 Unauthorized`.
  - Unauthenticated access to `/` serves Kiosk check-in mode.
  - Successful login as admin unlocks student enrollment, log viewing, faculty management, and audit trail tabs.
  - Verified CSRF protection on forms and API mutations.

## Phase 5: Biometric Performance Evaluation (FAR / FRR / EER)
- **Status**: DONE
- **Files Changed**:
  - `evaluate.py`: Created evaluation script measuring False Accept Rate (FAR), False Reject Rate (FRR), and Equal Error Rate (EER) across thresholds from `0.30` to `0.90` for both FaceNet and Voice MFCC.
  - `evaluation_results.csv`: Saved threshold-by-threshold FAR/FRR metrics for viva reporting.
  - `evaluation_roc_curves.png`: Generated ROC / FAR-FRR curves plot using matplotlib with threshold indicators.
  - `eval_data/README.txt`: Created student instructions explaining genuine vs impostor data collection and photo/screen/video spoof test protocols.
  - `spoof_log.csv`: Created template tracking manual liveness test outcomes against printed photos, smartphone screen replays, and live human faces.
- **What to Test**:
  - Ran `python evaluate.py`: Generated `evaluation_results.csv` and `evaluation_roc_curves.png`.
  - Confirmed configured thresholds (0.55 face, 0.50 voice) produce reliable baseline FAR/FRR metrics.

## Phase 6: Automated Testing Suite
- **Status**: DONE
- **Files Changed**:
  - `tests/test_database.py`: Unit tests for database table creation, foreign keys, embedding serialization roundtrip (float32 byte-exact), user storage, voice upsert, duplicate attendance prevention, and audit logging.
  - `tests/test_biometrics.py`: Unit tests for cosine similarity properties, geometric Eye Aspect Ratio (EAR) on dummy landmark coordinates (open vs closed eyes), FaceEngine matching, and VoiceEngine matching.
  - `tests/test_auth_routes.py`: Integration tests for password hashing, kiosk public access, unauthenticated endpoint protection (401), login flow, role-based access control (Teacher vs Admin 403), and CSRF token verification.
- **What to Test**:
  - Ran `python -m pytest tests/ -v`: All 17 tests passed (100% success rate).

## Phase 7: Voice Challenge Speech-to-Text Integration
- **Status**: SKIPPED (Offline STT package and model not installed; zero download rule observed)
- **Reason**:
  - Checked for offline speech-to-text libraries (`vosk`, `whisper`, `speech_recognition`, `pocketsphinx`, `torch`); none are currently installed.
  - In adherence to the strict "no large downloads / unstable internet" constraint, external model bundle downloads were bypassed.
- **Files Changed**:
  - `app_web.py`: Updated voice verification status text from "Speak verification code below" to "Speak prompt into microphone for voice biometric matching" so the UI does not falsely claim semantic digit verification.
  - `templates/index.html`: Updated UI label to "Speak this prompt into microphone (MFCC acoustic matching)".
  - `app_gui.py`: Updated UI label to "Press 'Speak' & speak prompt for acoustic matching".
- **Instructions for Future Installation**:
  To add semantic 4-digit spoken challenge transcription later:
  1. Install Vosk: `pip install vosk`
  2. Download the lightweight offline model (~40 MB):
     URL: `https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip`
  3. Extract to: `E:\Aswin\data\vosk-model`
  4. In `core/voice_engine.py`, load `vosk.Model("data/vosk-model")` and check `json.loads(rec.Result())['text']` against `challenge_code`.

## Phase 8: Web Audio from Browser with Server Fallback
- **Status**: DONE
- **Files Changed**:
  - `config.py`: Added `USE_BROWSER_AUDIO` configuration flag (default: `True`).
  - `app_web.py`:
    - Updated `/api/verify_voice` to accept multipart uploaded audio blobs from the browser.
    - Decodes incoming web audio (WebM/WAV) using `librosa.load(audio_buf, sr=16000, mono=True)` in memory.
    - Added automatic fallback to the server microphone if no client audio file is received or if client permissions fail.
  - `templates/index.html`:
    - Updated `recordVoice()` to use `navigator.mediaDevices.getUserMedia({ audio: true })` and `MediaRecorder`.
    - Packages 3-second audio recording into FormData and POSTs to `/api/verify_voice`.
- **What to Test**:
  - Tested Flask audio upload decoding in memory.
  - All 17 automated tests continue to pass with 100% success rate.

## Phase 9: Project Documentation & Viva Preparation
- **Status**: DONE
- **Files Changed**:
  - `README.md`: Updated with full repository structure, run commands for all 3 interfaces, admin creation guide, testing commands, evaluation metrics, and configuration options.
  - `FUTURE_SCOPE.md`: Created detailed roadmap covering timetable integration, leave management, SMS alerts, PostgreSQL migration, mobile apps, and 3D anti-spoofing.
  - `SECURITY_NOTES.md`: Created comprehensive security analysis documenting solved vulnerabilities (pickle removal, RBAC, session management, CSRF) and real-world compliance guidelines (DPDP Act 2023 / GDPR).
  - `VIVA_NOTES.md`: Created complete viva guide covering pipeline architecture and the top 10 external examiner questions with clear, simple answers.
- **What to Test**:
  - Review markdown documents for accuracy, readability, and consistency across modules.

## Phase 1 (New Upgrade): Vosk Offline Speech Tool & Configuration
- **Status**: DONE
- **Files Changed**:
  - `requirements.txt`: Added `vosk>=0.3.45`.
  - `models/vosk-model-small-en-us-0.15`: Downloaded and extracted lightweight English acoustic model (~40 MB).
  - `config.py`: Added configuration flags:
    - `SPOKEN_CODE_CHECK = True`
    - `VOICE_MATCH_ENABLED = False` (MFCC matching bypassed by default)
    - `CODE_LENGTH = 4`
    - `CODE_EXPIRY_SECONDS = 60`
    - `MAX_CODE_ATTEMPTS = 3`
    - `VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15"`
- **Fallback Verification**:
  - Verified Vosk initializes cleanly from the extracted model path.
  - Added safety check: If model folder is missing or cannot load, the system falls back gracefully to bypassing spoken code verification with a warning, never crashing.
- **What to Test**:
  - Run `python -c "import vosk, config; m = vosk.Model(config.VOSK_MODEL_PATH); print('OK')"` to verify model loads.

## Phase 2 (New Upgrade): Face-Only Enrollment Transition
- **Status**: DONE
- **Files Changed**:
  - `app_web.py`: Removed voice-recording call from `/api/enroll` endpoint. Enrollment now captures 5 FaceNet embeddings from camera and updates enrolled cache without requiring voice.
  - `app_gui.py`: Removed Voice Profile Enrollment card from the Enrollment UI; enrollment now executes face-only capture. Kept existing voice functions and existing database rows untouched.
  - `templates/index.html`: Updated enrollment UI labels and status alerts to indicate face-only capture.
- **What to Test**:
  - Test enrolling a new user via web API or GUI; verify only face samples are captured and stored, with no prompt or requirement for voice.

## Phase 3 (New Upgrade): Spoken Code Verification Engine
- **Status**: DONE
- **Files Changed**:
  - `core/voice_engine.py`:
    - Added `verify_spoken_code(audio, expected_code)` with restricted Kaldi digit grammar (`zero`, `oh`, `one`..`nine`, `[unk]`).
    - Maps spoken `"oh"` and `"zero"` to `"0"`.
    - Converts input audio to 16 kHz mono 16-bit signed PCM format.
    - Added `ChallengeSession` for tracking code generation, timestamp expiration, and retry limits.
    - Added singleton `get_vosk_model()` with graceful error recovery.
  - `core/verification_service.py`:
    - Updated `process_voice_challenge()` to run: (1) Audio energy check, (2) Vosk spoken digit recognition matching expected challenge code, (3) Optional MFCC matching if `VOICE_MATCH_ENABLED = True`, and (4) Idempotent database attendance finalization.
- **What to Test**:
  - Unit tested with digit grammar simulation ("four two nine one" -> "4291", "oh zero five three" -> "0053", wrong order "one two three four" -> failed).

## Phase 4 (New Upgrade): Dynamic Random Code Challenge & Kiosk Privacy
- **Status**: DONE
- **Files Changed**:
  - `app_web.py`:
    - Generates dynamic 4-digit code upon successful FaceNet recognition.
    - Enforces 60-second expiration (`CODE_EXPIRY_SECONDS`): resets to Step 1 if expired.
    - Enforces maximum 3 attempts (`MAX_CODE_ATTEMPTS`): resets to Step 1 if exceeded.
    - Generates a fresh random code on each failed attempt (single-use code rule).
    - Anonymized `/api/status`: strips `user_name` and logs from public API payload.
  - `templates/index.html`:
    - Public Kiosk Step 3 displays exclusively: random 4-digit code challenge, digits heard, attempt counter, and pass/fail alert. Zero student names or log records are shown.
  - `app_gui.py`:
    - Updated Step 3 voice challenge card to show dynamic 4-digit prompt and attempt count.
  - `secure_verify.py` (CLI):
    - Uses interactive terminal challenge loop with `ChallengeSession` and automatic retry.
- **What to Test**:
  - Stand in front of camera, pass blink + face, observe random 4-digit challenge on screen.
  - Speak digits and verify kiosk shows heard digits and success alert without displaying student list.

## Phase 5 (New Upgrade): Database Schema Migration & Audit Logging
- **Status**: DONE
- **Files Changed**:
  - `core/database.py`:
    - Added safe automatic migration: checks `PRAGMA table_info(attendance_logs)` and executes `ALTER TABLE attendance_logs ADD COLUMN spoken_code_ok INTEGER DEFAULT 1;` and `ALTER TABLE attendance_logs ADD COLUMN heard_digits TEXT DEFAULT '';` if columns are missing.
    - Updated `record_attendance()` to record `spoken_code_ok` and `heard_digits`.
    - Updated `export_attendance_to_csv()` to include `Spoken_Code_OK` and `Heard_Digits` columns in generated reports.
  - `core/verification_service.py`:
    - Logs `SPOKEN_CODE_FAILED`, `SPEAKER_MATCH_FAILED`, and `ATTENDANCE_SUCCESS` events with spoken digit metadata to `audit_log`.
- **What to Test**:
  - Check `attendance_logs` schema to verify columns exist.
  - Verify attendance records contain `spoken_code_ok` (1) and `heard_digits`.

## Phase 6 (New Upgrade): Automated Test Suite Expansion
- **Status**: DONE
- **Files Changed**:
  - `tests/test_biometrics.py`:
    - Added `test_challenge_session_lifecycle`: tests single-use code creation, 60s expiration, and 3-attempt limit.
    - Added `test_spoken_code_digit_grammar_parsing`: tests exact matches, "oh" as 0, wrong digit order, missing digits, extra digits, and silence.
    - Added `test_spoken_code_disabled_flag`: tests graceful bypass when `SPOKEN_CODE_CHECK = False`.
    - Added `test_face_only_enrollment_does_not_require_voice`: verifies enrollment stores face vectors without requiring voice.
- **What to Test**:
  - Ran `python -m pytest tests/ -v`: All 21 tests passed (100% success rate).

## Phase 7 (New Upgrade): Security Housekeeping & .gitignore
- **Status**: DONE
- **Files Changed**:
  - `.gitignore`: Created repository ignore file excluding database files (`*.db`, `*.db.bak`), neural weights (`models/`), audio caches (`voices/`, `voice_db/`), image caches (`images/`), test datasets (`eval_data/`), and environment files (`.env`).
  - `app_web.py`: Added console startup check and prominent warning banner if the default administrator account credentials (`admin`/`admin123`) are still active, prompting execution of `python create_admin.py`.
- **What to Test**:
  - Check console startup output when starting server.
  - Check `git status` to ensure binary databases and models are ignored.

## Phase 8 (New Upgrade): Documentation & Viva Defense Preparation
- **Status**: DONE
- **Files Changed**:
  - `README.md`: Updated architecture, pipeline diagrams, privacy kiosk details, and test instructions.
  - `SECURITY_NOTES.md`: Documented anti-spoofing mechanism (face identifies WHO, blink + spoken code proves live presence NOW) and local microphone privacy guarantees.
  - `VIVA_NOTES.md`: Added 5 detailed viva Q&A pairs (Q11-Q15) covering dynamic codes, face-only enrollment rationale, offline Vosk advantages, missing model fallback, and speech recognition vs speaker verification trade-offs.
- **What to Test**:
  - Review markdown documents for presentation clarity.

---
