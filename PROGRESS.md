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

---
