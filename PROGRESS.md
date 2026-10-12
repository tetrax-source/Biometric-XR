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

---
