"""
secure_verify.py
Unified Terminal / CLI verification runner for the Biometric Attendance System.
Uses core.face_engine (FaceNet) and core.voice_engine (MFCC) via core.verification_service.
Execution Order: Liveness (Blink) -> Face Recognition -> Voice Verification -> Attendance Log.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from core.verification_service import run_cli_verification, finalize_attendance
from core.database import init_db

def main():
    init_db()
    result = run_cli_verification()
    if result.success:
        print("\n[SUCCESS] Multi-Modal Attendance Complete.")
    else:
        print(f"\n[INFO] Pipeline stopped: {result.message}")

if __name__ == "__main__":
    main()
