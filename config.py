"""
config.py
Central configuration for the Biometric Attendance System.
Contains paths, hardware settings, recognition thresholds, and security keys.
All values have safe development defaults and can be overridden via environment variables.
"""

import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Database path
DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE_DIR, "attendance_system.db"))

# Camera settings
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", 0))

# Biometric recognition thresholds
# NOTE: Do NOT change these thresholds without evaluating FAR/FRR impact.
FACE_THRESHOLD = float(os.environ.get("FACE_THRESHOLD", 0.55))    # Minimum cosine similarity for FaceNet
VOICE_THRESHOLD = float(os.environ.get("VOICE_THRESHOLD", 0.50))  # Minimum cosine similarity for MFCC
EAR_THRESHOLD = float(os.environ.get("EAR_THRESHOLD", 0.22))      # Eye Aspect Ratio threshold for blink detection
REQUIRED_BLINKS = int(os.environ.get("REQUIRED_BLINKS", 1))       # Number of blinks required for liveness

# Audio settings
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION = 3.0  # seconds
MIN_AUDIO_RMS = 0.005  # Energy threshold to reject silence

# Security and Session
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-biometric-key-2026-viva")
SESSION_TIMEOUT_MINUTES = int(os.environ.get("SESSION_TIMEOUT_MINUTES", 30))

# Model & data paths
DATA_DIR = os.path.join(BASE_DIR, "data")
FACE_LANDMARKER_PATH = os.path.join(DATA_DIR, "face_landmarker.task")
HAAR_FACE_PATH = os.path.join(DATA_DIR, "haarcascade_frontalface_default.xml")
HAAR_EYE_PATH = os.path.join(DATA_DIR, "haarcascade_eye.xml")
