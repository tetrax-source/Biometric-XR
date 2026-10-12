# AI Multi-Modal Biometric Attendance & Security System

A production-grade, multi-modal biometric attendance and access control system integrating **Eye Blink Liveness Detection (MediaPipe)**, **Deep Face Recognition (FaceNet)**, and **Acoustic Voice Verification (MFCC)** with Role-Based Access Control (RBAC), SQLite relational storage, and comprehensive security audit logging.

---

## 🌟 Key Highlights

- **Triple-Factor Verification**: Liveness (Blink) $\rightarrow$ Face Recognition (FaceNet) $\rightarrow$ Voice Acoustics (MFCC).
- **Role-Based Access Control (RBAC)**:
  - **Public Kiosk Mode**: Open student check-in without exposing administrative controls or records.
  - **Faculty Portal**: Password-protected dashboard for teachers to enroll students, view attendance logs, and export CSV reports.
  - **Administrator Portal**: Manage faculty credentials, view security audit trails, and oversee system configuration.
- **Enterprise-Grade Security**:
  - Zero-pickle raw `float32` byte serialization for all biometric templates.
  - Salted cryptographic password hashing via `werkzeug.security`.
  - 30-minute inactivity session expiration and CSRF protection on mutating actions.
  - Relational database foreign keys (`PRAGMA foreign_keys = ON;`) and composite indexing.
- **Multiple Operational Interfaces**:
  - **Modern Web Portal** (Flask, Bootstrap 5 Dark Theme, MJPEG stream, client-side Web Audio capture).
  - **Desktop GUI** (CustomTkinter, multi-tab workstation dashboard).
  - **Unified Terminal CLI** (`main.py --cli` or `secure_verify.py`).

---

## 📁 Repository Structure

```
E:\Aswin\
├── config.py                       # Centralized configuration (thresholds, DB path, camera index)
├── create_admin.py                 # Interactive CLI tool to create faculty/admin accounts
├── evaluate.py                     # Biometric evaluation script (FAR / FRR / EER / ROC curve)
├── main.py                         # Unified master entry point (--web, --cli, or default GUI)
├── app_web.py                      # Flask web server (Kiosk mode, RBAC, REST endpoints)
├── app_gui.py                      # CustomTkinter modern desktop GUI application
├── secure_verify.py                # Standalone unified CLI verification runner
├── run_web.bat                     # 1-click Windows launcher for Web Kiosk
├── attendance_system.db            # SQLite relational database (users, embeddings, logs, audit)
├── requirements.txt                # Python package dependency manifest
├── PROGRESS.md                     # Full upgrade log tracking all development phases
├── VIVA_NOTES.md                   # Architecture overview & Top 10 viva questions/answers
├── SECURITY_NOTES.md               # In-depth security analysis & DPDP Act compliance review
├── FUTURE_SCOPE.md                 # Institutional roadmap & enterprise scale features
│
├── core/                           # Core modular engine package
│   ├── database.py                 # SQLite manager, safe serialization, CRUD, audit logging
│   ├── face_engine.py              # FaceNet deep embedding engine via DeepFace + ONNX
│   ├── liveness.py                 # MediaPipe 1.x Tasks API Eye Aspect Ratio (EAR) detector
│   ├── migration.py                # Legacy flat-file data migrator to SQLite
│   ├── verification_service.py     # Single source of truth multi-modal decision service
│   └── voice_engine.py             # 20-band MFCC acoustic feature extractor and matcher
│
├── data/                           # Neural network weights and cascades
│   ├── face_landmarker.task        # MediaPipe 1.x FaceLandmarker model bundle
│   ├── haarcascade_frontalface_default.xml # OpenCV frontal face detector
│   └── haarcascade_eye.xml         # OpenCV eye detector (fallback)
│
├── eval_data/                      # Biometric benchmark evaluation samples
│   ├── README.txt                  # Instructions for gathering genuine/impostor/spoof samples
│   ├── genuine/                    # Enrolled student test samples
│   └── impostor/                   # Unauthorized test samples
│
├── templates/                      # Flask HTML templates
│   ├── index.html                  # Bootstrap 5 dark web dashboard & kiosk interface
│   └── login.html                  # Secure faculty & administrator login portal
│
├── tests/                          # Automated pytest test suite (17 tests)
│   ├── test_auth_routes.py         # Flask RBAC, session timeout, and CSRF tests
│   ├── test_biometrics.py          # Cosine similarity and EAR geometric calculation tests
│   └── test_database.py            # SQLite schema, duplicate rule, and serialization tests
│
└── archive/                        # Preserved legacy scripts and prototypes
```

---

## 🚀 Quick Start Guide

### 1. Installation

Ensure you have Python 3.10+ installed. Install all required dependencies:

```powershell
pip install -r requirements.txt
```

### 2. Create the First Administrator Account

Run the interactive admin creation tool (credentials are hashed and stored securely):

```powershell
python create_admin.py
```
*Prompt:* Enter username (e.g. `admin`) and a password (minimum 6 characters).

---

## 💻 Running the Application

### Option A: Web Portal & Attendance Kiosk (Recommended)

Start the web server:
```powershell
python main.py --web --port 5000
# Or double-click:
.\run_web.bat
```
- **Public Kiosk Mode**: Open `http://localhost:5000` in any web browser. Anyone can stand in front of the camera, blink, and complete voice verification.
- **Faculty / Admin Sign-In**: Click **Faculty / Admin Login** in the top navigation bar to sign in. Once authenticated, the dashboard unlocks student enrollment, log viewing, CSV export, faculty account management, and security audit trails.

### Option B: Desktop GUI Application

Launch the desktop workstation GUI:
```powershell
python main.py
# Or directly:
python app_gui.py
```

### Option C: Unified Terminal / CLI Mode

Run verification directly in your terminal:
```powershell
python main.py --cli
# Or directly:
python secure_verify.py
```

---

## 🧪 Testing & Evaluation

### Running Automated Unit & Integration Tests

Execute the 17-test automated test suite using `pytest`:

```powershell
python -m pytest tests/ -v
```
Verifies database integrity, duplicate attendance prevention, bit-exact embedding serialization, EAR geometric calculations, role-based access control, and CSRF protection.

### Running Biometric Performance Evaluation (FAR / FRR / EER)

Generate academic evaluation metrics for your viva report:

```powershell
python evaluate.py
```
- Measures False Accept Rate (FAR), False Reject Rate (FRR), and Equal Error Rate (EER) across thresholds from $0.30$ to $0.90$.
- Saves detailed metrics to `evaluation_results.csv`.
- Generates performance ROC curves to `evaluation_roc_curves.png`.

---

## 🔒 Configurable System Parameters (`config.py`)

All core parameters can be customized directly in `config.py` or overridden via environment variables:

| Setting | Default | Description |
| :--- | :---: | :--- |
| `CAMERA_INDEX` | `0` | Hardware index of the connected webcam |
| `FACE_THRESHOLD` | `0.55` | Minimum cosine similarity for FaceNet matching |
| `VOICE_THRESHOLD` | `0.50` | Minimum cosine similarity for MFCC acoustic matching |
| `EAR_THRESHOLD` | `0.22` | Eye Aspect Ratio threshold for blink detection |
| `REQUIRED_BLINKS` | `1` | Blinks required to confirm liveness |
| `USE_BROWSER_AUDIO` | `True` | Record audio in browser with server microphone fallback |
| `SESSION_TIMEOUT_MINUTES` | `30` | Inactivity duration before automatic logout |
