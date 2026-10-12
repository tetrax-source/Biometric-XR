# ⚡ Multi-Modal Biometric Attendance & Security System

An advanced, production-ready AI attendance system integrating **Facial Recognition (DeepFace/FaceNet)**, **Voice Verification (MFCC & Cosine Similarity)**, and **Liveness Detection (MediaPipe Blink EAR)** with a modern **CustomTkinter Desktop GUI** and persistent **SQLite Database**.

---

## 🚀 Key Highlights & Architecture

```
                       [ Live Webcam Stream ]
                                 │
                                 ▼
         ┌─────────────────────────────────────────────────┐
         │ Step 1: Liveness Check (Blink EAR via MediaPipe) │
         └───────────────────────┬─────────────────────────┘
                                 │ (Passed)
                                 ▼
         ┌─────────────────────────────────────────────────┐
         │ Step 2: Face Recognition (FaceNet Embeddings)    │
         └───────────────────────┬─────────────────────────┘
                                 │ (Match Identified)
                                 ▼
         ┌─────────────────────────────────────────────────┐
         │ Step 3: Voice Verification (MFCC Audio Challenge)│
         └───────────────────────┬─────────────────────────┘
                                 │ (Voice Verified)
                                 ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ Step 4: Automated Attendance Logged in SQLite & CSV              │
 └──────────────────────────────────────────────────────────────────┘
```

1. **Anti-Spoofing Liveness Engine** ([core/liveness.py](file:///e:/Aswin/core/liveness.py)):
   - Tracks 468 facial mesh landmarks using MediaPipe.
   - Computes Eye Aspect Ratio (EAR) to detect natural eyelid motion, blocking spoof attempts with photographs or phone screens.

2. **Deep Learning Face Recognition** ([core/face_engine.py](file:///e:/Aswin/core/face_engine.py)):
   - Generates 128-dimensional normalized embeddings with FaceNet via ONNX Runtime.
   - Measures identity similarity using Cosine Distance against enrolled profiles in the database.

3. **Acoustic Voice Verification** ([core/voice_engine.py](file:///e:/Aswin/core/voice_engine.py)):
   - Audio energy / RMS validation to detect silence or low signal.
   - Extracts 20-dimensional Mel-Frequency Cepstral Coefficients (MFCC) with Librosa.
   - Dynamic 4-digit spoken verification challenge code to counter audio replay attacks.

4. **Persistent SQLite Database** ([core/database.py](file:///e:/Aswin/core/database.py)):
   - Structured relational storage (`attendance_system.db`) for user identities, face embeddings, voice templates, and daily attendance logs.
   - Daily duplicate check preventing repeated entries for the same individual on a single day.

5. **Modern Desktop Dashboard** ([app_gui.py](file:///e:/Aswin/app_gui.py)):
   - Built with **CustomTkinter** (dark theme).
   - Real-time video preview with bounding box and HUD overlays.
   - Interactive user enrollment tab (Face capture + Voice recording).
   - Filterable attendance history log with one-click **Export to CSV**.
   - Adjustable security thresholds (EAR, Face match confidence, Voice similarity).

---

## 🛠 Project Structure

```
├── core/
│   ├── database.py         # SQLite database schema, CRUD, and CSV exporter
│   ├── face_engine.py      # FaceNet deep embedding & recognition engine
│   ├── voice_engine.py     # Librosa MFCC voice extraction & verification
│   ├── liveness.py         # MediaPipe FaceMesh EAR blink detector
│   └── migration.py        # Migrates legacy images/ & voice_db/ into SQLite
├── legacy/                 # Preserved prototype scripts (01.py, 02.py, 03.py, etc.)
├── data/                   # Haarcascade XML models
├── images/                 # Enrolled face images
├── voice_db/               # Enrolled voice profiles
├── attendance_system.db    # SQLite database
├── app_web.py              # Flask Web Application (localhost:5000)
├── app_gui.py              # CustomTkinter Desktop GUI Application
├── templates/              # Web portal HTML dashboard templates
├── secure_verify.py        # Terminal verification pipeline
├── main.py                 # Primary entry point (Web, GUI, or CLI mode)
└── requirements.txt        # Python package dependencies
```

---

## 💻 How to Run

### 1. Launch in Web Browser (Localhost)
```bash
python main.py --web
```
*(Or run `python app_web.py`)*  
Then open your web browser (Chrome, Edge, etc.) at:
👉 **http://localhost:5000** (or **http://127.0.0.1:5000**)

### 2. Launch the Desktop GUI Dashboard
```bash
python main.py
```

### 3. Run in Terminal / Headless CLI Mode
```bash
python main.py --cli
```

### 4. Sync Existing Data into SQLite
```bash
python -m core.migration
```
