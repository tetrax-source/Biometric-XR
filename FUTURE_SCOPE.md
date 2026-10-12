# Future Scope & Production Roadmap

This document outlines features designed for future production deployment that are deliberately beyond the scope of the current final-year prototype.

---

### 1. Institutional Hierarchy & Multi-Subject Timetable Integration
- **Current Prototype**: Global daily attendance per student.
- **Future Scope**: Expand database schema to model `Departments -> Sections -> Subjects -> Faculty`. Integrate live class timetables so the kiosk automatically logs attendance against the active subject period (e.g. Period 2: Engineering Mathematics).

### 2. Online Student Leave Management System
- **Current Prototype**: Binary presence recording (Present / Not Marked).
- **Future Scope**: Web portal for students to apply for medical leave or on-duty permissions with file uploads. Faculty can approve or reject leaves, auto-excluding approved absences from shortage calculations.

### 3. Automated Parent & Student Notifications (SMS / WhatsApp / Email)
- **Current Prototype**: Local database and CSV export.
- **Future Scope**: Automated SMS/WhatsApp API integration (e.g. Twilio, Gupshup) sending real-time notifications to parents when a student is absent, plus weekly email reports to faculty.

### 4. Cloud Database Migration (PostgreSQL / Supabase)
- **Current Prototype**: Local SQLite database (`attendance_system.db`).
- **Future Scope**: Migrate to managed cloud PostgreSQL to support concurrent writes from hundreds of classroom terminals simultaneously without file-locking bottlenecks.

### 5. Multi-Terminal IoT Deployment (Raspberry Pi Kiosks)
- **Current Prototype**: Laptop webcam and microphone.
- **Future Scope**: Deploy dedicated Raspberry Pi 5 / NVIDIA Jetson Nano edge terminals with wide-angle camera modules mounted outside each classroom door, communicating via secure REST APIs.

### 6. Cross-Platform Mobile Applications (React Native / Flutter)
- **Current Prototype**: Web interface and CustomTkinter desktop GUI.
- **Future Scope**: Mobile apps for teachers to monitor live classroom attendance and for students to track their overall attendance percentage against the mandatory 75% threshold.

### 7. Advanced Replay-Resistant & 3D Anti-Spoofing
- **Current Prototype**: 2D Eye Aspect Ratio (EAR) blink detection via MediaPipe.
- **Future Scope**: Multi-challenge liveness including randomized head rotation (yaw/pitch), mouth open/smile challenges, and remote photoplethysmography (rPPG) measuring facial blood pulse absorption to defeat digital video replay attacks.

### 8. Deep Neural Speaker Embeddings (ECAPA-TDNN / SpeechBrain)
- **Current Prototype**: 20-band Mel-Frequency Cepstral Coefficients (MFCC) with cosine distance.
- **Future Scope**: State-of-the-art deep neural acoustic embedding models (such as ECAPA-TDNN or ResNet-SE) trained on VoxCeleb to provide robust speaker verification even in noisy classroom corridors.

### 9. Semantic Spoken Challenge Code Transcription (Vosk / Whisper)
- **Current Prototype**: Acoustic timbre similarity check.
- **Future Scope**: Offline speech-to-text models verifying that the student accurately vocalized the randomized 4-digit spoken challenge code displayed on screen.
