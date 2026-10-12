# Final-Year Project Viva Defense Notes

A complete guide to the architecture, pipeline, and the 10 most common external examiner viva questions with clear, concise answers.

---

## 1. System Architecture & Pipeline Overview

Our project is a **Multi-Modal Biometric Attendance & Access System** combining three layers of identity verification:

```
[ User in front of camera ]
            │
            ▼
1. Liveness Detection      --> MediaPipe FaceLandmarker tracks 6 eye landmarks.
   (Anti-Spoofing)             Calculates Eye Aspect Ratio (EAR). Rejects static photos.
            │ (Pass: Blink detected)
            ▼
2. Face Recognition        --> OpenCV detects face bounding box.
   (Identity Matching)         DeepFace FaceNet extracts 128-d neural embedding vector.
                               Cosine Similarity >= 0.55 matches against SQLite profiles.
            │ (Pass: Face identified)
            ▼
3. Voice Verification      --> Records 3s audio at 16 kHz. Checks energy RMS >= 0.005.
   (Acoustic Confirmation)     Librosa extracts 20 mean MFCC acoustic features.
                               Cosine Similarity >= 0.50 confirms speaker identity.
            │ (Pass: Voice verified)
            ▼
4. Database Attendance     --> SQLite checks if attendance already recorded today.
                               If new: logs user, date, time, confidence scores.
```

---

## 2. Top 10 Likely Viva Questions & Answers

### Q1: What is "Multi-Modal Biometrics" and why is it superior to unimodal biometrics?
**Answer**: Unimodal systems rely on a single biometric trait (e.g., face only or fingerprint only), which can be spoofed using printed photos or silicone casts. Multi-modal biometrics fuses multiple physiological or behavioral traits—here, **Face Geometry + Vocal Acoustics + Blink Liveness**. An attacker would need to spoof all three simultaneously, drastically reducing the False Acceptance Rate (FAR) and preventing proxy attendance.

---

### Q2: How does your liveness (anti-spoofing) detection work? What is EAR?
**Answer**: We use the **Eye Aspect Ratio (EAR)** calculated from 6 2D facial landmarks per eye provided by MediaPipe:
$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$
When the eye is open, EAR is approximately $0.30 - 0.35$. When the eye closes during a blink, the vertical distance drops near zero ($\text{EAR} < 0.22$). By tracking the state transition (Open $\rightarrow$ Closed $\rightarrow$ Open), we verify that a live human is present, rejecting static printed photos or phone screen selfies.

---

### Q3: Why did you choose FaceNet deep embeddings over traditional OpenCV Haar Cascade / LBPH?
**Answer**: LBPH (Local Binary Patterns Histograms) uses handcrafted texture patterns that fail under varying room lighting, head angles, or facial expressions. **FaceNet** is a deep Convolutional Neural Network trained with Triplet Loss on millions of faces. It maps facial features into a compact 128-dimensional Euclidean space where Euclidean distance directly corresponds to face similarity. FaceNet achieves $>99\%$ benchmark accuracy and is invariant to illumination and slight pose changes.

---

### Q4: What is a face "embedding", and how do you compare two embeddings?
**Answer**: An embedding is a 128-dimensional floating-point vector representing the high-level semantic features of a human face (distance between eyes, nose bridge curvature, jaw shape). We compare embeddings using **Cosine Similarity**:
$$\text{Cosine Similarity} = 1 - \text{Cosine Distance} = \frac{A \cdot B}{\|A\|_2 \|B\|_2}$$
Because our vectors are $L_2$-normalized ($\|A\|_2 = 1$), Cosine Similarity is simply the dot product. If the score is $\ge 0.55$, the system declares a match.

---

### Q5: How does the voice verification stage work? What are MFCCs?
**Answer**: MFCC stands for **Mel-Frequency Cepstral Coefficients**. Human hearing does not perceive audio frequencies linearly; we perceive pitches on a logarithmic scale called the Mel scale. MFCC extraction:
1. Applies a pre-emphasis filter ($y[n] - 0.97 y[n-1]$) to boost high frequencies.
2. Performs Short-Time Fourier Transform (STFT) with a Hann window.
3. Maps powers through a triangular Mel filter bank.
4. Takes the logarithm and applies Discrete Cosine Transform (DCT) to yield 20 coefficients.
We compute the mean vector across time frames to create an acoustic voiceprint and match it using Cosine Similarity ($\ge 0.50$).

---

### Q6: Why did you use SQLite for database storage instead of CSV files?
**Answer**: Flat CSV files suffer from:
1. No concurrency control: multiple simultaneous writes corrupt file contents.
2. No data integrity constraints: foreign keys and unique constraints cannot be enforced.
3. Slow queries: searching requires full-file text scans.
SQLite is an ACID-compliant relational database engine. We enabled `PRAGMA foreign_keys = ON;`, indexed `(user_name, date)` for instant duplicate lookups, and created dedicated tables for users, embeddings, attendance, faculty, and audit trails.

---

### Q7: What are FAR, FRR, and EER? What do your evaluation numbers show?
**Answer**:
- **FAR (False Accept Rate)**: Percentage of times an unauthorized impostor is incorrectly accepted as genuine.
- **FRR (False Reject Rate)**: Percentage of times a genuine registered user is incorrectly rejected.
- **EER (Equal Error Rate)**: The threshold where FAR equals FRR; lower EER indicates higher overall biometric accuracy.
In our evaluation script (`evaluate.py`), our configured threshold of $0.55$ for FaceNet yields $0.0\%$ FAR on genuine test sets, and $0.50$ for Voice yields low error while maintaining user convenience.

---

### Q8: What security issues were present in the initial version, and how did you fix them?
**Answer**:
1. **No Authentication**: The web interface was open to anyone on the network. We implemented **Role-Based Access Control (RBAC)**: public users only access an isolated "Kiosk mode", while Teachers and Admins must log in.
2. **Insecure Pickle Deserialization**: Storing embeddings with Python `pickle` allows arbitrary code execution if modified. We replaced it with raw `float32` byte serialization (`tobytes()` and `np.frombuffer()`).
3. **Plaintext Passwords**: Replaced with cryptographic hashing via `werkzeug.security` (`scrypt`/`pbkdf2:sha256`).
4. **CSRF Vulnerability**: Added session-validated CSRF tokens on all mutating endpoints.

---

### Q9: What happens if a student attempts to mark attendance twice in one day?
**Answer**: The system enforces daily idempotency. In `core/database.py`, `record_attendance()` queries:
`SELECT id FROM attendance_logs WHERE user_name = ? AND date = ?`
If a record exists for today, the function immediately returns `False` and a status message stating attendance was already marked. This is tested and verified by our automated test suite.

---

### Q10: What are the main future improvements for this project?
**Answer**:
1. **Hardware anti-spoofing**: Adding 3D depth cameras or active head pose challenges (e.g. "turn head left") to prevent prerecorded video playback attacks.
2. **Speech-to-text challenge verification**: Integrating offline Vosk or Whisper to confirm that the student vocalized the exact 4-digit challenge code displayed on screen.
3. **Cloud scaling**: Migrating SQLite to PostgreSQL to support hundreds of classroom kiosks running across a campus network.
