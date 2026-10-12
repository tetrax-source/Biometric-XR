# Biometric System Security & Privacy Analysis

This document summarizes the security vulnerabilities addressed during the upgrade and the remaining architectural considerations necessary for enterprise-grade deployment.

---

## 1. Vulnerabilities Addressed & Resolved

| Vulnerability / Risk | Severity Before | Fix Implemented |
| :--- | :---: | :--- |
| **Unauthenticated Access to Admin Functions** | **Critical** | Implemented Role-Based Access Control (RBAC). Admin and Teacher accounts must authenticate via `/login`. Public interface is locked down as a read-only biometric "Kiosk Mode". |
| **Insecure Deserialization (`pickle.loads`)** | **High** | Replaced `pickle` with safe raw `float32` binary buffers (`np.ndarray.tobytes()` and `np.frombuffer()`). Eliminates Remote Code Execution (RCE) vectors from tampered database records. |
| **Hardcoded or Unprotected Passwords** | **High** | Passwords are never stored in plaintext. Hashed using modern, salted cryptographic functions (`werkzeug.security` with scrypt / pbkdf2:sha256). |
| **Cross-Site Request Forgery (CSRF)** | **Medium** | Implemented custom cryptographic CSRF session tokens (`secrets.token_hex(16)`). Mutating endpoints (`/api/enroll`, `/api/create_teacher`, `/api/delete_user`) reject requests without matching tokens. |
| **Orphaned Sessions (Session Hijacking)** | **Medium** | Configured 30-minute inactivity session expiration. Sessions auto-invalidate on timeout and log an audit event. |
| **Data Integrity Flaws** | **Medium** | Enabled SQLite relational foreign key enforcement (`PRAGMA foreign_keys = ON;`). Added `ON DELETE CASCADE` and index on `attendance_logs(user_name, date)`. |
| **Lack of Accountability** | **Medium** | Introduced an immutable `audit_log` table tracking every login, logout, student enrollment, deletion, faculty creation, and report export with timestamps. |

---

## 2. Known Limitations & Remaining Security Considerations

### A. Biometric Templates Stored Unencrypted at Rest
- **Current Status**: Facial embeddings (128-d vectors) and voice embeddings (20-d vectors) are stored as raw binary blobs in SQLite.
- **Risk**: An attacker with direct physical or file system access to `attendance_system.db` can copy the embeddings. While deep embeddings cannot be directly converted back into human-recognizable photographs, facial feature reconstruction attacks (feature inversion) are theoretically possible.
- **Production Recommendation**: Encrypt biometric blobs at rest using AES-256-GCM with keys managed in a hardware security module (HSM) or cloud key vault (e.g., AWS KMS, HashiCorp Vault), or explore Cancelable Biometrics / Fuzzy Commitments.

### B. Video Replay & Synthetic Audio Liveness Spoofing
- **Current Status**: Liveness is determined by 2D Eye Aspect Ratio (EAR) tracking eye blinks followed by a random, dynamic 4-digit spoken challenge code transcribed using offline Vosk ASR.
- **Protection Achieved**: Static printed photos, tablet selfies, and prerecorded video loops fail because:
  1. The user's eyes must transition from open to closed to open in real time.
  2. The random 4-digit challenge cannot be known in advance by an attacker, defeating static prerecorded audio replay attacks.
- **Production Recommendation**: Add active head-pose challenge prompts (e.g. "turn head left 30 degrees") or 3D structured light sensors (Intel RealSense, iPhone TrueDepth) to defend against sophisticated real-time 3D deepfake masks.

### C. Audio Processing & Microphone Privacy
- **Current Status**: Vosk processes speech locally on CPU in memory without transmitting audio across the network.
- **Protection Achieved**: Raw audio buffers are discarded immediately after digit extraction; only the numeric heard string is logged to `attendance_logs` and `audit_log`.

### D. Privacy & Regulatory Compliance (DPDP Act 2023 / GDPR)
- **Legal Context**: Biometric data (facial geometry and acoustic templates) constitutes Sensitive Personal Data under the Digital Personal Data Protection (DPDP) Act 2023 (India) and GDPR Article 9 (EU).
- **Required Safeguards**:
  1. **Consent Architecture**: Students must provide explicit, revocable digital or written consent before enrollment.
  2. **Right to Erasure**: The newly implemented `delete_user()` function supports GDPR "Right to be Forgotten" by deleting enrolled vectors.
  3. **Purpose Limitation**: Biometric signatures must strictly be utilized for attendance verification and never shared with third parties.

