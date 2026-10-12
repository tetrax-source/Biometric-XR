"""
core/database.py
SQLite Database Manager for Multi-Modal Biometric Attendance System.
Handles:
- User profiles and attendance records
- Safe raw float32 embedding serialization (NO pickle for new records)
- System users (teachers/admins) with roles
- Security audit logs
- Foreign key enforcement and query indexes
"""

import sqlite3
import datetime
import os
import pandas as pd
import numpy as np

try:
    from config import DB_PATH as DB_FILE
except ImportError:
    DB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "attendance_system.db")

def get_connection():
    """Returns an SQLite connection with Foreign Keys and Row factory enabled."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn

def serialize_embedding(embedding: np.ndarray) -> bytes:
    """Converts a numpy array to raw float32 bytes for safe storage without pickle."""
    arr = np.asarray(embedding, dtype=np.float32)
    return arr.tobytes()

def deserialize_embedding(blob: bytes) -> np.ndarray:
    """Loads raw float32 bytes back into a numpy array. Supports backward-compatible pickle fallback."""
    if blob is None or len(blob) == 0:
        return None
    # Check if this is legacy pickle (pickle protocols start with \x80)
    if blob.startswith(b"\x80"):
        import pickle
        data = pickle.loads(blob)
        return np.asarray(data, dtype=np.float32)
    return np.frombuffer(blob, dtype=np.float32).copy()

def init_db():
    """Initializes tables, constraints, and indexes."""
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Registered biometric users (students / participants)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 2. Face embeddings (128-d FaceNet float32 vectors)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS face_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_name TEXT NOT NULL,
        embedding BLOB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_name) REFERENCES users (name) ON DELETE CASCADE
    );
    """)

    # 3. Voice embeddings (20-d MFCC float32 vectors)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS voice_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_name TEXT UNIQUE NOT NULL,
        embedding BLOB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_name) REFERENCES users (name) ON DELETE CASCADE
    );
    """)

    # 4. Attendance records
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_name TEXT NOT NULL,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        verification_method TEXT DEFAULT 'Face + Spoken Code + Liveness',
        face_confidence REAL,
        voice_similarity REAL,
        spoken_code_ok INTEGER DEFAULT 1,
        heard_digits TEXT DEFAULT '',
        status TEXT DEFAULT 'Present'
    );
    """)

    # Safe ALTER TABLE migration for existing databases
    cursor.execute("PRAGMA table_info(attendance_logs)")
    existing_cols = [r["name"] for r in cursor.fetchall()]
    if "spoken_code_ok" not in existing_cols:
        cursor.execute("ALTER TABLE attendance_logs ADD COLUMN spoken_code_ok INTEGER DEFAULT 1")
    if "heard_digits" not in existing_cols:
        cursor.execute("ALTER TABLE attendance_logs ADD COLUMN heard_digits TEXT DEFAULT ''")

    # Index for fast duplicate checks and daily queries
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_attendance_user_date ON attendance_logs(user_name, date);
    """)

    # 5. Teachers & Admins authentication table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS system_users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('admin', 'teacher')),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 6. Audit log table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        action TEXT NOT NULL,
        details TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()
    conn.close()

# ----------------- BIOMETRIC DATA OPERATIONS -----------------

def add_or_get_user(name: str):
    name = name.strip().upper().replace(" ", "_")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO users (name) VALUES (?)", (name,))
    conn.commit()
    cursor.execute("SELECT id, name FROM users WHERE name = ?", (name,))
    user = cursor.fetchone()
    conn.close()
    return user["name"] if user else name

def save_face_embedding(user_name: str, embedding: np.ndarray):
    add_or_get_user(user_name)
    conn = get_connection()
    cursor = conn.cursor()
    data = serialize_embedding(embedding)
    cursor.execute("INSERT INTO face_embeddings (user_name, embedding) VALUES (?, ?)", (user_name, data))
    conn.commit()
    conn.close()

def save_voice_embedding(user_name: str, embedding: np.ndarray):
    add_or_get_user(user_name)
    conn = get_connection()
    cursor = conn.cursor()
    data = serialize_embedding(embedding)
    cursor.execute("""
        INSERT INTO voice_embeddings (user_name, embedding) 
        VALUES (?, ?) 
        ON CONFLICT(user_name) DO UPDATE SET embedding=excluded.embedding
    """, (user_name, data))
    conn.commit()
    conn.close()

def get_all_face_embeddings():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT user_name, embedding FROM face_embeddings")
    rows = cursor.fetchall()
    conn.close()

    result = {}
    for row in rows:
        name = row["user_name"]
        emb = deserialize_embedding(row["embedding"])
        if emb is not None:
            if name not in result:
                result[name] = []
            result[name].append(emb)
    return result

def get_voice_embedding(user_name: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT embedding FROM voice_embeddings WHERE user_name = ?", (user_name,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return deserialize_embedding(row["embedding"])
    return None

def get_all_users():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name, created_at FROM users ORDER BY name ASC")
    users = cursor.fetchall()
    conn.close()
    return [dict(u) for u in users]

def delete_user(user_name: str):
    user_name = user_name.strip().upper().replace(" ", "_")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE name = ?", (user_name,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted

# ----------------- ATTENDANCE LOGGING -----------------

def record_attendance(user_name: str, method="Face + Spoken Code + Liveness", face_conf=0.0, voice_sim=0.0, spoken_code_ok=1, heard_digits=""):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.datetime.now().strftime("%H:%M:%S")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM attendance_logs WHERE user_name = ? AND date = ?", (user_name, today))
    existing = cursor.fetchone()
    if existing:
        conn.close()
        return False, f"Attendance already logged for {user_name} today at {today}"

    cursor.execute("""
        INSERT INTO attendance_logs (user_name, date, time, verification_method, face_confidence, voice_similarity, spoken_code_ok, heard_digits, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Present')
    """, (user_name, today, now_time, method, face_conf, voice_sim, spoken_code_ok, heard_digits))
    conn.commit()
    conn.close()
    return True, f"Attendance marked for {user_name} at {now_time}"

def get_attendance_logs(date=None):
    conn = get_connection()
    cursor = conn.cursor()
    if date:
        cursor.execute("SELECT * FROM attendance_logs WHERE date = ? ORDER BY id DESC", (date,))
    else:
        cursor.execute("SELECT * FROM attendance_logs ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def export_attendance_to_csv(filepath: str, date=None):
    logs = get_attendance_logs(date)
    cols = ["user_name", "date", "time", "verification_method", "face_confidence", "voice_similarity", "spoken_code_ok", "heard_digits", "status"]
    if not logs:
        df = pd.DataFrame(columns=cols)
    else:
        df = pd.DataFrame(logs)
        if "id" in df.columns:
            df = df.drop(columns=["id"])
    df.to_csv(filepath, index=False)
    return len(df)

# ----------------- TEACHERS / ADMINS & AUDIT LOG -----------------

def create_system_user(username: str, password_hash: str, role: str = "teacher"):
    username = username.strip().lower()
    if role not in ("admin", "teacher"):
        raise ValueError("Role must be 'admin' or 'teacher'")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO system_users (username, password_hash, role)
        VALUES (?, ?, ?)
    """, (username, password_hash, role))
    conn.commit()
    uid = cursor.lastrowid
    conn.close()
    return uid

def get_system_user(username: str):
    username = username.strip().lower()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM system_users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_system_users():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, role, created_at FROM system_users ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def log_audit(username: str, action: str, details: str = None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO audit_log (username, action, details)
        VALUES (?, ?, ?)
    """, (username, action, details))
    conn.commit()
    conn.close()

def get_audit_logs(limit: int = 100):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ----------------- ONE-TIME EMBEDDING MIGRATION -----------------

def migrate_pickled_embeddings():
    """Converts legacy pickled embeddings in attendance_system.db to raw float32 buffers."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Migrate Face Embeddings
    cursor.execute("SELECT id, embedding FROM face_embeddings")
    face_rows = cursor.fetchall()
    face_converted = 0
    for row in face_rows:
        blob = row["embedding"]
        if blob and blob.startswith(b"\x80"):
            import pickle
            arr = np.asarray(pickle.loads(blob), dtype=np.float32)
            new_blob = arr.tobytes()
            cursor.execute("UPDATE face_embeddings SET embedding = ? WHERE id = ?", (new_blob, row["id"]))
            face_converted += 1

    # 2. Migrate Voice Embeddings
    cursor.execute("SELECT id, embedding FROM voice_embeddings")
    voice_rows = cursor.fetchall()
    voice_converted = 0
    for row in voice_rows:
        blob = row["embedding"]
        if blob and blob.startswith(b"\x80"):
            import pickle
            arr = np.asarray(pickle.loads(blob), dtype=np.float32)
            new_blob = arr.tobytes()
            cursor.execute("UPDATE voice_embeddings SET embedding = ? WHERE id = ?", (new_blob, row["id"]))
            voice_converted += 1

    conn.commit()
    conn.close()
    return face_converted, voice_converted

# Initialize tables and indexes on import
init_db()
