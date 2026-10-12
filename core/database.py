"""
core/database.py
SQLite Database Manager for Multi-Modal Biometric Attendance System.
Handles user profiles, face embeddings, voice templates, and daily attendance logs.
"""

import sqlite3
import datetime
import pickle
import os
import pandas as pd
import numpy as np

try:
    from config import DB_PATH as DB_FILE
except ImportError:
    DB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "attendance_system.db")

def get_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS face_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_name TEXT NOT NULL,
        embedding BLOB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_name) REFERENCES users (name) ON DELETE CASCADE
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS voice_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_name TEXT UNIQUE NOT NULL,
        embedding BLOB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_name) REFERENCES users (name) ON DELETE CASCADE
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_name TEXT NOT NULL,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        verification_method TEXT DEFAULT 'Face + Voice + Liveness',
        face_confidence REAL,
        voice_similarity REAL,
        status TEXT DEFAULT 'Present'
    );
    """)

    conn.commit()
    conn.close()

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
    data = pickle.dumps(embedding)
    cursor.execute("INSERT INTO face_embeddings (user_name, embedding) VALUES (?, ?)", (user_name, data))
    conn.commit()
    conn.close()

def save_voice_embedding(user_name: str, embedding: np.ndarray):
    add_or_get_user(user_name)
    conn = get_connection()
    cursor = conn.cursor()
    data = pickle.dumps(embedding)
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
        emb = pickle.loads(row["embedding"])
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
        return pickle.loads(row["embedding"])
    return None

def get_all_users():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name, created_at FROM users ORDER BY name ASC")
    users = cursor.fetchall()
    conn.close()
    return [dict(u) for u in users]

def record_attendance(user_name: str, method="Face + Voice + Liveness", face_conf=0.0, voice_sim=0.0):
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
        INSERT INTO attendance_logs (user_name, date, time, verification_method, face_confidence, voice_similarity, status)
        VALUES (?, ?, ?, ?, ?, ?, 'Present')
    """, (user_name, today, now_time, method, face_conf, voice_sim))
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
    if not logs:
        df = pd.DataFrame(columns=["user_name", "date", "time", "verification_method", "face_confidence", "voice_similarity", "status"])
    else:
        df = pd.DataFrame(logs)
        if "id" in df.columns:
            df = df.drop(columns=["id"])
    df.to_csv(filepath, index=False)
    return len(df)

# Initialize on module load
init_db()
