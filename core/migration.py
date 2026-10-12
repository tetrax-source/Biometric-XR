"""
core/migration.py
Imports legacy data from images/ and voice_db/ into the SQLite database.
"""

import sys
import os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
import pickle
import cv2
import numpy as np
from core.database import add_or_get_user, save_face_embedding, save_voice_embedding
from core.face_engine import FaceEngine

def migrate_legacy_data(image_dir="images", voice_dir="voice_db"):
    face_engine = FaceEngine()
    migrated_users = set()

    # 1. Migrate Voice Data
    if os.path.exists(voice_dir):
        for f in os.listdir(voice_dir):
            if f.endswith(".pkl"):
                name = os.path.splitext(f)[0].upper()
                file_path = os.path.join(voice_dir, f)
                try:
                    with open(file_path, "rb") as fp:
                        voice_data = pickle.load(fp)
                    if isinstance(voice_data, np.ndarray):
                        norm = np.linalg.norm(voice_data)
                        if norm > 0:
                            voice_data = voice_data / norm
                        save_voice_embedding(name, voice_data)
                        migrated_users.add(name)
                        print(f"[OK] Migrated voice profile for {name}")
                except Exception as e:
                    print(f"[WARN] Could not migrate {f}: {e}")

    # 2. Migrate Face Images (extract deep embeddings from existing image samples)
    if os.path.exists(image_dir):
        for user_folder in os.listdir(image_dir):
            person_path = os.path.join(image_dir, user_folder)
            if not os.path.isdir(person_path):
                continue
            name = user_folder.upper().replace(" ", "_")
            img_files = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            # Take up to 5 representative samples to keep database lean and fast
            selected_files = img_files[:5]
            added_count = 0
            for img_file in selected_files:
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                emb = face_engine.get_embedding(img)
                if emb is not None:
                    save_face_embedding(name, emb)
                    added_count += 1
            
            if added_count > 0:
                migrated_users.add(name)
                print(f"[OK] Migrated {added_count} face embeddings for {name}")

    print(f"Migration completed. Total users synced: {len(migrated_users)}")
    return len(migrated_users)

if __name__ == "__main__":
    migrate_legacy_data()
