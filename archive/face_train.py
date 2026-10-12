"""
face_train.py — Step 2: Train the LBPH face recognizer.
Reads images from images/<name>/ folders.
Saves trainer.yml and labels.pkl.
Run after face_collect.py.
"""

import cv2
import numpy as np
import os
import pickle

faces = []
labels = []
label_map = {}
label_id = 0

# FIXED: Use 'images' folder (same as face_collect.py output)
image_dir = "images"

if not os.path.exists(image_dir):
    print(f"❌ '{image_dir}' folder not found! Run face_collect.py first.")
    exit()

print("🔍 Scanning dataset...")

for name in sorted(os.listdir(image_dir)):
    person_path = os.path.join(image_dir, name)
    if not os.path.isdir(person_path):
        continue

    image_files = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.png'))]
    if len(image_files) == 0:
        print(f"⚠️  Skipping '{name}': no images found.")
        continue

    label_map[label_id] = name
    print(f"📁 Loading {len(image_files)} images for: {name}")

    for image_file in sorted(image_files):
        image_path = os.path.join(person_path, image_file)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"  ⚠️ Cannot read: {image_path}")
            continue

        try:
            image = cv2.resize(image, (200, 200))
        except Exception as e:
            print(f"  ⚠️ Skipping corrupted image: {image_path} ({e})")
            continue

        faces.append(image)
        labels.append(label_id)

    label_id += 1

if len(faces) == 0:
    print("❌ No valid training images found! Run face_collect.py first.")
    exit()

print(f"📚 Total images loaded: {len(faces)} from {label_id} person(s)")

# Train LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

# Save model and label map
recognizer.save("trainer.yml")
with open("labels.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("✅ Training complete! Saved: trainer.yml and labels.pkl")
print(f"👥 Trained persons: {list(label_map.values())}")
