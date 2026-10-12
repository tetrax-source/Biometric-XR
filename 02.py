import cv2
import numpy as np
import os
import pickle

faces = []
labels = []
label_map = {}
label_id = 0

image_dir = "images"

print("🔍 Scanning dataset...")

for name in sorted(os.listdir(image_dir)):
    person_path = os.path.join(image_dir, name)
    if not os.path.isdir(person_path):
        continue

    label_map[label_id] = name
    print(f"📁 Loading images for: {name}")

    for image_file in sorted(os.listdir(person_path)):
        image_path = os.path.join(person_path, image_file)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"⚠️ Cannot read: {image_path}")
            continue

        # Images already 200x200 from collector
        try:
            image = cv2.resize(image, (200, 200))
        except:
            print(f"⚠️ Skipping corrupted image: {image_path}")
            continue

        faces.append(image)
        labels.append(label_id)

    label_id += 1

if len(faces) == 0:
    print("❌ No valid training images found!")
    exit()

print(f"📚 Total images collected: {len(faces)}")

# Train recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

# Save model and labels
recognizer.save("trainer.yml")
with open("labels.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("✅ Training complete! Model saved.")
