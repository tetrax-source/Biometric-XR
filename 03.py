import cv2
import numpy as np
import pandas as pd
import os
import datetime
import pickle
import time

# ==============================
# Load Haar Cascade
# ==============================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("❌ Error: Haar cascade not loaded.")
    exit()

# ==============================
# Load Trained LBPH Model
# ==============================
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# Load Labels
with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

# ==============================
# Attendance File Handling
# ==============================
today = datetime.datetime.now().strftime("%Y-%m-%d")
filename = f"attendance_{today}.csv"

if not os.path.exists(filename):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv(filename, index=False)
else:
    df = pd.read_csv(filename)

# Store names already marked
marked_names = set(df["Name"].tolist())

# ==============================
# Start Webcam
# ==============================
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("❌ Error: Webcam not detected.")
    exit()

print("🔁 Attendance system running... Press ESC to exit.")

last_seen = {}  # stores name : timestamp for debounce

# ==============================
# Main Loop
# ==============================
while True:
    ret, frame = cam.read()
    if not ret:
        print("❌ Error reading from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # improve detection in low light

    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label, confidence = recognizer.predict(face)

        # ==============================
        # Recognition Threshold
        # ==============================
        threshold = 70  # stricter threshold for better accuracy

        if confidence < threshold:
            name = labels[label]

            # Debounce: Allow marking only every 5 seconds
            now = time.time()
            if name not in last_seen or (now - last_seen[name]) > 5:

                if name not in marked_names:
                    current_time = datetime.datetime.now().strftime("%H:%M:%S")

                    new_row = pd.DataFrame([[name, today, current_time]], columns=["Name", "Date", "Time"])
                    df = pd.concat([df, new_row], ignore_index=True)
                    df.to_csv(filename, index=False)

                    marked_names.add(name)

                    print(f"✅ Attendance marked for {name} at {current_time}")

                last_seen[name] = now

        else:
            name = "Unknown"

        # ==============================
        # Draw bounding box
        # ==============================
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, f"{name}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)

    cv2.imshow("Face Recognition Attendance", frame)

    # Exit on ESC
    if cv2.waitKey(1) == 27:
        print("👋 Exiting attendance system.")
        break

cam.release()
cv2.destroyAllWindows()
