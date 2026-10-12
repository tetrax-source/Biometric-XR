import cv2
import os
import sounddevice as sd
import numpy as np
import librosa
import pickle
import time

# ========================
# FACE SETTINGS
# ========================
MAX_FACE_IMAGES = 30
FACE_SIZE = (200, 200)

# ========================
# VOICE SETTINGS
# ========================
SAMPLE_RATE = 16000
DURATION = 3
VOICE_SAMPLES = 5

# ========================
# LOAD FACE DETECTOR
# ========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("❌ Haar Cascade not loaded")
    exit()

# ========================
# USER NAME
# ========================
name = input("Enter your name: ").strip()
if not name:
    print("❌ Invalid name")
    exit()

# ========================
# CREATE DIRECTORIES
# ========================
face_dir = f"images/{name}"
voice_dir = "voice_db"

os.makedirs(face_dir, exist_ok=True)
os.makedirs(voice_dir, exist_ok=True)

# ========================
# FACE ENROLLMENT
# ========================
print("\n📸 FACE ENROLLMENT STARTED")
cam = cv2.VideoCapture(0)

count = 0
while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, FACE_SIZE)

        cv2.imwrite(f"{face_dir}/{count}.jpg", face)
        count += 1

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.putText(frame, f"Face Images: {count}/{MAX_FACE_IMAGES}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.imshow("Face Enrollment", frame)

    if cv2.waitKey(1) & 0xFF == 27 or count >= MAX_FACE_IMAGES:
        break

cam.release()
cv2.destroyAllWindows()

print(f"✅ Face enrollment completed ({count} images)")

# ========================
# VOICE FUNCTIONS
# ========================
def record_audio():
    print("🎤 Speak clearly...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32")
    sd.wait()
    return audio.flatten()

def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=20)
    return np.mean(mfcc, axis=1)

# ========================
# VOICE ENROLLMENT
# ========================
print("\n🎧 VOICE ENROLLMENT STARTED")
embeddings = []

for i in range(VOICE_SAMPLES):
    input(f"Press ENTER to record voice sample {i+1}/{VOICE_SAMPLES}")
    audio = record_audio()
    mfcc = extract_mfcc(audio)
    embeddings.append(mfcc)
    print("✔ Sample saved")

avg_voice = np.mean(embeddings, axis=0)

with open(f"{voice_dir}/{name}.pkl", "wb") as f:
    pickle.dump(avg_voice, f)

print("✅ Voice enrollment completed")

# ========================
# DONE
# ========================
print("\n🎉 AUTO ENROLLMENT COMPLETE!")
print(f"👤 User: {name}")
print("📁 Face images saved in images/")
print("🎤 Voice saved in voice_db/")
