import sounddevice as sd
import numpy as np
import librosa
import os

# =========================
# Ask user name
# =========================
name = input("Enter your name: ").strip().replace(" ", "_")

if name == "":
    print("❌ Invalid name")
    exit()

os.makedirs("voices", exist_ok=True)

# =========================
# Recording settings
# =========================
DURATION = 5      # seconds
SAMPLE_RATE = 16000

print("🎤 Speak clearly... Recording voice")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)

sd.wait()

audio = audio.flatten()

# =========================
# Extract MFCC features
# =========================
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=SAMPLE_RATE,
    n_mfcc=20
)

mfcc_mean = np.mean(mfcc, axis=1)

# =========================
# Save voice features
# =========================
np.save(f"voices/{name}.npy", mfcc_mean)

print(f"✅ Voice enrollment completed for {name}")
print(f"📁 Saved voice data: voices/{name}.npy")
