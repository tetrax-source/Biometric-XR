"""
voice_enroll.py — Enroll a user's voice profile.
Records 5 voice samples, computes average MFCC,
saves to voice_db/<NAME>.pkl (uppercase, consistent naming).
"""

import sounddevice as sd
import numpy as np
import librosa
import pickle
import os

SAMPLE_RATE = 16000
DURATION = 3        # seconds per sample
N_SAMPLES = 5       # number of voice samples to record


def record_audio(duration=DURATION, sr=SAMPLE_RATE):
    print("🎤 Recording... Speak now!")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    print("✔ Done recording.")
    return audio.flatten()


def extract_mfcc(audio, sr=SAMPLE_RATE, n_mfcc=20):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)


def enroll_user(username, out_dir="voice_db"):
    os.makedirs(out_dir, exist_ok=True)
    embeddings = []

    print(f"\n🎧 Enrolling voice for: {username}")
    print(f"You will record {N_SAMPLES} voice samples (3 seconds each).")
    print("Speak a consistent phrase like: 'My name is [name], please verify me'")

    for i in range(N_SAMPLES):
        input(f"\n  ➤ Press ENTER to record sample {i+1}/{N_SAMPLES}...")
        audio = record_audio()
        mfcc = extract_mfcc(audio)
        embeddings.append(mfcc)
        print(f"  ✅ Sample {i+1} saved.")

    avg_emb = np.mean(embeddings, axis=0)
    db_path = os.path.join(out_dir, f"{username}.pkl")
    with open(db_path, "wb") as f:
        pickle.dump(avg_emb, f)

    print(f"\n✅ Voice enrollment complete for '{username}'.")
    print(f"📁 Saved to: {db_path}")


if __name__ == "__main__":
    # FIXED: Force UPPERCASE to match face model naming
    user = input("Enter username to enroll: ").strip().upper().replace(" ", "_")
    if not user:
        print("❌ Invalid name.")
        exit()
    enroll_user(user)
