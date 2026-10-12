# voice_verify.py
import sounddevice as sd
import numpy as np
import librosa
import pickle
import os
from scipy.spatial.distance import cosine

SAMPLE_RATE = 16000
DURATION = 3

def record_audio(duration=DURATION, sr=SAMPLE_RATE):
    print("Recording voice for verification...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def extract_mfcc(audio, sr=SAMPLE_RATE, n_mfcc=20):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

def verify_user(username, voice_db="voice_db", threshold=0.45):
    path = os.path.join(voice_db, f"{username}.pkl")
    if not os.path.exists(path):
        print("No enrollment found for", username)
        return False, None
    enrolled = pickle.load(open(path, "rb"))
    audio = record_audio()
    mfcc = extract_mfcc(audio)
    # cosine similarity: 1 means identical; convert to similarity
    sim = 1 - cosine(mfcc, enrolled)
    print("Voice similarity:", sim)
    return (sim is not None and sim >= threshold), sim

if __name__ == "__main__":
    user = input("Username to verify: ").strip()
    ok, s = verify_user(user)
    print("Voice Verified" if ok else "Voice Rejected", s)
