"""
core/voice_engine.py
Voice verification engine using MFCC audio feature extraction (librosa),
dynamic challenge phrase generation, audio energy validation, and cosine similarity.
"""

import sounddevice as sd
import numpy as np
import librosa
from scipy.spatial.distance import cosine
import random

SAMPLE_RATE = 16000
DEFAULT_DURATION = 3.0
N_MFCC = 20

class VoiceEngine:
    def __init__(self, sample_rate=SAMPLE_RATE, duration=DEFAULT_DURATION, threshold=0.60):
        self.sample_rate = sample_rate
        self.duration = duration
        self.threshold = threshold

    @staticmethod
    def generate_challenge_code():
        """Generates a random 4-digit spoken verification challenge."""
        digits = [str(random.randint(0, 9)) for _ in range(4)]
        return " - ".join(digits)

    def record_audio(self, duration=None):
        """
        Records mono audio from the default microphone.
        Returns: 1D float32 numpy array.
        """
        dur = duration or self.duration
        num_frames = int(dur * self.sample_rate)
        audio = sd.rec(num_frames, samplerate=self.sample_rate, channels=1, dtype='float32')
        sd.wait()
        return audio.flatten()

    def check_audio_energy(self, audio, min_rms=0.005):
        """Validates that the recorded audio is not just silence or ambient room noise."""
        rms = np.sqrt(np.mean(audio**2))
        return rms >= min_rms, float(rms)

    def extract_mfcc_embedding(self, audio):
        """
        Extracts 20-dimensional normalized mean MFCC feature vector.
        """
        if len(audio) == 0:
            return None

        # Pre-emphasis filter to boost high frequencies
        audio_filtered = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

        mfcc = librosa.feature.mfcc(y=audio_filtered, sr=self.sample_rate, n_mfcc=N_MFCC)
        mean_mfcc = np.mean(mfcc, axis=1)

        # Normalize vector
        norm = np.linalg.norm(mean_mfcc)
        if norm > 0:
            mean_mfcc = mean_mfcc / norm
        return mean_mfcc

    def verify(self, recorded_embedding, enrolled_embedding):
        """
        Computes cosine similarity between recorded audio and enrolled profile.
        Returns: (passed: bool, similarity: float)
        """
        if recorded_embedding is None or enrolled_embedding is None:
            return False, 0.0

        dist = cosine(recorded_embedding, enrolled_embedding)
        sim = float(1.0 - dist)
        passed = sim >= self.threshold
        return passed, sim
