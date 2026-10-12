"""
core/voice_engine.py
Voice verification & Speech-to-Text engine.
Supports:
1. Offline 4-digit spoken challenge transcription via Vosk (Kaldi recognizer with digit grammar).
2. Speaker identification via 20-band MFCC cosine similarity (controlled by VOICE_MATCH_ENABLED).
3. Audio energy / silence detection.
"""

import os
import sys
import json
import time
import random
import sounddevice as sd
import numpy as np
import librosa
from scipy.spatial.distance import cosine

try:
    from config import (
        AUDIO_SAMPLE_RATE, AUDIO_DURATION, MIN_AUDIO_RMS,
        VOICE_THRESHOLD, VOSK_MODEL_PATH, SPOKEN_CODE_CHECK,
        VOICE_MATCH_ENABLED, CODE_LENGTH, CODE_EXPIRY_SECONDS,
        MAX_CODE_ATTEMPTS
    )
except ImportError:
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_DURATION = 3.0
    MIN_AUDIO_RMS = 0.005
    VOICE_THRESHOLD = 0.50
    VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15"
    SPOKEN_CODE_CHECK = True
    VOICE_MATCH_ENABLED = False
    CODE_LENGTH = 4
    CODE_EXPIRY_SECONDS = 60
    MAX_CODE_ATTEMPTS = 3

SAMPLE_RATE = AUDIO_SAMPLE_RATE
DEFAULT_DURATION = AUDIO_DURATION
N_MFCC = 20

WORD_TO_DIGIT = {
    "zero": "0", "oh": "0",
    "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9"
}

DIGIT_GRAMMAR = json.dumps([
    "zero", "oh", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine", "[unk]"
])

# Singleton Vosk Model Cache
_VOSK_MODEL = None
_VOSK_LOAD_FAILED = False

def get_vosk_model():
    """Lazily loads the Vosk model. Returns None if model is unavailable without crashing."""
    global _VOSK_MODEL, _VOSK_LOAD_FAILED
    if _VOSK_LOAD_FAILED:
        return None
    if _VOSK_MODEL is None:
        try:
            import vosk
            if os.path.exists(VOSK_MODEL_PATH):
                # Suppress Vosk internal log spam if supported
                if hasattr(vosk, "SetLogLevel"):
                    vosk.SetLogLevel(-1)
                _VOSK_MODEL = vosk.Model(VOSK_MODEL_PATH)
            else:
                _VOSK_LOAD_FAILED = True
        except Exception as e:
            _VOSK_LOAD_FAILED = True
    return _VOSK_MODEL

class ChallengeSession:
    """Manages random single-use challenge codes with expiration and retry limits."""
    def __init__(self, length=CODE_LENGTH):
        self.code = "".join([str(random.randint(0, 9)) for _ in range(length)])
        self.created_at = time.time()
        self.attempts = 0
        self.max_attempts = MAX_CODE_ATTEMPTS
        self.expiry_seconds = CODE_EXPIRY_SECONDS

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.expiry_seconds

    def has_attempts_left(self) -> bool:
        return self.attempts < self.max_attempts

    def record_attempt(self):
        self.attempts += 1

class VoiceEngine:
    def __init__(self, sample_rate=SAMPLE_RATE, duration=DEFAULT_DURATION, threshold=VOICE_THRESHOLD):
        self.sample_rate = sample_rate
        self.duration = duration
        self.threshold = threshold

    @staticmethod
    def generate_challenge_code(length=CODE_LENGTH):
        """Generates a random N-digit spoken verification challenge string."""
        digits = [str(random.randint(0, 9)) for _ in range(length)]
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

    def check_audio_energy(self, audio, min_rms=MIN_AUDIO_RMS):
        """Validates that the recorded audio is not just silence or ambient room noise."""
        if audio is None or len(audio) == 0:
            return False, 0.0
        rms = np.sqrt(np.mean(audio**2))
        return rms >= min_rms, float(rms)

    def verify_spoken_code(self, audio, expected_code: str):
        """
        Transcribes speech using Vosk with a restricted digit grammar.
        Maps 'oh' and 'zero' to '0'.
        Passes ONLY if heard digits exactly equal expected code in the same order.
        Returns: (passed: bool, heard_digits: str)
        """
        clean_expected = "".join([c for c in str(expected_code) if c.isdigit()])
        if not clean_expected:
            return False, ""

        model = get_vosk_model()
        if model is None:
            # Model missing: if spoken code check is disabled in config, pass; else fail safely
            if not SPOKEN_CODE_CHECK:
                return True, clean_expected
            return False, "UNAVAILABLE"

        # Convert audio to 16 kHz 16-bit signed PCM
        if isinstance(audio, np.ndarray):
            audio_pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        elif isinstance(audio, bytes):
            audio_pcm = audio
        else:
            return False, ""

        try:
            import vosk
            rec = vosk.KaldiRecognizer(model, self.sample_rate, DIGIT_GRAMMAR)
            rec.AcceptWaveform(audio_pcm)
            result = json.loads(rec.FinalResult())
            text = result.get("text", "").lower().strip()

            # Parse spoken words into digits
            words = text.split()
            heard = [WORD_TO_DIGIT[w] for w in words if w in WORD_TO_DIGIT]
            heard_digits = "".join(heard)

            passed = (heard_digits == clean_expected)
            return passed, heard_digits
        except Exception as e:
            return False, ""

    # ----------------- OPTIONAL SPEAKER VERIFICATION (MFCC) -----------------

    def extract_mfcc_embedding(self, audio):
        """Extracts 20-dimensional normalized mean MFCC feature vector."""
        if audio is None or len(audio) == 0:
            return None
        audio_filtered = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        mfcc = librosa.feature.mfcc(y=audio_filtered, sr=self.sample_rate, n_mfcc=N_MFCC)
        mean_mfcc = np.mean(mfcc, axis=1)
        norm = np.linalg.norm(mean_mfcc)
        if norm > 0:
            mean_mfcc = mean_mfcc / norm
        return mean_mfcc

    def verify(self, recorded_embedding, enrolled_embedding):
        """Computes cosine similarity between recorded audio and enrolled profile."""
        if recorded_embedding is None or enrolled_embedding is None:
            return False, 0.0
        dist = cosine(recorded_embedding, enrolled_embedding)
        sim = float(1.0 - dist)
        passed = sim >= self.threshold
        return passed, sim
