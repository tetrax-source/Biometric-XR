========================================================================
BIOMETRIC EVALUATION DATASET GUIDE (FOR VIVA REPORT & TESTING)
========================================================================

This folder is used by `evaluate.py` to calculate False Accept Rate (FAR),
False Reject Rate (FRR), and Equal Error Rate (EER) for your viva project.

1. DIRECTORY STRUCTURE
-----------------------
eval_data/
  ├── genuine/
  │     └── <STUDENT_NAME>/       <- Put images and voice files of ENROLLED users here
  │           ├── face1.jpg
  │           ├── face2.jpg
  │           └── voice1.wav (or .npy / .pkl)
  └── impostor/                   <- Put images and voice files of UNREGISTERED people here
        ├── stranger1.jpg
        ├── stranger2.jpg
        └── stranger_voice1.wav

2. RECOMMENDED SAMPLES TO COLLECT
---------------------------------
For each enrolled student:
- 5 to 10 Genuine Face Photos:
  * Taken with regular lighting
  * Taken with slightly dim/bright lighting
  * Taken at slight angles (+/- 15 degrees)
- 3 to 5 Genuine Voice Audio Recordings:
  * 3-second audio clips saved as .wav (16 kHz mono) or .npy

For impostors (friends or random public dataset samples):
- 10 Impostor Face Photos (people NOT in the database)
- 5 Impostor Voice Recordings (voices of people NOT in the database)

3. SPOOFING TESTS (ANTI-SPOOFING / LIVENESS)
--------------------------------------------
Use `spoof_log.csv` to record manual anti-spoofing results:
1. Static Photo Test:
   - Hold a printed color photo of an enrolled student in front of webcam.
   - Record if liveness blink detection rejects it. (Expected: REJECTED).
2. Phone Screen Test:
   - Display a selfie photo on a smartphone screen and hold it up to webcam.
   - Record if liveness blink detection rejects it. (Expected: REJECTED).
3. Video Replay Test:
   - Play a video clip of a student blinking on a smartphone/tablet in front of webcam.
   - Note whether the blink was detected or rejected.
4. Live Human Test:
   - Real enrolled student blinks naturally. (Expected: ACCEPTED).

Run `python evaluate.py` after adding samples to generate:
- `evaluation_results.csv`
- `evaluation_roc_curves.png`
