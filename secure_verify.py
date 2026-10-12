# secure_verify.py
import os
import time
import pandas as pd
import datetime
from liveness import check_blink
from face_verify import recognize_face
from voice_verify import verify_user
from config import CAMERA_INDEX, EAR_THRESHOLD, REQUIRED_BLINKS

def mark_attendance(name, face_conf=0.0, voice_score=0.0):
    from core.database import record_attendance as db_record
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"attendance_{today}.csv"
    if not os.path.exists(filename):
        df = pd.DataFrame(columns=["Name","Date","Time"])
        df.to_csv(filename, index=False)
    df = pd.read_csv(filename)
    if name in set(df["Name"].tolist()):
        print("Already marked today in CSV.")
    else:
        now = datetime.datetime.now().strftime("%H:%M:%S")
        new_row = pd.DataFrame([[name, today, now]], columns=["Name","Date","Time"])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(filename, index=False)
        print(f"Attendance marked for {name} at {now}")

    # Also log to SQLite database
    db_ok, db_msg = db_record(name, method="Face + Voice + Liveness", face_conf=face_conf, voice_sim=voice_score)
    print(f"Database status: {db_msg}")

def main():
    print("STEP 1: LIVENESS (blink)")
    if not check_blink(timeout=8, required_blinks=1):
        print("Liveness failed. Abort.")
        return

    print("STEP 2: FACE RECOGNITION")
    name, conf = recognize_face(timeout=6, confidence_threshold=70)
    if name is None:
        print("Face not recognized. Abort.")
        return
    print(f"Face recognized as {name} (conf {conf:.1f})")

    print("STEP 3: VOICE VERIFICATION")
    ok, score = verify_user(name)
    if not ok:
        print("Voice verification failed. Abort.")
        return

    mark_attendance(name, face_conf=conf, voice_score=score)
    print("✅ All checks passed. Attendance recorded.")

if __name__ == "__main__":
    main()
