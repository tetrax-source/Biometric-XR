"""
face_verify.py — Reusable face recognition module.
Used by secure_verify.py.
Loads trainer.yml and labels.pkl to recognize a face from webcam.
"""

import cv2
import pickle
import time
import os

def load_recognizer(trainer_path="trainer.yml", labels_path="labels.pkl"):
    if not os.path.exists(trainer_path):
        raise FileNotFoundError(f"❌ '{trainer_path}' not found. Run face_train.py first.")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"❌ '{labels_path}' not found. Run face_train.py first.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainer_path)
    with open(labels_path, "rb") as f:
        labels = pickle.load(f)
    return recognizer, labels


def recognize_face(timeout=8, confidence_threshold=70, camera_index=0):
    """
    Opens webcam, attempts to recognize a face within `timeout` seconds.
    Returns (name, confidence) if recognized, or (None, None) if not.
    Lower confidence value = better match in LBPH.
    """
    try:
        recognizer, labels = load_recognizer()
    except FileNotFoundError as e:
        print(e)
        return None, None

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Cannot open camera for face recognition.")
        return None, None

    start = time.time()
    predicted_name = None
    predicted_conf = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            try:
                face = cv2.resize(face, (200, 200))
            except Exception:
                continue

            label, conf = recognizer.predict(face)
            name = labels.get(label, "Unknown")

            # Draw results
            color = (0, 255, 0) if conf < confidence_threshold else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} ({conf:.1f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Only record if confident
            if conf < confidence_threshold:
                predicted_name = name
                predicted_conf = conf

        elapsed = int(timeout - (time.time() - start))
        cv2.putText(frame, f"Time left: {elapsed}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Face Verify", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        # Return as soon as a confident match is found
        if predicted_name is not None:
            cap.release()
            cv2.destroyAllWindows()
            return predicted_name, predicted_conf

        if time.time() - start > timeout:
            break

    cap.release()
    cv2.destroyAllWindows()
    return None, None


if __name__ == "__main__":
    print("🔍 Testing face recognition...")
    name, conf = recognize_face()
    if name:
        print(f"✅ Recognized: {name} (confidence: {conf:.1f})")
    else:
        print("❌ Face not recognized.")
