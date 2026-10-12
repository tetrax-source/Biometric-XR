"""
liveness.py
Interactive blink detection module.
Uses core.liveness.LivenessDetector with MediaPipe 1.x & OpenCV fallback.
"""

import time
import cv2
from core.liveness import LivenessDetector

def check_blink(timeout=8, required_blinks=1, ear_thresh=0.22, camera_index=0):
    detector = LivenessDetector(ear_threshold=ear_thresh, required_blinks=required_blinks)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Cannot open camera for liveness.")
        return False

    start = time.time()
    passed = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        passed, ear, blinks, ann_frame = detector.process_frame(frame)

        elapsed = int(timeout - (time.time() - start))
        cv2.putText(ann_frame, f"Time left: {elapsed}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Liveness Detection", ann_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        if passed:
            break

        if time.time() - start > timeout:
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    return passed

if __name__ == "__main__":
    ok = check_blink()
    print("Liveness Passed" if ok else "Liveness Failed")
