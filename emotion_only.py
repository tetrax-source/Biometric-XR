import cv2
from deepface import DeepFace

# Camera start
cap = cv2.VideoCapture(0)

print("Press Q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False
        )

        emotions = result[0]["emotion"]

        # ✅ Smart emotion decision (ONLY 3 emotions)
        if emotions["happy"] > emotions["sad"] and emotions["happy"] > emotions["neutral"]:
            final_emotion = "Happy"
        elif emotions["sad"] > emotions["neutral"]:
            final_emotion = "Sad"
        else:
            final_emotion = "Neutral"

    except:
        final_emotion = "Neutral"

    # Display emotion
    cv2.putText(
        frame,
        f"Emotion: {final_emotion}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
