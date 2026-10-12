import cv2
import os

# Load face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
if face_cascade.empty():
    print("❌ Error: Haar cascade not loaded.")
    exit()

# Get user name
name = input("Enter your name: ").strip()
if name == "":
    print("❌ Invalid name.")
    exit()

# Save faces inside DATASET folder (IMPORTANT)
path = f"dataset/{name}"
os.makedirs(path, exist_ok=True)

# Start webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("❌ Cannot access webcam.")
    exit()

count = 0
MAX_IMAGES = 30

print("📸 Look at the camera...")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) > 0:
        # Take only the biggest face
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        (x, y, w, h) = faces[0]

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        count += 1
        cv2.imwrite(f"{path}/{count}.jpg", face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"Images: {count}/{MAX_IMAGES}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Face Enrollment", frame)

    if cv2.waitKey(1) & 0xFF == 27 or count >= MAX_IMAGES:
        break

cam.release()
cv2.destroyAllWindows()

print(f"✅ Face enrollment completed for {name}")
