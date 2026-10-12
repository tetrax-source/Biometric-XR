import cv2
import os
import time

# ==============================
# Load Haar Cascade Face Detector
# ==============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("❌ Haar cascade not loaded")
    exit()

# ==============================
# Ask User Name
# ==============================
name = input("Enter your name: ").strip().replace(" ", "_")

if name == "":
    print("❌ Invalid name")
    exit()

save_path = f"images/{name}"
os.makedirs(save_path, exist_ok=True)

# ==============================
# Start Webcam
# ==============================
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("❌ Cannot access webcam")
    exit()

print("📸 Look at the camera... Collecting face data")

count = 0
MAX_IMAGES = 30

# ==============================
# Capture Loop
# ==============================
while count < MAX_IMAGES:
    ret, frame = cam.read()
    if not ret:
        print("❌ Failed to capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        cv2.imwrite(f"{save_path}/{count}.jpg", face)
        count += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"Images: {count}/{MAX_IMAGES}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow("Automatic Face Enrollment", frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()

print(f"✅ Face enrollment complete for {name}")
print(f"📁 Saved {count} images in {save_path}")
