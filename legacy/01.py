import cv2
import os
import time

# =========================
# Load Haar Cascade Model
# =========================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    print("❌ Error: Haar cascade file not found!")
    exit()

# =========================
# Get User Input
# =========================
name = input("Enter your name: ").strip().replace(" ", "_")

# Validate name
if name == "":
    print("❌ Invalid name. Exiting...")
    exit()

path = f"images/{name}"
os.makedirs(path, exist_ok=True)

# =========================
# Start Webcam
# =========================
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("❌ Error: Unable to access webcam!")
    exit()

print("📸 Starting image capture... Press ESC to quit early.")

count = 0
MAX_IMAGES = 30  # Number of images per person

# =========================
# Capture Loop
# =========================
while True:
    ret, frame = cam.read()
    if not ret:
        print("❌ Failed to read frame from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Improve face detection accuracy

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        # Filter out blurry faces
        blur = cv2.Laplacian(face, cv2.CV_64F).var()
        if blur < 80:
            cv2.putText(frame, "Move closer / Avoid blur", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            continue

        face = cv2.resize(face, (200, 200))

        img_path = os.path.join(path, f"{count}.jpg")
        cv2.imwrite(img_path, face)
        count += 1

        # Draw face box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display progress
    cv2.putText(frame, f"Images Collected: {count}/{MAX_IMAGES}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Face Collector", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        print("⚠️ Early exit requested.")
        break

    if count >= MAX_IMAGES:
        print("✅ Data collection complete.")
        break

cam.release()
cv2.destroyAllWindows()

print(f"📁 Saved {count} images to: {path}")
