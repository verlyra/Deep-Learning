import cv2
import os

#KONFIGURASI
PERSON_NAME = "2311501601"
DATASET_DIR = "dataset"
TOTAL_IMAGES = 100
IMG_SIZE = 100
CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Folder
person_path = os.path.join(DATASET_DIR, PERSON_NAME)
os.makedirs(person_path, exist_ok=True)

#Load Haar Cascade
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

#open webcam
cap = cv2.VideoCapture(0)
count = 0

print("[INFO] Capture dimulai...")
print("[INFO] Tekan 'q' untuk keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

        if count < TOTAL_IMAGES:
            filename = f"img_{count:03d}.jpg"
            filepath = os.path.join(person_path, filename)
            cv2.imwrite(filepath, face)
            count += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"Images: {count}/{TOTAL_IMAGES}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if count >= TOTAL_IMAGES:
        print("[INFO] Dataset capture selesai")
        break

cap.release()
cv2.destroyAllWindows()
