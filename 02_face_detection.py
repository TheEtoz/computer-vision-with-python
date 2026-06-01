"""
TEACHING COMPUTERS TO SEE
Script 2 of 4 — Face Detection

What this does:
  - Opens your webcam
  - Detects faces in real time using a pre-trained Haar cascade
  - Draws a green bounding box around each face
  - Shows face count and FPS on screen
  - Press Q to quit

Packages needed:
  pip install opencv-python
"""

import cv2

# ── Load the pre-trained face cascade ────────────────────────────
# OpenCV ships this XML file — no download needed
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)


# ── Open camera ──────────────────────────────────────────────────
cap = cv2.VideoCapture(0)

# ── Main loop ────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale — Haar cascade works on single channel
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    # scaleFactor=1.1  → how much image size is reduced at each scale
    # minNeighbors=5   → how many neighbors each candidate needs (higher = stricter)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
    )

    # Draw bounding box for each detected face
    for i, (x, y, w, h) in enumerate(faces):
        # Green rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Label above the box
        cv2.putText(
            frame,
            f"Face {i + 1}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Face count — top left
    face_count = len(faces)
    cv2.putText(
        frame,
        f"Faces detected: {face_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),             # yellow
        2
    )

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# ── Clean up ─────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
