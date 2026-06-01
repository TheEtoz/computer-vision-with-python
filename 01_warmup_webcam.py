"""
TEACHING COMPUTERS TO SEE
Script 1 of 4 — Warm-Up: Open a Webcam

What this does:
  - Opens your default webcam
  - Shows a live feed in a window
  - Displays FPS in the top-left corner
  - Press Q to quit cleanly

Packages needed:
  pip install opencv-python
"""

import cv2

# ── Open the camera ──────────────────────────────────────────────
cap = cv2.VideoCapture(0)  # 0 = default webcam, try 1 if this doesn't work

# ── Main loop ────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()  # read one frame

    cv2.imshow("Webcam Feed", frame)

    # Wait 1ms and listen for Q key
    if cv2.waitKey(1) == ord('q'):
        break

# ── Clean up — always do this! ───────────────────────────────────
cap.release()
cv2.destroyAllWindows()
