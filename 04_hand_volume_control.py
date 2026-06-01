"""
TEACHING COMPUTERS TO SEE
Script 4 of 4 — Hand Landmark Volume Control

Packages needed:
  py -3.11 -m pip install opencv-python mediapipe pyautogui
"""

import cv2
import mediapipe as mp
from mediapipe import solutions
import pyautogui
import math

# ── Initialize MediaPipe Hands ───────────────────────────────────
mp_hands = solutions.hands
hands    = mp_hands.Hands()
mp_draw  = solutions.drawing_utils

# ── Initialize webcam ────────────────────────────────────────────
webcam = cv2.VideoCapture(0)

x1, y1, x2, y2 = 0, 0, 0, 0

while True:
    success, img = webcam.read()
    if not success:
        break

    img     = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output  = hands.process(img_rgb)

    if output.multi_hand_landmarks:
        for hand in output.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

            h, w, _ = img.shape

            for id, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id == 8:   # index finger tip
                    x1, y1 = cx, cy
                    cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)

                if id == 4:   # thumb tip
                    x2, y2 = cx, cy
                    cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)

            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            dist = math.hypot(x2 - x1, y2 - y1) / 4

            if dist > 50:
                pyautogui.press("volumeup")
                action, col = "VOL UP",   (0, 255, 100)
            else:
                pyautogui.press("volumedown")
                action, col = "VOL DOWN", (0, 100, 255)

            cv2.putText(img, f"Dist: {int(dist)}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, action, (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)

    cv2.imshow("Hand Volume Control", img)

    if cv2.waitKey(1) == 27:   # Esc to quit
        break

webcam.release()
cv2.destroyAllWindows()