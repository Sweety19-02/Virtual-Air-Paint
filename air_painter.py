"""
Professional AR Air Painter & Drawing Canvas
Description: Allows users to paint in the air using their index fingertip, 
             select colors from an on-screen palette, and clear the canvas dynamically.
"""

import os
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# --- CONFIGURATION CONSTANTS ---
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"
CAMERA_INDEX = 0

def download_model_if_missing(path, url):
    if not os.path.exists(path):
        print("[INFO] Downloading MediaPipe Hand Landmarker model...")
        try:
            urllib.request.urlretrieve(url, path)
            print("[INFO] Model downloaded successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to download model: {e}")
            exit(1)

def main():
    download_model_if_missing(MODEL_PATH, MODEL_URL)

    # Initialize MediaPipe Hand Landmarker Detector
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options, 
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FPS, 60)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    init_time = time.time()
    
    # Drawing Canvas and State Variables
    canvas = None
    xp, yp = 0, 0
    brush_thickness = 8
    eraser_thickness = 40
    
    # Default color: Blue (BGR format)
    current_color = (255, 0, 0)
    color_name = "BLUE"

    print("[INFO] AR Air Painter is running. Press 'c' to clear canvas, 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # Initialize canvas once frame dimensions are known
        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # --- DRAW HEADER / PALETTE ON FRAME ---
        # Palette boxes: Blue, Green, Red, Yellow, Clear / Eraser
        cv2.rectangle(frame, (40, 10), (140, 70), (255, 0, 0), cv2.FILLED)       # Blue
        cv2.putText(frame, "BLUE", (60, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.rectangle(frame, (160, 10), (260, 70), (0, 255, 0), cv2.FILLED)     # Green
        cv2.putText(frame, "GREEN", (175, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.rectangle(frame, (280, 10), (380, 70), (0, 0, 255), cv2.FILLED)     # Red
        cv2.putText(frame, "RED", (310, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.rectangle(frame, (400, 10), (500, 70), (0, 255, 255), cv2.FILLED)   # Yellow
        cv2.putText(frame, "YELLOW", (415, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.rectangle(frame, (520, 10), (620, 70), (50, 50, 50), cv2.FILLED)     # Eraser
        cv2.putText(frame, "ERASER", (535, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Convert frame for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        current_time = time.time()
        frame_timestamp_ms = int((current_time - init_time) * 1000)
        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        if detection_result.hand_landmarks:
            landmarks = detection_result.hand_landmarks[0]

            # Index fingertip (landmark 8) and Thumb tip (landmark 4)
            index_x, index_y = int(landmarks[8].x * w), int(landmarks[8].y * h)
            thumb_x, thumb_y = int(landmarks[4].x * w), int(landmarks[4].y * h)

            # Check if index and middle fingers are up (Selection / Hover mode)
            index_extended = landmarks[8].y < landmarks[6].y
            middle_extended = landmarks[12].y < landmarks[10].y

            # Draw index tip cursor
            cv2.circle(frame, (index_x, index_y), 10, current_color, cv2.FILLED)

            # SELECTION MODE: Both index and middle fingers raised up
            if index_extended and middle_extended:
                xp, yp = 0, 0  # Reset drawing anchor so lines don't connect across modes
                
                # Check if touching header palette
                if index_y < 80:
                    if 40 < index_x < 140:
                        current_color = (255, 0, 0)
                        color_name = "BLUE"
                    elif 160 < index_x < 260:
                        current_color = (0, 255, 0)
                        color_name = "GREEN"
                    elif 280 < index_x < 380:
                        current_color = (0, 0, 255)
                        color_name = "RED"
                    elif 400 < index_x < 500:
                        current_color = (0, 255, 255)
                        color_name = "YELLOW"
                    elif 520 < index_x < 620:
                        current_color = (0, 0, 0)  # Eraser color
                        color_name = "ERASER"

            # DRAWING MODE: Only Index finger raised up
            elif index_extended and not middle_extended:
                if xp == 0 and yp == 0:
                    xp, yp = index_x, index_y

                # If Eraser is active
                if current_color == (0, 0, 0):
                    cv2.line(canvas, (xp, yp), (index_x, index_y), (0, 0, 0), eraser_thickness)
                    cv2.circle(frame, (index_x, index_y), eraser_thickness // 2, (200, 200, 200), 2)
                else:
                    cv2.line(canvas, (xp, yp), (index_x, index_y), current_color, brush_thickness)

                xp, yp = index_x, index_y
            else:
                xp, yp = 0, 0

        # Merge canvas drawing onto camera frame using bitwise operations
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, inv_canvas = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
        inv_canvas = cv2.cvtColor(inv_canvas, cv2.COLOR_GRAY2BGR)
        
        frame = cv2.bitwise_and(frame, inv_canvas)
        frame = cv2.bitwise_or(frame, canvas)

        # Status text overlay
        cv2.putText(frame, f"Tool: {color_name}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("AR Air Painter", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            print("[INFO] Canvas cleared!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()