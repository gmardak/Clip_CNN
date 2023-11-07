from PIL import Image, ImageOps
import numpy as np
import cv2

# Function to capture a single image from the webcam
def capture_image_from_webcam_single():
    cap = cv2.VideoCapture(2)  # Change to 0 if it is your laptop's camera
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        print("Can't receive frame. Exiting ...")
        exit()
    cap.release()  # When everything done, release the capture
    return frame