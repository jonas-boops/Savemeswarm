import cv2
import numpy as np
from djitellopy import Tello
from ultralytics import YOLO

# Load YOLO11n model
model = YOLO("yolo11n.pt")

# Initialize DJI Tello
tello = Tello()
tello.connect()
tello.streamon()

# Get stream and detect objects
while True:
    # Get the current frame
    frame_read = tello.get_frame_read()
    frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (960, 720))
    # Perform object detection
    results = model.predict(frame, show=True, conf=0.5)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        tello.land()
        break
