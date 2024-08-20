import cv2
import torch
import threading
import queue
import time
from urls import RTSP_URLS
from win32api import GetSystemMetrics

# Load the YOLOv5 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to(device)
model.eval()  # Set the model to evaluation model

# Define the RTSP stream URL
rtsp_url = RTSP_URLS

print(f"RTSP URL: {rtsp_url}")

# Open the RTSP stream using OpenCV
if isinstance(rtsp_url, str):
    cap = cv2.VideoCapture(rtsp_url)
else:
    print("Error: RTSP_URLS is not a string")
    exit()

if not cap.isOpened():
    print(f"Error: Unable to open video source {rtsp_url}")
    exit()

# Get screen size
screen_width = GetSystemMetrics(0)  # Screen width
screen_height = GetSystemMetrics(1) # Screen height

# Set frame size to fit within the monitor screen (e.g., half of the screen size)
frame_width = screen_width // 2
frame_height = screen_height // 2

# Set the desired frame rate
desired_fps = 30
frame_interval = 1 / desired_fps

# Queues for frames
frame_queue = queue.Queue(maxsize=1)

def capture_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)

def process_frames():
    prev_time = time.time()
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Resize frame for faster processing
            frame = cv2.resize(frame, (frame_width, frame_height))

            # Perform object detection
            results = model(frame)

            # Render the results on the frame
            frame = results.render()[0]

            # Display the frame
            cv2.imshow('YOLOv5 Object Detection', frame)

            # Maintain the desired frame rate
            current_time = time.time()
            elapsed_time = current_time - prev_time
            if elapsed_time < frame_interval:
                time.sleep(frame_interval - elapsed_time)
            prev_time = current_time

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()  # Terminate the script

# Start threads
capture_thread = threading.Thread(target=capture_frames)
process_thread = threading.Thread(target=process_frames)

capture_thread.start()
process_thread.start()

capture_thread.join()
process_thread.join()

# Release the capture and close the display window
cap.release()
cv2.destroyAllWindows()
