import torch
import cv2
import time
import threading
import numpy as np
from urls import get_camera_url

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLOv5 model
def load_model(weights='yolov5s.pt'):
    print(f"Loading model with weights: {weights}")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=True)
    model.to(device)
    if device == 'cuda':
        model.half()  # Use half precision on CUDA
    return model

# Inference function
def infer(model, source, img_size=640):
    cap = cv2.VideoCapture(source)

    # Retry mechanism to handle stream access issues
    retries = 5
    while retries > 0:
        if cap.isOpened():
            break
        print(f"Retrying to open video source {source}...")
        time.sleep(5)  # Wait before retrying
        retries -= 1

    if not cap.isOpened():
        print(f"Error: Unable to open video source {source}")
        return

    frame_queue = []
    frame_lock = threading.Lock()

    def capture_frames():
        nonlocal frame_queue
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            with frame_lock:
                frame_queue.append(frame)

    def process_frames():
        nonlocal frame_queue
        while True:
            with frame_lock:
                if not frame_queue:
                    continue
                frame = frame_queue.pop(0)
            
            # Resize frame to match YOLOv5 input size
            frame_resized = cv2.resize(frame, (img_size, img_size))

            # Convert frame to tensor
            img_tensor = torch.from_numpy(frame_resized).to(device).float()
            img_tensor /= 255.0  # Normalize
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # Change to CHW format

            # Perform inference
            with torch.no_grad():
                results = model(img_tensor)
            
            # Draw results on the frame
            annotated_frame = results.render()[0]  # Render the results on the frame

            # Display the frame
            cv2.imshow('RTSP Stream', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Start frame capture and processing threads
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    process_thread = threading.Thread(target=process_frames, daemon=True)
    capture_thread.start()
    process_thread.start()

    capture_thread.join()
    process_thread.join()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load the model
    weights_path = "yolov5s.pt"  # Ensure this path is correct for your weights file
    model = load_model(weights=weights_path)

    # Get RTSP stream URL from urls.py
    rtsp_index = 0  # Set this to the appropriate index for the RTSP URL you want to use
    source = get_camera_url(rtsp_index)

    # Perform inference on the RTSP stream
    infer(model, source, img_size=640)
