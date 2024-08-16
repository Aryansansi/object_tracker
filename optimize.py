import torch
import cv2
import time

def optimize_model(model, device='cpu'):
    """
    Optimize the YOLOv5 model for better performance.
    
    Parameters:
    - model: The loaded YOLOv5 model.
    - device: The device to move the model to (e.g., 'cpu', 'cuda').
    
    Returns:
    - Optimized model.
    """

    model.to(device)

    model.eval()
    
    if device == 'cuda':
        model.half()
    
    return model

def process_frame(frame, model, device='cpu'):
    """
    Process a single frame using the YOLOv5 model.
    
    Parameters:
    - frame: The frame to process.
    - model: The YOLOv5 model.
    - device: The device to move the frame to (e.g., 'cpu', 'cuda').
    
    Returns:
    - The processed frame with detections.
    """
    # Convert frame to tensor
    img = torch.from_numpy(frame).to(device).float()
    img /= 255.0  # Normalize to [0, 1]
    
    # Add batch dimension
    img = img.unsqueeze(0)
    
    # Perform inference
    with torch.no_grad():
        results = model(img)
    
    # Process results and draw bounding boxes (for demonstration)
    # results.xyxy[0] contains the bounding boxes
    for *box, conf, cls in results.xyxy[0]:
        if conf > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def main(video_source):
    # Load the YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    # Optimize the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = optimize_model(model, device)
    
    # Open video source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Set frame rate for processing
    target_fps = 30
    frame_time = 1.0 / target_fps
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        frame = process_frame(frame, model, device)
        
        # Display the frame
        cv2.imshow('YOLOv5 Object Detection', frame)
        
        # Control frame rate
        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_time - elapsed_time % frame_time)
        time.sleep(sleep_time)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_source = 0  # Change to your video source or RTSP stream URL
    main(video_source)
