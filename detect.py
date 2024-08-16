import cv2
import torch
import time

def detect_objects(frame, model):
    """
    Detect objects in a frame using the YOLOv5 model.
    
    Parameters:
    - frame: The image frame to process.
    - model: The loaded YOLOv5 model.
    
    Returns:
    - Processed frame with detections rendered.
    """
    # Convert frame to tensor
    img = torch.from_numpy(frame).float().unsqueeze(0)
    img /= 255.0  # Normalize to [0, 1]
    
    # Perform object detection
    results = model(img)
    
    # Render the results on the frame
    frame = results.render()[0]
    
    return frame

def main(input_path):
    # Load the YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()  # Set the model to evaluation mode

    # Open the input stream
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Unable to open video source {input_path}")
        return

    # Set the desired frame rate
    desired_fps = 30
    frame_time = 1.0 / desired_fps

    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame to a smaller resolution for faster processing
        frame = cv2.resize(frame, (640, 480))

        # Detect objects in the frame
        frame = detect_objects(frame, model)
        
        # Display the frame
        cv2.imshow('YOLOv5 Object Detection', frame)
        
        # Calculate elapsed time and sleep if necessary
        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_time - elapsed_time)
        time.sleep(sleep_time)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys

    # Example usage from command line
    if len(sys.argv) != 2:
        print("Usage: python detect.py <path_to_video_or_image>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    main(input_path)
