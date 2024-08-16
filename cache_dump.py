import pickle
import cv2
import time

# Function to "delete" the frame by simply passing over it (no saving)
def delete_frame(frame):
    pass

# Save the cache data in memory without storing it in the file system
def save_cache(data):
    # Serialize the data using pickle but do not save to disk
    serialized_data = pickle.dumps(data)
    return serialized_data

# Load the cache from serialized data in memory (not from disk)
def load_cache(serialized_data):
    # Deserialize the data from the in-memory serialized form
    data = pickle.loads(serialized_data)
    return data

# Calculate and store the FPS of the video stream
def calculate_fps(video_source):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        elapsed_time = time.time() - start_time
        
        if elapsed_time > 1.0:  # Calculate FPS every second
            fps = frame_count / elapsed_time
            start_time = time.time()
            frame_count = 0
        
        # Optionally display FPS on the frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Video Stream', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return fps

# Example usage
if __name__ == "__main__":
    video_source = 0  # Change to your video source or RTSP stream URL
    fps = calculate_fps(video_source)
    if fps is not None:
        print(f"Video FPS: {fps:.2f}")
        # Optionally save or use the FPS value as needed
