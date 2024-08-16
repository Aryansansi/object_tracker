import cv2
import threading
import queue
import time
from urls import RTSP_URLS
from win32api import GetSystemMetrics
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.applications import VGG16 # type: ignore

# Define the list of class names
class_names = ['USB', 'server rack', 'router', 'phone', 'person', 'mouse', 'laptop', 'keyboard', 'dog', 'cat', 'bottle']

# Define the VGG16-based model architecture
def create_model(img_height, img_width, num_classes):
    base_model = VGG16(input_shape=(img_height, img_width, 3),
                       include_top=False,
                       weights='imagenet')
    base_model.trainable = False  # Freeze the base model

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')  # Multi-class classification
    ])

    model.build((None, img_height, img_width, 3))  # Explicitly define the input shape
    return model

# Load model with pre-trained weights
def load_model_with_weights(weights_path, img_height, img_width, num_classes):
    model = create_model(img_height, img_width, num_classes)
    model.load_weights(weights_path)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    return model

# Load the model and weights
img_height = 224  # Replace with the appropriate image height
img_width = 224   # Replace with the appropriate image width
num_classes = len(class_names)  # Number of classes
weights_path = 'model_weights.weights.h5'  # Path to your weights file

model = load_model_with_weights(weights_path, img_height, img_width, num_classes)

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

            # Preprocess the frame to match the input requirements of the model
            preprocessed_frame = cv2.resize(frame, (img_height, img_width))  # Resize to model's input size
            preprocessed_frame = preprocessed_frame.astype(np.float32) / 255.0  # Normalize the image
            preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)  # Add batch dimension

            # Perform object detection
            predictions = model.predict(preprocessed_frame)

            # Interpret the predictions
            class_id = np.argmax(predictions[0])  # Get the class with the highest probability
            confidence = np.max(predictions[0])   # Get the confidence of the prediction
            class_name = class_names[class_id]    # Get the actual class name

            # Draw text on the frame
            label = f"Class: {class_name}, Confidence: {confidence:.2f}"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('Custom Model-based Object Detection', frame)

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
