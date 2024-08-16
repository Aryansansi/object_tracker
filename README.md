# Object Detection Project

This project implements object detection using both a custom model based on VGG16 and the YOLOv5 model. The project consists of various Python scripts that perform tasks such as capturing frames, processing video streams, and detecting objects in real-time.

## Project Structure

- **cache_dump.py**: Handles caching of data in memory, calculating FPS for video streams, and provides functions for serializing and deserializing data using `pickle`.
  
- **Custom_model.py**: Implements a custom object detection model based on the VGG16 architecture. It captures and processes frames from an RTSP stream, performs object detection, and displays the results.
  
- **detect.py**: Uses a pre-trained YOLOv5 model for object detection on video streams or images. It processes each frame, performs detection, and renders the results.
  
- **model.py**: Loads the YOLOv5 model and performs object detection on RTSP streams. It includes a retry mechanism to handle stream access issues and processes frames in real-time.
  
- **optimize.py**: Optimizes the YOLOv5 model for better performance, particularly on CUDA-enabled devices. It processes video frames and renders detection results.
  
- **trained_model.py**: Loads a pre-trained YOLOv5 model, captures frames from an RTSP stream, processes them for object detection, and displays the results in real-time.
  
- **urls.py**: Contains the RTSP stream URLs and provides a function to retrieve the URL for camera streams.

## Requirements

- Python 3.7+
- OpenCV
- TensorFlow
- PyTorch
- CUDA (optional, for GPU acceleration)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <github.com/Aryansansi/object_tracker>
   cd Object-Detection-Project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv5 model weights**:
   Ensure that the YOLOv5 weights (`yolov5s.pt`) are downloaded and placed in the project directory.

## Usage

1. **Using Custom Model**:
   - Run the `Custom_model.py` script to perform object detection using the custom VGG16-based model.
   ```bash
   python Custom_model.py
   ```

2. **Using YOLOv5 Model**:
   - Run the `detect.py` script to perform object detection using the YOLOv5 model on a video or image.
   ```bash
   python detect.py <path_to_video_or_image>
   ```
   - Run the `model.py` or `trained_model.py` scripts to detect objects in real-time from an RTSP stream.
   ```bash
   python model.py
   ```
   or
   ```bash
   python trained_model.py
   ```

## Customization

- **Modify Class Names**: Update the `class_names` list in `Custom_model.py` to match the classes your model is trained to detect.
- **Change RTSP URLs**: Update the `RTSP_URLS` variable in `urls.py` to use different RTSP streams.
- **Adjust Image Dimensions**: Modify `img_height` and `img_width` in `Custom_model.py` to match the input size required by your model.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) for the pre-trained YOLOv5 model.
