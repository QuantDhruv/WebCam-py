# Webcam Capture & Analysis Application

## Overview
The **Webcam Capture & Analysis Application** is a Python-based tool that utilizes OpenCV and Streamlit to capture webcam images, apply real-time filters, and analyze video feeds. Users can choose from multiple image processing features, including edge detection, face detection, and motion detection.

## Features
- **Live Webcam Feed**: Captures and processes video from the webcam in real time.
- **Edge Detection**: Uses the Canny edge detection algorithm to highlight object boundaries.
- **Face Detection**: Detects faces using OpenCV's Haar Cascade classifier and marks them with bounding boxes.
- **Motion Detection**: Identifies movement by analyzing frame differences.
- **Image Capture & Storage**: Allows users to capture and save processed images.
- **Streamlit UI**: Provides an interactive web-based interface for feature selection and image visualization.

## Usage
1. Run the application:
   ```bash
   streamlit run app.py
   ```
2. Use the sidebar to select a processing feature:
   - **None**: Displays the raw webcam feed.
   - **Edge Detection**: Applies edge detection to the video stream.
   - **Face Detection**: Identifies and marks faces in the feed.
   - **Motion Detection**: Highlights motion changes.
3. Click "Capture Image" to save the processed frame as a `.jpg` file.

## Project Structure
```
├── app.py               # Main application script
├── requirements.txt     # Dependencies list
├── README.md            # Project documentation
└── captured_images/     # Directory for saved images (optional)
```

## Future Enhancements
- Implement a background subtraction method for improved motion detection.
- Enable video recording and storage of processed footage.
- Add more real-time filters like object detection and color-based segmentation.
- Develop a cloud-based storage system for captured images.
