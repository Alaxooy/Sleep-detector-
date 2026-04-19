# Drowsiness Detection Project

This is an end-to-end computer vision project to detect drowsiness in real-time using facial landmarks and eye aspect ratio calculation.

## Features
- Real-time video processing from webcam
- Face and eye detection using MediaPipe
- Eye Aspect Ratio (EAR) calculation
- Drowsiness alert when eyes are closed for too long

## Requirements
- Python 3.8+
- Webcam

## Installation
1. Clone or download this repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (on macOS/Linux)
4. Install dependencies: `pip install -r requirements.txt`

## Usage
Run the script: `python drowsiness_detector.py`

Press 'q' to quit.

## How it works
- Detects faces in the video stream using MediaPipe Face Mesh
- Identifies eye landmarks
- Calculates the Eye Aspect Ratio (EAR)
- If EAR is below a threshold for a consecutive number of frames, triggers an alert

## Learning Outcomes
- Computer vision basics with OpenCV
- Facial landmark detection with MediaPipe
- Real-time video processing
- Algorithm implementation for drowsiness detection

## Next Steps for End-to-End
- Add data collection for training a ML model
- Train a classifier for drowsiness
- Deploy as a web app or mobile app
- Add more features like yawn detection or head pose estimation

## Resources
- [MediaPipe Documentation](https://mediapipe.dev/)
- [OpenCV Documentation](https://docs.opencv.org/)
- Computer Vision courses on Coursera/YouTube