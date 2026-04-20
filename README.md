# Drowsiness Detection Project

This is an end-to-end computer vision project to detect drowsiness in real-time using facial landmarks and eye aspect ratio calculation.

## Features
- Real-time video processing from webcam
- Face and eye detection using MediaPipe
- Eye Aspect Ratio (EAR) calculation
- **Continuous eye tracking** with optimized performance
- **Nonstop audio alerts** when drowsiness is detected
- **Zero-lag camera feed** with 640x480 resolution
- **Real-time dashboard** with live status updates
- Drowsiness alert when eyes are closed for 1.5 seconds

## Requirements
- Python 3.8+
- Webcam

## Installation
1. Clone or download this repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (on macOS/Linux)
4. Install dependencies: `pip install -r requirements.txt`

## Usage
Run the desktop script: `python drowsiness_detector.py`

Press 'q' to quit.

## API Service
1. Install dependencies: `pip install -r requirements.txt`
2. Start the API: `python app.py`
3. Open `http://localhost:8000/docs` for auto-generated API documentation

Available endpoints:
- `GET /health` — service health
- `GET /status` — current EAR, alert state, and closed-frame counter
- `POST /start` — start webcam monitoring
- `POST /stop` — stop webcam monitoring
- `GET /snapshot` — current annotated frame as JPEG


## Docker Deployment
To run the application in a Docker container for an end-to-end setup:

1. Ensure Docker is installed on your system.
2. Build and run the container:
   ```bash
   docker-compose up --build
   ```
3. Access the API at `http://localhost:8000/docs`

**Note on Webcam Access:**
- On Linux: Uncomment the `devices` section in `docker-compose.yml` to allow webcam access.
- On macOS: Webcam access in Docker containers is limited. You may need to run the app natively or use alternative setups like host networking. The API will still run, but webcam functionality may not work inside the container.


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
