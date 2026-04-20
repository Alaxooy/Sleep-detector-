import io
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import simpleaudio as sa
from scipy.spatial import distance


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = distance.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear


def create_beep_wave(frequency=880, duration=0.35, sample_rate=44100, volume=0.4):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(frequency * t * 2 * np.pi)
    audio = tone * (2**15 - 1) * volume
    audio = audio.astype(np.int16)
    return sa.WaveObject(audio.tobytes(), 1, 2, sample_rate)


def play_alert(sound):
    try:
        sound.play()
    except Exception:
        pass


def draw_status_overlay(frame, ear, counter, alert_active):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    status_text = "ALERT" if alert_active else "SAFE"
    status_color = (0, 0, 255) if alert_active else (0, 255, 0)

    cv2.putText(frame, f"EAR: {ear:.2f}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Closed frames: {counter}", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, status_text, (frame.shape[1] - 140, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)


def draw_eye_contour(frame, coords, color):
    pts = np.array(coords, np.int32)
    cv2.polylines(frame, [pts], True, color, 1, cv2.LINE_AA)
    for (x, y) in coords:
        cv2.circle(frame, (x, y), 2, color, -1)


class DrowsinessDetector:
    def __init__(self, webcam_index=0):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self.left_eye_indices = [33, 160, 158, 133, 153, 144]
        self.right_eye_indices = [263, 387, 385, 362, 380, 373]
        self.EYE_AR_THRESH = 0.22
        self.EYE_AR_CONSEC_FRAMES = 30  # 1.5 seconds for faster alert
        self.counter = 0
        self.alert_sound = create_beep_wave(frequency=880, duration=0.2, volume=0.5)
        self.camera_index = webcam_index
        self.capture = None
        self.frame = None
        self.display_frame = None
        self.running = False
        self.alert_active = False
        self.current_ear = 0.0
        self.lock = threading.Lock()
        self.thread = None
        self.input_width = 640
        self.input_height = 480

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _run(self):
        print(f"[DEBUG] Attempting to open camera at index: {self.camera_index}")
        self.capture = cv2.VideoCapture(self.camera_index)
        if not self.capture.isOpened():
            print(f"[ERROR] Failed to open camera at index {self.camera_index}")
            self.running = False
            return
        
        # Set camera resolution for performance
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.input_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.input_height)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[DEBUG] Camera opened successfully at {self.input_width}x{self.input_height}")

        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = face_landmarks.landmark
                left_eye_coords = self._get_eye_coords(landmarks, self.left_eye_indices, frame)
                right_eye_coords = self._get_eye_coords(landmarks, self.right_eye_indices, frame)
                leftEAR = eye_aspect_ratio(left_eye_coords)
                rightEAR = eye_aspect_ratio(right_eye_coords)
                ear = (leftEAR + rightEAR) / 2.0

                with self.lock:
                    self.current_ear = ear
                    if ear < self.EYE_AR_THRESH:
                        self.counter += 1
                        # Alert when threshold is reached and play continuously while closed
                        if self.counter >= self.EYE_AR_CONSEC_FRAMES:
                            self.alert_active = True
                            play_alert(self.alert_sound)  # Play every frame for nonstop alert
                    else:
                        self.counter = 0
                        self.alert_active = False

                draw_status_overlay(frame, ear, self.counter, self.alert_active)
                draw_eye_contour(frame, left_eye_coords, (0, 255, 0))
                draw_eye_contour(frame, right_eye_coords, (0, 255, 0))
            else:
                with self.lock:
                    self.current_ear = 0.0
                    self.counter = 0
                    self.alert_active = False

            with self.lock:
                self.display_frame = frame

    def _get_eye_coords(self, landmarks, indices, frame):
        coords = []
        for i in indices:
            lm = landmarks[i]
            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])
            coords.append((x, y))
        return coords

    def get_status(self):
        with self.lock:
            return {
                "running": self.running,
                "ear": float(self.current_ear),
                "alert": bool(self.alert_active),
                "closed_frames": int(self.counter),
            }

    def get_snapshot(self):
        with self.lock:
            if self.display_frame is None:
                return None
            # Ultra-low JPEG quality (40) for fastest encoding
            success, jpeg = cv2.imencode('.jpg', self.display_frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
            return jpeg.tobytes() if success else None


def main():
    detector = DrowsinessDetector()
    detector.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        detector.stop()


if __name__ == "__main__":
    main()
