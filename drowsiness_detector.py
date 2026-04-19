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

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Eye landmark indices for left and right eyes
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

# Thresholds
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 15  # about 0.5-1 second at 24fps

COUNTER = 0
ALERT_SOUND = create_beep_wave(frequency=880, duration=0.35, volume=0.4)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam. Please check camera permissions in System Settings > Privacy & Security > Camera.")
    print("Make sure to allow access for Terminal or your code editor.")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Face Mesh
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Get left eye coordinates
            left_eye_coords = []
            for i in LEFT_EYE:
                lm = landmarks[i]
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                left_eye_coords.append((x, y))

            # Get right eye coordinates
            right_eye_coords = []
            for i in RIGHT_EYE:
                lm = landmarks[i]
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                right_eye_coords.append((x, y))

            # Calculate EAR
            leftEAR = eye_aspect_ratio(left_eye_coords)
            rightEAR = eye_aspect_ratio(right_eye_coords)
            ear = (leftEAR + rightEAR) / 2.0

            alert_active = False
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    alert_active = True
                    play_alert(ALERT_SOUND)
            else:
                COUNTER = 0

            draw_status_overlay(frame, ear, COUNTER, alert_active)
            draw_eye_contour(frame, left_eye_coords, (0, 255, 0))
            draw_eye_contour(frame, right_eye_coords, (0, 255, 0))

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()