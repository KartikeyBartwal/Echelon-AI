import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Blinking detection parameters
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
BLINK_THRESHOLD = 0.20

# Initialize variables for blink detection
blink_count = 0
last_blink_time = time.time()

# Capture video
cap = cv2.VideoCapture(0)

def eye_aspect_ratio(eye_landmarks, landmarks):
    """Calculate eye aspect ratio (EAR) to detect blinks."""
    left_pt = np.linalg.norm(np.array([landmarks[eye_landmarks[1]].x, landmarks[eye_landmarks[1]].y]) - 
                             np.array([landmarks[eye_landmarks[5]].x, landmarks[eye_landmarks[5]].y]))
    right_pt = np.linalg.norm(np.array([landmarks[eye_landmarks[2]].x, landmarks[eye_landmarks[2]].y]) - 
                              np.array([landmarks[eye_landmarks[4]].x, landmarks[eye_landmarks[4]].y]))
    width = np.linalg.norm(np.array([landmarks[eye_landmarks[0]].x, landmarks[eye_landmarks[0]].y]) - 
                           np.array([landmarks[eye_landmarks[3]].x, landmarks[eye_landmarks[3]].y]))
    return (left_pt + right_pt) / (2.0 * width)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            # Compute EAR for both eyes
            left_eye_ratio = eye_aspect_ratio(LEFT_EYE_LANDMARKS, landmarks)
            right_eye_ratio = eye_aspect_ratio(RIGHT_EYE_LANDMARKS, landmarks)
            avg_eye_ratio = (left_eye_ratio + right_eye_ratio) / 2.0

            # Detect blinking
            if avg_eye_ratio < BLINK_THRESHOLD:
                if time.time() - last_blink_time > 0.1:  # Avoid multiple counts for one blink
                    blink_count += 1
                    last_blink_time = time.time()

            # Estimate gaze direction based on nose position
            nose_x = landmarks[1].x
            gaze_direction = "Center"
            if nose_x < 0.45:
                gaze_direction = "Right"
            elif nose_x > 0.55:
                gaze_direction = "Left"

            print(f"Gaze Direction: {gaze_direction}, Blinking Count: {blink_count}")

    cv2.imshow("Eye Movement Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
