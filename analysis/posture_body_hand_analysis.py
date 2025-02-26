import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize MediaPipe Pose and Face Detection
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
pose = mp_pose.Pose()
face_mesh = mp_face.FaceMesh()

# Open Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Metrics storage for averaging
metrics = {
    "shoulder_slouching": [],
    "head_tilt_angle": [],
    "spine_curvature": [],
    "body_movement_index": [],
    "hand_movement_frequency": []
}

start_time = time.time()
log_interval = 5  # seconds
prev_shoulder_y = None
prev_hand_y = {"left": None, "right": None}
hand_movement_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    pose_results = pose.process(rgb_frame)
    
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        
        # Extract keypoints
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        mid_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]

        # Shoulder Slouching (Higher difference means more slouching)
        shoulder_slouching = abs((left_shoulder.y + right_shoulder.y) / 2 - mid_hip.y)
        
        # Head Tilt Angle (Angle between nose and ears)
        head_tilt_angle = np.arctan2(right_ear.y - left_ear.y, right_ear.x - left_ear.x) * (180 / np.pi)
        
        # Spine Curvature (Deviation of mid-shoulder from nose-hip line)
        mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        spine_curvature = abs(mid_shoulder_y - ((nose.y + mid_hip.y) / 2))

        # Body Movement Index (Shoulder movement across frames)
        if prev_shoulder_y is not None:
            body_movement_index = abs(mid_shoulder_y - prev_shoulder_y)
        else:
            body_movement_index = 0
        prev_shoulder_y = mid_shoulder_y
        
        # Hand Movement Frequency (Wrist movement fluctuations)
        for side, wrist in zip(["left", "right"], [left_wrist, right_wrist]):
            if prev_hand_y[side] is not None and abs(prev_hand_y[side] - wrist.y) > 0.02:
                hand_movement_count += 1
            prev_hand_y[side] = wrist.y

        # Store metrics
        metrics["shoulder_slouching"].append(shoulder_slouching)
        metrics["head_tilt_angle"].append(head_tilt_angle)
        metrics["spine_curvature"].append(spine_curvature)
        metrics["body_movement_index"].append(body_movement_index)
        metrics["hand_movement_frequency"].append(hand_movement_count)

    # Display camera feed
    cv2.imshow("Echelon AI - Non-Verbal Analysis", frame)

    # Check if 5 seconds have passed
    if time.time() - start_time >= log_interval:
        avg_metrics = {key: np.mean(values) if values else 0 for key, values in metrics.items()}
        print(f"\n[LOG] Metrics (Averaged over 5 seconds):")
        for key, value in avg_metrics.items():
            print(f"{key.replace('_', ' ').title()}: {value:.4f}")
        
        # Reset metrics storage
        metrics = {key: [] for key in metrics}
        start_time = time.time()
        hand_movement_count = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
