import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
import torch
from collections import defaultdict

def initialize_mediapipe():
    """Initialize MediaPipe Pose."""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    return pose, mp_pose

def calculate_metrics(frame, pose, mp_pose, prev_state):
    """Process a single frame and return posture metrics."""
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)
    
    if not pose_results.pose_landmarks:
        return None  # No detection
    
    # Extract keypoints
    landmarks = pose_results.pose_landmarks.landmark
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    mid_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    
    # Shoulder Slouching
    shoulder_slouching = abs((left_shoulder.y + right_shoulder.y) / 2 - mid_hip.y)
    
    # Head Tilt Angle
    head_tilt_angle = np.arctan2(right_ear.y - left_ear.y, right_ear.x - left_ear.x) * (180 / np.pi)
    
    # Spine Curvature
    mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    spine_curvature = abs(mid_shoulder_y - ((nose.y + mid_hip.y) / 2))
    
    # Body Movement Index
    body_movement_index = abs(mid_shoulder_y - prev_state["prev_shoulder_y"]) if prev_state["prev_shoulder_y"] else 0
    prev_state["prev_shoulder_y"] = mid_shoulder_y
    
    # Hand Movement Frequency
    hand_movement_count = prev_state["hand_movement_count"]
    for side, wrist in zip(["left", "right"], [left_wrist, right_wrist]):
        if prev_state["prev_hand_y"][side] is not None and abs(prev_state["prev_hand_y"][side] - wrist.y) > 0.02:
            hand_movement_count += 1
        prev_state["prev_hand_y"][side] = wrist.y
    prev_state["hand_movement_count"] = hand_movement_count
    
    return {
        "shoulder_slouching": shoulder_slouching,
        "head_tilt_angle": head_tilt_angle,
        "spine_curvature": spine_curvature,
        "body_movement_index": body_movement_index,
        "hand_movement_frequency": hand_movement_count
    }

def analyze_emotion(frame):
    """Analyze emotions from a given frame using DeepFace."""
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
        return analysis[0]['emotion']  # Return emotion probabilities
    except Exception as e:
        print(f"Error in emotion analysis: {e}")
        return None

# Initialize
pose, mp_pose = initialize_mediapipe()
prev_state = {"prev_shoulder_y": None, "prev_hand_y": {"left": None, "right": None}, "hand_movement_count": 0}

# Example usage:
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    metrics = calculate_metrics(frame, pose, mp_pose, prev_state)
    emotions = analyze_emotion(frame)
    print(metrics, emotions)
cap.release()
cv2.destroyAllWindows()
