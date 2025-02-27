import cv2
import mediapipe as mp
import numpy as np
import asyncio
from deepface import DeepFace
from collections import defaultdict, deque
import time

def initialize_mediapipe():
    """Initialize MediaPipe Pose."""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    return pose, mp_pose

async def calculate_metrics_async(frame, pose, mp_pose, prev_state):
    """Asynchronous posture analysis."""
    return await asyncio.to_thread(calculate_metrics, frame, pose, mp_pose, prev_state)

def calculate_metrics(frame, pose, mp_pose, prev_state):
    """Process a single frame and return posture metrics."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)
    
    if not pose_results.pose_landmarks:
        return None  
    
    landmarks = pose_results.pose_landmarks.landmark
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    mid_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    
    shoulder_slouching = abs((left_shoulder.y + right_shoulder.y) / 2 - mid_hip.y)
    head_tilt_angle = np.arctan2(right_ear.y - left_ear.y, right_ear.x - left_ear.x) * (180 / np.pi)
    mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    spine_curvature = abs(mid_shoulder_y - ((nose.y + mid_hip.y) / 2))
    
    body_movement_index = abs(mid_shoulder_y - prev_state["prev_shoulder_y"]) if prev_state["prev_shoulder_y"] else 0
    prev_state["prev_shoulder_y"] = mid_shoulder_y
    
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

async def analyze_emotion_async(frame):
    """Asynchronous emotion analysis."""
    return await asyncio.to_thread(analyze_emotion, frame)

def analyze_emotion(frame):
    """Analyze emotions from a given frame using DeepFace."""
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
        return analysis[0]['emotion']  
    except Exception as e:
        print(f"Error in emotion analysis: {e}")
        return None

async def capture_frame(cap):
    """Asynchronously capture a frame from OpenCV."""
    return await asyncio.to_thread(cap.read)

async def compute_average_metrics(metric_buffer, prev_state):
    """Compute and print the average of metrics every 5 seconds."""
    while True:
        await asyncio.sleep(5)  # Wait for 5 seconds

        if metric_buffer:
            avg_metrics = {key: sum(d[key] for d in metric_buffer) / len(metric_buffer) for key in metric_buffer[0] if key != "emotions"}
            avg_emotions = {key: sum(d["emotions"].get(key, 0) for d in metric_buffer) / len(metric_buffer) for key in metric_buffer[0]["emotions"]}

            print(f"\n[5-Second Average Metrics] {avg_metrics}")
            print(f"[5-Second Average Emotions] {avg_emotions}\n")

            metric_buffer.clear()  # Reset buffer
            prev_state["hand_movement_count"] = 0  # Reset hand movement count


async def process_video():
    """Main function to process video asynchronously."""
    pose, mp_pose = initialize_mediapipe()
    prev_state = {"prev_shoulder_y": None, "prev_hand_y": {"left": None, "right": None}, "hand_movement_count": 0}
    
    cap = cv2.VideoCapture(0)
    metric_buffer = deque()  # Buffer to store frame-wise metrics

    # Start async task for computing averages every 5 seconds
    asyncio.create_task(compute_average_metrics(metric_buffer, prev_state))

    while cap.isOpened():
        ret, frame = await capture_frame(cap)
        if not ret:
            break

        # Run posture and emotion analysis in parallel
        posture_task = calculate_metrics_async(frame, pose, mp_pose, prev_state)
        emotion_task = analyze_emotion_async(frame)
        
        metrics, emotions = await asyncio.gather(posture_task, emotion_task)
        
        if metrics:
            metric_buffer.append(metrics)  # Store metrics for averaging
        
        # print(metrics, emotions)

    cap.release()
    cv2.destroyAllWindows()

# Run async function
asyncio.run(process_video())
