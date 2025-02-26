from deepface import DeepFace
import cv2
import torch
import os
import time
from collections import defaultdict
import sys 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logging_module import setup_logger
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disables GPU, forces CPU execution

# Force DeepFace to run on CPU
torch.backends.cudnn.enabled = False

# Initialize logger
logger = setup_logger(log_dir="logs", logger_name="emotion_analysis")

def calculate_average_emotions(emotion_list):
    if not emotion_list:
        return None
    
    # Initialize counters for each emotion
    emotion_sums = defaultdict(float)
    for analysis in emotion_list:
        for emotion, value in analysis['emotion'].items():
            emotion_sums[emotion] += value
    
    # Calculate averages
    num_analyses = len(emotion_list)
    return {emotion: value/num_analyses for emotion, value in emotion_sums.items()}

# Initialize variables for time-based logging
last_log_time = time.time()
emotion_buffer = []

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
        emotion_buffer.append(analysis[0])  # DeepFace.analyze returns a list
        
        # Check if 5 seconds have passed
        current_time = time.time()
        if current_time - last_log_time >= 5:
            # Calculate and log average emotions
            avg_emotions = calculate_average_emotions(emotion_buffer)
            if avg_emotions:
                logger.info(f"Average emotions over last 5 seconds: {avg_emotions}")
            
            # Reset buffer and update last log time
            emotion_buffer = []
            last_log_time = current_time
            
    except Exception as e:
        logger.error(f"Error in emotion analysis: {e}")

    cv2.imshow("Emotion Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
