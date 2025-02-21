import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from logging_module import setup_logger  # Import the logging module
import gradio as gr
from gradio_webrtc import WebRTC
import cv2
import mediapipe as mp

# Initialize logger
logger = setup_logger()

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3)

# Function for hand movement detection using Mediapipe
def detection(image, conf_threshold=0.3):
    logger.info("Processing new frame for hand detection")

    # Convert image to RGB (Mediapipe uses RGB images)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Draw the hand landmarks and contours on the image
    if results.multi_hand_landmarks:
        logger.info(f"Detected {len(results.multi_hand_landmarks)} hand(s)")
        for landmarks in results.multi_hand_landmarks:
            for landmark in landmarks.landmark:
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    else:
        logger.info("No hands detected")

    return image

# WebRTC configuration for video streaming
rtc_configuration = {
    "iceServers": [
        {"urls": "stun:stun.l.google.com:19302"},
        {"urls": "turn:turn.example.org", "username": "user", "credential": "pass"}
    ]
}

# Custom CSS for styling the interface
css = """.my-group {max-width: 600px !important; max-height: 600px !important;}
         .my-column {display: flex !important; justify-content: center !important; align-items: center !important;}"""

# Gradio Blocks interface
with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
        <h1 style='text-align: center'>
        Hand Movement Detection (Powered by WebRTC ⚡️)
        </h1>
        """
    )
    with gr.Column(elem_classes=["my-column"]):
        with gr.Group(elem_classes=["my-group"]):
            # WebRTC component to stream video
            image = WebRTC(label="Stream", rtc_configuration=rtc_configuration)
            
            # Confidence threshold slider for object detection
            conf_threshold = gr.Slider(
                label="Confidence Threshold",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.30,
            )

        # Stream function for hand movement detection
        image.stream(
            fn=detection, inputs=[image, conf_threshold], outputs=[image], time_limit=10
        )

# Launch the demo
if __name__ == "__main__":
    logger.info("Launching the hand movement detection demo")
    demo.launch()
