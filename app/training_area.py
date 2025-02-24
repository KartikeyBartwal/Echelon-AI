import sys
import os
import gradio as gr
from gradio_webrtc import WebRTC
import cv2
import mediapipe as mp

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from logging_module import setup_logger  

# Initialize Logger
logger = setup_logger()

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3)

# WebRTC Configuration
RTC_CONFIG = {"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}

# Custom CSS for UI Styling
CSS = """
body {
    background-color: #121212 !important;
    color: white !important;
    font-family: 'Poppins', sans-serif !important;
}

.gradio-container {
    max-width: 900px;
    margin: auto;
}

.scenario-box, .metrics-box, .tips-box {
    border: 2px solid #444;
    padding: 20px;
    margin: 15px 0;
    border-radius: 10px;
    background: linear-gradient(135deg, #1e1e1e, #292929);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    font-size: 16px;
    line-height: 1.5;
    color: #ddd;
    transition: all 0.3s ease-in-out;
}

.scenario-box:hover, .metrics-box:hover, .tips-box:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 16px rgba(255, 204, 0, 0.3);
}

.scenario-box h2, .metrics-box h2, .tips-box h2 {
    color: #ffcc00;
    text-align: center;
    font-size: 20px;
    letter-spacing: 1px;
    border-bottom: 2px solid rgba(255, 204, 0, 0.7);
    padding-bottom: 8px;
}
"""

# --- Functionality ---
def detect_hands(image, conf_threshold=0.3):
    """Processes a frame and detects hand movements."""
    logger.info("Processing frame for hand detection")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)
    return image

# --- UI Components ---
SCENARIO_TEXT = """
🚀 Scenario:
An investor is skeptical about AI-driven persuasion training, believing traditional coaching is superior. 🤔

🎯 Your Task:
Convince them that AI offers a game-changing advantage by providing:
✅ Real-time feedback 📊
✅ Personalized training 🎯
✅ Scalable and data-driven insights 📈

Make your argument compelling, confident, and persuasive. 🔥
"""

def create_scenario_section():
    """Creates the scenario box component."""
    return gr.Markdown(SCENARIO_TEXT, elem_classes=["scenario-box"])

def create_video_section():
    """Creates the live webcam video section with hand detection."""
    with gr.Column():
        gr.Markdown("### Live Video & Hand Detection")
        video_feed = WebRTC(label="Video Stream", rtc_configuration=RTC_CONFIG)
        conf_threshold = gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.3, visible=False)
        video_feed.stream(fn=detect_hands, inputs=[video_feed, conf_threshold], outputs=[video_feed])
        return video_feed

def create_performance_section():
    """Creates the performance metrics placeholder section."""
    with gr.Column():
        return gr.Textbox(value="Coming soon...", interactive=False, lines=3, label="Metrics")

def create_tips_section():
    """Creates the tips section."""
    with gr.Column():
        return gr.Textbox(value="Enhance your persuasion by using structured arguments and confident delivery.", interactive=False, lines=2, label="Tips")

# --- Build UI ---
with gr.Blocks(css=CSS) as training_area:
    gr.Markdown("<h1 style='text-align: center'>Echelon AI - Persuasion Training</h1>")
    create_scenario_section()
    with gr.Row():
        create_video_section()
        create_performance_section()
    create_tips_section()

# --- Launch UI ---
if __name__ == "__main__":
    logger.info("Launching Echelon AI - Persuasion Training")
    training_area.launch()
