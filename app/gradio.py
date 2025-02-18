import gradio as gr
import time
import threading
import random

def analyze_persuasion(audio):
    """
    Simulates AI-driven persuasion analysis. Returns a score and feedback.
    """
    time.sleep(2)  # Simulating processing delay
    score = random.randint(60, 95)  # Simulated persuasion score
    feedback = [
        "Strong logical reasoning detected.",
        "Avoid using personal attacks (Ad Hominem).",
        "Try balancing emotional appeal with factual arguments.",
        "Great structure, but watch out for False Dichotomy fallacies."
    ]
    return score, random.choice(feedback)

def process_audio(audio, history):
    """
    Processes audio in a separate thread every 10 seconds.
    """
    global scores, feedbacks
    score, feedback = analyze_persuasion(audio)
    scores.append(score)
    feedbacks.append(feedback)
    return f"Persuasion Score: {score}%", feedback

def start_analysis(audio):
    """
    Starts a new thread for real-time persuasion analysis every 10 seconds.
    """
    global scores, feedbacks
    scores, feedbacks = [], []
    thread = threading.Thread(target=process_audio, args=(audio, None))
    thread.start()
    return "Processing audio... Please wait."

def get_live_feedback():
    """
    Returns the latest persuasion score and feedback dynamically.
    """
    if scores and feedbacks:
        return f"Live Persuasion Score: {scores[-1]}%", feedbacks[-1]
    return "Waiting for analysis...", "No feedback available yet."

# Corporate UI Styling
gr.Markdown("""
# Echelon AI - Persuasion Coach

Speak naturally, and get real-time feedback on your persuasive effectiveness.

- 🟢 **Live Persuasion Score** updates every 10 seconds
- 🟡 **AI-Driven Feedback** identifies strengths & areas to improve
- 🔵 **Corporate Aesthetic & Professional Design**
""")

# Gradio Interface
with gr.Blocks(css=".gradio-container { background-color: #f4f4f9; padding: 20px; font-family: 'Arial', sans-serif; }") as ui:
    with gr.Row():
        audio_input = gr.Audio(label="Speak to Analyze", type="filepath")
        score_output = gr.Textbox(label="Live Persuasion Score", interactive=False)
    
    feedback_output = gr.Textbox(label="AI Feedback", interactive=False)
    start_button = gr.Button("Start Analysis", variant="primary")
    
    start_button.click(start_analysis, inputs=[audio_input], outputs=[score_output])
    gr.Button("Get Live Feedback").click(get_live_feedback, outputs=[score_output, feedback_output])
    
ui.launch()
