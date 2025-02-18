import gradio as gr
import cv2
import time

def video_stream():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', frame)
        yield buffer.tobytes()
    cap.release()

with gr.Blocks() as ui:
    video_output = gr.Image(type="pil", height=400, width=400)
    
    def update_video():
        for frame in video_stream():
            video_output.update(value=frame)
            time.sleep(0.1)

ui.launch()
