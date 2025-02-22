import gradio as gr
from training_area import training_area
from dashboard import dashboard
from performance_analytics import performance_analytics

# ✅ **DO NOT create a new Blocks() instance inside training_area.py**
# Just call the function that returns the UI component.

with gr.Blocks() as app:
    with gr.Row():
        gr.Markdown("# 📊 Echelon AI\n**Navigate through different sections using the tabs below.**")

    with gr.Tabs():
        with gr.Tab("Dashboard"):
            gr.Markdown(dashboard)
        with gr.Tab("Training Area"):
            training_area()  
        with gr.Tab("Performance Analytics"):
            gr.Markdown(performance_analytics)

app.launch()