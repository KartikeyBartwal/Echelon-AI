import gradio as gr
from dashboard import dashboard
from training_area import training_area  # Import the Blocks UI directly
from performance_analytics import performance_analytics

# Create the UI layout
with gr.Blocks() as app:
    with gr.Row():
        gr.Markdown("""
        # 📊 Echelon AI
        **Navigate through different sections using the tabs below.**
        """)

    with gr.Tabs():
        with gr.Tab("Dashboard"):
            gr.Markdown(dashboard)
        with gr.Tab("Training Area"):
            training_area.render()  # Corrected: Render the training area properly
        with gr.Tab("Performance Analytics"):
            gr.Markdown(performance_analytics)

# Launch the app
app.launch()
