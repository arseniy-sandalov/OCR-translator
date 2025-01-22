import gradio as gr
import main

interface = gr.Interface(
    fn=main.run_pipeline,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Image(type="filepath"),
    title="RU-KA Image Translation",
    description="Upload an image and see the processed result."
)

interface.launch()