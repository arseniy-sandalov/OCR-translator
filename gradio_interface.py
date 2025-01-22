import gradio as gr
import main

def process_image(image_path):
    result = main.run_pipeline(image_path)
    return result

interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Image(type="filepath"),
    title="RU-KA Image Translation",
    description="Upload an image and see the processed result."
)

interface.launch(share=True)