import gradio as gr
import main
import os

current_dir = os.getcwd()
font_path = os.path.join(current_dir, "/content/OCR-translator/NotoSerifGeorgian-VariableFont_wdth,wght.ttf")

def process_image(image_path):
    result = main.run_pipeline(image_path, font_path)
    return result

interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Image(type="numpy"),
    title="RU-KA Image Translation",
    description="Upload an image and see the processed result."
)

interface.launch(share=True)