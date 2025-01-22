import gradio as gr
import cv2
import numpy as np
import main

interface = gr.Interface(
    fn=main,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Image(type="filepath"),
    title="RU-KA Image Translation",
    description="Upload an image and see the processed result."
)

interface.launch()