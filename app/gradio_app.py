"""Gradio front-end for the image captioning + Grad-CAM demo."""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import gradio as gr  # noqa: E402
import cv2  # noqa: E402

from src.model import load_blip_model  # noqa: E402
from src.inference import generate_caption  # noqa: E402
from src.xai import (
    register_gradcam_hooks,
    generate_gradcam,
    overlay_heatmap,
)  # noqa: E402

processor, model, device = load_blip_model()
register_gradcam_hooks(model)


def process_image(image):
    temp_path = "app/uploads/temp_gradio.jpg"
    os.makedirs("app/uploads", exist_ok=True)
    image.save(temp_path)

    caption = generate_caption(temp_path, processor, model, device)
    raw_image, cam_resized, _ = generate_gradcam(temp_path, processor, model, device)
    overlay = overlay_heatmap(raw_image, cam_resized)
    overlay_rgb = (
        cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB) if overlay.shape[-1] == 3 else overlay
    )

    return caption, overlay_rgb


demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=[
        gr.Textbox(label="Generated Caption"),
        gr.Image(label="Grad-CAM Overlay"),
    ],
    title="Image Captioning with Explainability",
    description="Upload an image to generate a caption and see which regions the model focused on.",
)

if __name__ == "__main__":
    demo.launch()
