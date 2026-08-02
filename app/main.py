"""FastAPI service for image captioning with Grad-CAM explanation."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import shutil
import uuid
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from src.model import load_blip_model
from src.inference import generate_caption
from src.xai import register_gradcam_hooks, generate_gradcam, overlay_heatmap
import cv2

app = FastAPI(title="Image Captioning API")

UPLOAD_DIR = "app/uploads"
OUTPUT_DIR = "app/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

processor, model, device = load_blip_model()
register_gradcam_hooks(model)


@app.get("/")
def read_root():
    return {"message": "Image Captioning API is running."}


@app.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    """Accept an image, return a generated caption and a Grad-CAM overlay image path."""
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}.jpg")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    caption = generate_caption(input_path, processor, model, device)

    raw_image, cam_resized, _ = generate_gradcam(input_path, processor, model, device)
    overlay = overlay_heatmap(raw_image, cam_resized)

    output_filename = f"{file_id}_gradcam.jpg"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return JSONResponse({
        "caption": caption,
        "gradcam_overlay_path": output_path
    })