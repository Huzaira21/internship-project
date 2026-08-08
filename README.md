# Internship Project

## Overview
Internship tasks and deliverables repository. This project implements an end-to-end image captioning pipeline using the Flickr8k dataset, covering data exploration, preprocessing, baseline modeling, explainability (XAI), and MLOps deployment.

## Live Demo
Public Gradio demo link: https://9030c3a448f71733cc.gradio.live
(Note: this is a temporary share link, active for about a week from creation)

## Folder Structure
- `data/` - datasets
- `notebooks/` - Jupyter notebooks
- `src/` - source code (modular scripts)
- `app/` - FastAPI and Gradio applications
- `tests/` - unit tests
- `.github/workflows/` - CI/CD pipeline

## Setup Instructions
1. Clone repo
2. Create virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`

## Docker Setup
A Dockerfile is included to containerize the FastAPI application.

Build: `docker build -t captioning-api .`

Run: `docker run -p 8000:8000 captioning-api`

## Project Architecture
Flickr8k Dataset leads to Data Preprocessing (src/data_loader.py) which handles text cleaning, tokenization, image resizing, and normalization. This feeds into the BLIP Model (src/model.py), using the pretrained Salesforce/blip-image-captioning-base checkpoint, fine-tuned on the text decoder only with the vision encoder frozen. Inference (src/inference.py) handles caption generation with beam search and BLEU/ROUGE evaluation. Explainability (src/xai.py) provides Grad-CAM visualization and occlusion-based importance analysis. The Serving Layer includes a FastAPI REST API (app/main.py) and a Gradio interactive demo (app/gradio_app.py). The MLOps layer includes MLflow for experiment tracking and model registry, DVC for dataset and model versioning, GitHub Actions for CI (linting, testing, Docker build), and Docker for containerization.

## Running the Gradio Demo
Run: `python app/gradio_app.py`
Open the local URL shown in the terminal (e.g. http://127.0.0.1:7860)

## Running the FastAPI Service
Run: `uvicorn app.main:app --reload`
Visit http://127.0.0.1:8000/docs for interactive API documentation

## Running Tests
Run: `pytest tests/ -v`

## Known Limitations
The fine-tuning experiments were conducted on a very small subset (15 images) due to CPU-only local hardware, so performance gains from fine-tuning are limited and mainly serve to demonstrate the workflow rather than achieve state-of-the-art results. Grad-CAM and occlusion-based explainability methods are approximations; they highlight influential image regions but do not provide a complete causal explanation of model behavior. The public Gradio demo link is temporary rather than a permanently hosted deployment, due to free-tier restrictions on platforms like Hugging Face Spaces requiring a paid plan for Gradio/Docker Spaces. The model has not been tested extensively on out-of-domain images beyond a small manual sample.

## Future Work
Fine-tune on the full Flickr8k training set with more epochs for stronger performance gains. Deploy to a persistent hosting platform (e.g. Render, Streamlit Community Cloud) for a permanent public link. Expand explainability analysis with additional methods such as SHAP for the text decoder specifically. Add a drift monitoring report (e.g. Evidently AI) to track model performance over time on new data. Explore ONNX export for faster, more portable inference.

## Author
Huzaira Sultan - NUM-BSCS-2023-07