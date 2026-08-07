# Internship Project

## Overview
Internship tasks and deliverables repository.

## Folder Structure
- `data/` - datasets
- `notebooks/` - Jupyter notebooks
- `src/` - source code
- `tests/` - unit tests

## Setup Instructions
1. Clone repo
2. Create virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
## Docker Setup

A Dockerfile is included to containerize the FastAPI application.

**Build:**
docker build -t captioning-api .
**Run:**
docker run -p 8000:8000 captioning-api
**Note:** Local testing of the Docker build was affected by a Docker 
Desktop environment issue on the development machine (stuck background 
processes preventing the application from starting). The Dockerfile 
follows standard practices (Python 3.11-slim base image, required 
system libraries for OpenCV, CPU-only torch installation to reduce 
image size) and builds successfully up to the dependency installation 
stage, confirmed via terminal build logs.

## Author
Huzaira Sultan - NUM-BSCS-2023-07
