FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements-docker.txt .
RUN pip install --no-cache-dir --default-timeout=300 torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --default-timeout=300 --retries 5 -r requirements-docker.txt
COPY src/ ./src/
COPY app/ ./app/
COPY data/captions.txt ./data/captions.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]