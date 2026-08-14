# Serves the FastAPI inference API. Build the model first (see README) --
# this image expects a trained model under models/distilbert-sentiment/best.
FROM python:3.11-slim

WORKDIR /app

# System deps for torch/transformers wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch keeps the image small; swap for a CUDA base image if you
# need GPU inference in production.
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY configs/ configs/
COPY models/distilbert-sentiment/best/ models/distilbert-sentiment/best/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
