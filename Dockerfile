# PixelForge AI Upscaler — GPU webapp
# Build:  docker build -t pixelforge .
# Run:    docker compose up   (needs NVIDIA Container Toolkit for GPU)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY backend/ ./backend/
COPY frontend/ ./frontend/
RUN mkdir -p weights uploads outputs

# Torch CUDA build first (large), then the rest of the stack.
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
 && python3 -m pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart "pillow>=10" "numpy<2" \
      opencv-python basicsr facexlib gfpgan realesrgan httpx

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
