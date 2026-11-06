#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "=== AI Image Upscaler ==="
if ! command -v uv &>/dev/null; then echo "Installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; fi
echo "[1/3] Syncing dependencies (torch CUDA build)..."
uv sync 2>&1 | tail -5
echo "[2/3] Ensuring CUDA torch (if NVIDIA GPU present)..."
if nvidia-smi -L &>/dev/null; then
  uv pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3 || true
fi
echo "[3/3] Starting server on http://localhost:8000 ..."
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
