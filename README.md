# PixelForge — AI Image Upscaler (Local GPU Web App)

Batch AI upscaling with **Real-ESRGAN** on your local GPU (auto-detects CUDA / MPS / CPU), sequential queue, full metadata + before/after comparison.

## Get the app

| Way | How | Best for |
|---|---|---|
| **Run from source (uv)** | `cd ai-upscaler && ./run.sh` | Development, Linux/macOS/WSL |
| **Portable Windows app (one click)** | Download `PixelForge.exe` from [Releases](https://github.com/rickviper/pixelForge/releases) — no Python, no terminal. See [Windows portable build](#windows-portable-build-one-click-exe). | Daily use on Windows |
| **Docker** | `docker compose up --build` → http://localhost:8000 (needs [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) for GPU; without it runs on CPU). | Clean isolated install |
| **Zip bundle** | `./package.sh [version]` → produces `pixelforge-upscaler-vX.zip`. Unzip anywhere, then `./run.sh`. Weights auto-download on first upscale. | Sharing / moving to another PC |

> **No `pip install` / `uvx` CLI**: This project is a web app with a desktop launcher, not a library. There is no `console_scripts` entry point — run it via `./run.sh` (source), `docker compose up`, or the portable `.exe`.

## Features

- **Models** (4 built-in, auto-downloaded on first use):
  - Real-ESRGAN x4plus — general photos, best all-round
  - Real-ESRNet x4plus — faithful, less hallucination
  - Real-ESRGAN x4plus Anime — anime, cartoons, line art
  - Real-ESRGAN x2plus — native 2×, cleaner for small boosts
- **Scale**: 2× / 3× / 4× (resampled exactly even if native netscale differs)
- **VRAM control**: tile 0 / 128 / 256 / 400 / 512 (clamped 0–1024), FP16 toggle — use 128–256 on 4 GB cards
- **Extras**: GFPGAN face enhance, denoise slider (0–1), sharpen slider (0–1), output PNG / JPEG / WEBP + quality, keep EXIF, transparency preserved (PNG/WEBP), DPI kept (PNG/JPEG)
- **Batch**: multi-upload drag & drop, processed **one-by-one** with per-file status (single GPU worker)
- **Details per image**: original vs upscaled size, resolution, MP, aspect, mode, format, DPI, file sizes, model, engine/device, inference time
- **Compare slider** before/after, individual download + **Download-all ZIP**
- **Fallback**: if AI weights/torch fail, auto-falls back to classical Lanczos + denoise so the app never breaks
- **PWA**: installable from browser (manifest + service worker), works offline for UI

## Run from source

```bash
cd ai-upscaler
chmod +x run.sh
./run.sh
# open http://localhost:8000
```

`run.sh` will:
1. Install `uv` if missing
2. Sync Python dependencies (`uv sync`)
3. Install CUDA PyTorch if `nvidia-smi` is present (otherwise CPU torch)
4. Start Uvicorn on `http://localhost:8000` with `--reload`

### Desktop launcher (native window, same server)

```bash
uv sync --extra desktop
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv run python desktop.py
```

Options (also work on the built `PixelForge.exe`):

```bash
python desktop.py --selftest              # check server + device, then exit
python desktop.py --selftest-upscale      # also run one CUDA upscale
python desktop.py --no-window             # headless server only
python desktop.py --no-browser            # suppress browser fallback
python desktop.py --port 8000             # fixed port
python desktop.py --data-dir D:\PFData    # custom writable data folder
```

## Windows portable build (one-click `.exe`)

The packaged app is a normal folder containing `PixelForge.exe`, the Python
runtime, CUDA PyTorch, the frontend (with vendored offline assets), and the
Real-ESRGAN x4plus weights. Double-clicking the exe starts the local server and
opens a native WebView2 window — no terminal, no second process to start.

Downloads in the native window use the standard Windows **Save As** dialog
(pywebview disables downloads by default; the launcher enables them at
startup). If a download button ever does nothing, check that
`logs/pixelforge.log` contains `Native downloads enabled: True`.

### Build it

From Windows (PowerShell 7 / Windows PowerShell 5.1):

```powershell
# One-time: install uv if needed -> https://astral.sh/uv
powershell -NoProfile -ExecutionPolicy Bypass -File .\build_windows.ps1
```

From WSL/Linux, the helper copies the project to a real Windows folder, builds
there, and copies the portable folder back:

```bash
./build_windows.sh
```

Build output:

```
dist/PixelForge-Windows/
  PixelForge.exe          <- double-click this
  _internal/              <- Python + torch/CUDA + frontend + weights
  RUNNING.txt
```

Notes:

- The first build downloads several GB of CUDA wheels and takes a while. The
  finished folder is about 3.3 GB (unavoidable with a bundled CUDA torch).
- Use `TORCH_INDEX=https://download.pytorch.org/whl/cu130` (default) or another
  CUDA wheel index matching your NVIDIA driver. For a CPU-only build, install
  CPU torch first and run `.\build_windows.ps1 -SkipInstall` (not recommended on
  the RTX 2050).
- The build script runs a real GPU self-test (`--selftest-upscale`) and writes
  `dist/selftest-data/selftest.json`.
- WebView2 is preinstalled on Windows 11 and current Windows 10. If it is
  missing, the exe automatically opens the default browser instead.
- The build is unsigned, so SmartScreen may warn on first launch: *More info →
  Run anyway*.

### Where the app stores data

The portable build keeps everything next to the exe in `PixelForge-Data\`:
`outputs/`, `weights/`, `logs/pixelforge.log`, `webview/`, and (for other
models) `gfpgan/` / `facexlib/` weight caches. If that folder is read-only, it
falls back to `%LOCALAPPDATA%\PixelForge`.

Only the x4plus photo weight is bundled. The other three Real-ESRGAN models and
the optional GFPGAN face model download automatically on first use (internet
required once); afterwards the app is fully offline. The first face-enhance job
additionally fetches ~520 MB of GFPGAN/facexlib weights (cached in
`PixelForge-Data\gfpgan\weights\`), so allow a few minutes for that one job.

## Docker

```bash
docker compose up --build
# http://localhost:8000
```

GPU support requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) on the host. Without it, the container runs on CPU.

Volumes (persisted on host):
- `./weights` — Real-ESRGAN models (~65 MB each)
- `./gfpgan` — GFPGAN/facexlib cache (~520 MB on first face enhance)
- `./outputs` — upscaled results
- `./uploads` — temporary uploads

## API

- `GET /api/system` → device (cuda/mps/cpu, GPU name, VRAM, torch), model list
- `POST /api/upscale` (multipart: `file` + model/scale/tile/use_fp16/face_enhance/output_format/jpeg_quality/denoise/sharpen/keep_exif/force_classical) → JSON with original/upscaled metadata, `download_url` (percent-encoded) and `download_file` (raw on-disk name for `batch-zip`). EXIF is kept for PNG/JPEG/WEBP; the Orientation tag is removed after rotation is applied.
- `GET /api/download/{fname}`
- `POST /api/batch-zip` `{files:[...]}` → `{zip_url}`

Desktop builds (`PixelForge.exe`) additionally require the per-launch
`X-PixelForge-Token` header on mutating endpoints; the bundled UI sends it
automatically. The token and port are in `PixelForge-Data\instance.json`.
Dev/server mode (no launcher) needs no token.

## Notes

- Weights auto-download on first use into `weights/` (~65 MB each).
- Best quality option available locally = CUDA FP16 Real-ESRGAN; CPU is automatic fallback (10–30× slower).
- For 4 GB VRAM keep tile ≤256 and close other GPU apps. Tile 256 is usually faster than 128 and stays around
  1.5 GB VRAM; tile 512 can hit the 4 GB limit and get *slower*.
- Speed: expect roughly 10–15 s per **input megapixel** on a 4 GB laptop GPU (RTX 2050, FP16, tile 256).
  A "200 KB" JPEG can still be 12 MP — the image card shows the real pixel size. Choosing 2× instead of 4×
  does not reduce inference time (the model always runs at its native scale; only the final resize differs).
- Every job writes one line to `logs/pixelforge.log` with input/output size, model, tile, engine and
  seconds, e.g. `upscale 1600x1200 -> 6400x4800 | model=realesrgan-x4plus scale=4 tile=256 ... 26.4s`.
- Maximum input: 80 megapixels (≈ 8000×10000) — larger images are rejected with a clear error.
- File upload limit: 50 MB.