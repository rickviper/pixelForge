"""FastAPI backend for AI Image Upscaler."""
from __future__ import annotations
import hmac
import io
import json
import logging
import os
import re
import threading
import time
import uuid
import zipfile
from pathlib import Path
from urllib.parse import quote
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from upscaler_engine import (
    MODEL_REGISTRY, get_device_info, upscale_ai, upscale_classical, post_process_bgr,
)

BASE = Path(__file__).resolve().parent.parent
# The desktop launcher points these at a writable portable data folder so the
# app also works when the executable lives in a read-only location.
FRONTEND = Path(os.environ.get("PIXELFORGE_FRONTEND_DIR") or (BASE / "frontend")).resolve()
OUTPUTS = Path(os.environ.get("PIXELFORGE_OUTPUTS_DIR") or (BASE / "outputs")).resolve()
OUTPUTS.mkdir(parents=True, exist_ok=True)

DESKTOP = os.environ.get("PIXELFORGE_DESKTOP") == "1"
DESKTOP_TOKEN = os.environ.get("PIXELFORGE_TOKEN", "")
# Set by POST /api/shutdown; the desktop launcher watches this to stop cleanly.
SHUTDOWN_EVENT = threading.Event()

# Refuse absurd inputs instead of letting Pillow/torch run out of memory.
MAX_INPUT_PIXELS = 80_000_000

# One line per job: makes "why is this slow?" answerable from the log without
# the API response or the UI.
_job_log = logging.getLogger("pixelforge.jobs")

app = FastAPI(title="AI Image Upscaler", version="1.0.0")
# In the packaged app everything is same-origin. Only the dev web server needs
# permissive CORS for accidental cross-origin use.
if not DESKTOP:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

# Serve frontend static
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")
    if (FRONTEND / "vendor").exists():
        app.mount("/vendor", StaticFiles(directory=str(FRONTEND / "vendor")), name="vendor")


# Characters that are illegal in file names on Windows. Uploaded names can
# contain them (API clients, archives, non-Windows browsers), and saving would
# otherwise fail with a 500 on the user's machine.
_BAD_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _clean_stem(name: str) -> str:
    """Original filename without extension, safe on every platform."""
    stem = Path(name or "image").stem
    stem = _BAD_FILENAME_CHARS.sub("_", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    # Windows silently drops trailing dots/spaces and rejects them in some APIs.
    stem = stem.rstrip(". ")
    return (stem or "image")[:80]


_JOB_PREFIX = re.compile(r"^[0-9a-f]{12}_(.+)$")


def _public_name(disk_name: str) -> str:
    """On-disk files are '<12-hex-jobid>_<clean name>'; serve the clean name."""
    m = _JOB_PREFIX.match(disk_name)
    return m.group(1) if m else disk_name


def _image_info(pil: Image.Image, nbytes: int, fname: str, fmt: str | None = None) -> dict:
    w, h = pil.size
    dpi = None
    try:
        dpi_info = pil.info.get("dpi")
        if dpi_info:
            dpi = f"{round(dpi_info[0])}x{round(dpi_info[1])} DPI"
    except Exception:
        pass
    megapixels = round(w * h / 1e6, 2)
    return {
        "filename": fname,
        "width": w, "height": h,
        "megapixels": megapixels,
        "resolution": f"{w} x {h}",
        "aspect": f"{w/h:.2f}:1" if h else "?",
        "mode": pil.mode,
        "format": fmt or (pil.format or Path(fname).suffix.replace('.', '').upper() or "PNG"),
        "dpi": dpi or "—",
        "bytes": nbytes,
        "kb": round(nbytes / 1024, 1),
        "mb": round(nbytes / 1024 / 1024, 2),
    }


def _require_desktop_token(request: Request) -> None:
    """Reject cross-site POSTs in the packaged app.

    Desktop builds have no CORS middleware, so a random web page cannot read
    our responses -- but it *can* fire simple cross-origin multipart POSTs at
    localhost (drive-by GPU load / disk fill). Requiring the per-launch token
    on mutating endpoints closes that hole. Dev mode stays open.
    """
    if not DESKTOP:
        return
    token = request.headers.get("x-pixelforge-token", "")
    if not DESKTOP_TOKEN or not hmac.compare_digest(token, DESKTOP_TOKEN):
        raise HTTPException(403, "Forbidden")


def _merge_alpha(out_pil: Image.Image, alpha: Image.Image, output_format: str) -> Image.Image:
    """Carry the source transparency over to the upscaled image.

    The AI model only sees RGB, so the alpha channel is resampled separately.
    JPEG cannot store alpha: flatten onto white rather than leaving the
    transparent regions black (what convert('RGB') does).
    """
    if alpha.size != out_pil.size:
        alpha = alpha.resize(out_pil.size, Image.LANCZOS)
    if output_format in ("PNG", "WEBP"):
        out_pil = out_pil.convert("RGBA")
        out_pil.putalpha(alpha)
        return out_pil
    canvas = Image.new("RGB", out_pil.size, (255, 255, 255))
    canvas.paste(out_pil.convert("RGB"), (0, 0), alpha)
    return canvas


@app.get("/")
def root():
    idx = FRONTEND / "index.html"
    if idx.exists():
        # Never cache the HTML shell: otherwise browsers/PWA show the old
        # design until a hard reload. Static assets are cached by sw.js instead.
        headers = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
        if DESKTOP:
            # Inject desktop mode + the shutdown token into the shell. No
            # template engine needed, and the file on disk stays untouched.
            html = idx.read_text(encoding="utf-8")
            boot = json.dumps({
                "desktop": True,
                "version": app.version,
                "token": DESKTOP_TOKEN,
            })
            inject = (
                f"<script>window.__PIXELFORGE_BOOT__={boot};"
                f"window.__PIXELFORGE_DESKTOP__=true;"
                f"window.__PIXELFORGE_TOKEN__={json.dumps(DESKTOP_TOKEN)};</script>"
            )
            html = html.replace("</head>", inject + "\n</head>", 1)
            return HTMLResponse(html, headers=headers)
        return FileResponse(str(idx), headers=headers)
    return {"status": "ok", "hint": "frontend/index.html missing"}


@app.get("/favicon.ico")
def favicon():
    p = FRONTEND / "icons" / "icon-192.png"
    if p.exists():
        return FileResponse(str(p), media_type="image/png")
    raise HTTPException(404, "Not found")


@app.get("/api/ping")
def ping():
    """Cheap liveness probe used by the desktop launcher (does not import torch)."""
    return {
        "app": "pixelforge", "version": app.version, "desktop": DESKTOP,
        "pid": os.getpid(), "instance": DESKTOP_TOKEN,
    }


@app.post("/api/shutdown")
def shutdown(request: Request):
    """Stop the packaged desktop app (browser-fallback quit button)."""
    if not DESKTOP:
        raise HTTPException(404, "Not found")
    token = request.headers.get("x-pixelforge-token", "")
    if DESKTOP_TOKEN and not hmac.compare_digest(token, DESKTOP_TOKEN):
        raise HTTPException(403, "Forbidden")
    SHUTDOWN_EVENT.set()
    return {"ok": True}



# PWA: manifest, service worker (root scope) and icons served from frontend/
@app.get("/manifest.webmanifest")
def manifest():
    return FileResponse(
        str(FRONTEND / "manifest.webmanifest"),
        media_type="application/manifest+json",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/sw.js")
def service_worker():
    # sw.js MUST NOT be cached or the browser will never pick up a new
    # service worker (and keep serving the old cached index.html).
    return FileResponse(
        str(FRONTEND / "sw.js"),
        media_type="application/javascript",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/icons/{fname}")
def icon(fname: str):
    p = FRONTEND / "icons" / Path(fname).name
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(str(p))


@app.get("/api/system")
def system():
    dev = get_device_info()
    models = [
        {"key": k, "label": v["label"], "netscale": v["netscale"],
         "description": v["description"], "default_tile": v["default_tile"]}
        for k, v in MODEL_REGISTRY.items()
    ]
    return {"device": dev, "models": models, "desktop": DESKTOP, "version": app.version}


def _single_job(func):
    """Run one upscale at a time (GPU memory) without blocking the event loop.

    The endpoint is a plain ``def`` so Starlette runs it in its worker pool;
    slow model downloads and inference therefore never freeze /api/ping,
    /api/shutdown or the static UI.
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _UPSCALE_LOCK:
            return func(*args, **kwargs)

    return wrapper


_UPSCALE_LOCK = threading.Lock()


@app.post("/api/upscale")
@_single_job
def upscale(
    file: UploadFile = File(...),
    model: str = Form("realesrgan-x4plus"),
    scale: float = Form(4.0),
    tile: int = Form(256),
    use_fp16: bool = Form(True),
    face_enhance: bool = Form(False),
    output_format: str = Form("PNG"),
    jpeg_quality: int = Form(92),
    denoise: float = Form(0.0),
    sharpen: float = Form(0.0),
    keep_exif: bool = Form(False),
    force_classical: bool = Form(False),
    _token: None = Depends(_require_desktop_token),
):
    t_start = time.time()
    raw = file.file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    if len(raw) > 50 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 50MB)")

    try:
        orig_pil = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, "Not a valid image (use PNG/JPG/WEBP)")

    max_pixels = MAX_INPUT_PIXELS
    if orig_pil.width * orig_pil.height > max_pixels:
        raise HTTPException(
            400,
            f"Image too large ({orig_pil.width}x{orig_pil.height}). "
            f"Maximum is {max_pixels // 1_000_000} megapixels.",
        )
    try:
        orig_pil.load()
    except Exception:
        raise HTTPException(400, "Not a valid image (use PNG/JPG/WEBP)")

    source_format = orig_pil.format

    # Normalise orientation first: metadata (size, DPI) must describe the
    # pixels we actually process, and exif_transpose also drops the
    # Orientation tag so viewers will not rotate the result a second time.
    try:
        from PIL.ImageOps import exif_transpose

        orig_pil = exif_transpose(orig_pil)
    except Exception:
        pass

    orig_info = _image_info(
        orig_pil, len(raw), file.filename or "image", fmt=source_format
    )

    # Keep EXIF only when asked; the transposed copy's EXIF already has the
    # Orientation tag removed.
    exif_bytes = None
    if keep_exif and "exif" in orig_pil.info:
        exif_bytes = orig_pil.info["exif"]

    dpi = None
    try:
        dpi_value = orig_pil.info.get("dpi")
        if dpi_value and len(dpi_value) >= 2 and dpi_value[0] and dpi_value[1]:
            dpi = (float(dpi_value[0]), float(dpi_value[1]))
    except Exception:
        dpi = None

    # Preserve transparency (PNG/WEBP logos, stickers, cut-outs).
    alpha = None
    try:
        if orig_pil.mode in ("RGBA", "LA", "PA") or (
            orig_pil.mode == "P" and "transparency" in orig_pil.info
        ):
            alpha = orig_pil.convert("RGBA").getchannel("A")
    except Exception:
        alpha = None

    rgb_pil = orig_pil.convert("RGB")

    scale = float(scale)
    if scale not in (2, 3, 4):
        # allow any 1..4 but clamp
        scale = min(4.0, max(1.0, scale))
    output_format = (output_format or "PNG").upper()
    if output_format not in ("PNG", "JPEG", "JPG", "WEBP"):
        output_format = "PNG"
    if output_format == "JPG":
        output_format = "JPEG"

    # Clamp user-supplied numbers instead of trusting the form.
    tile = max(0, min(1024, int(tile)))
    jpeg_quality = max(1, min(100, int(jpeg_quality)))
    denoise = min(1.0, max(0.0, float(denoise)))
    sharpen = min(1.0, max(0.0, float(sharpen)))

    dev = get_device_info()
    device = dev.get("device", "cpu")
    use_fp16 = bool(use_fp16 and device == "cuda")

    if model not in MODEL_REGISTRY:
        model = "realesrgan-x4plus"
    netscale = MODEL_REGISTRY[model]["netscale"]

    method = ""
    elapsed = 0.0
    out_pil = None
    err_note = None

    img_bgr = cv2.cvtColor(np.array(rgb_pil), cv2.COLOR_RGB2BGR)

    if not force_classical:
        try:
            out_bgr, elapsed = upscale_ai(
                img_bgr, model_key=model, tile=tile,
                device=device, fp16=use_fp16, face_enhance=face_enhance)
            out_bgr = post_process_bgr(out_bgr, scale, netscale, denoise, sharpen)
            out_pil = Image.fromarray(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
            method = "realesrgan-gpu" if device == "cuda" else f"realesrgan-{device}"
            if face_enhance:
                method += "+gfpgan"
        except Exception as e:
            # Keep the full traceback in the desktop log; the API response only
            # needs the short note. Frozen-build failures are otherwise very
            # hard to diagnose from selftest.json alone.
            import traceback as _traceback

            _traceback.print_exc()
            err_note = f"AI engine failed ({type(e).__name__}: {str(e)[:200]}), used classical fallback."
            out_pil = None

    if out_pil is None:
        t0 = time.time()
        out_pil = upscale_classical(rgb_pil, scale, denoise, sharpen)
        elapsed = time.time() - t0
        method = "classical-lanczos"

    if alpha is not None:
        out_pil = _merge_alpha(out_pil, alpha, output_format)

    # save output: same name as original + " (upscaled)"; job prefix keeps it unique on disk
    job_id = uuid.uuid4().hex[:12]
    ext = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}[output_format]
    out_name = f"{_clean_stem(file.filename)} (upscaled).{ext}"
    disk_name = f"{job_id}_{out_name}"
    out_path = OUTPUTS / disk_name
    save_kwargs: dict = {}
    if output_format == "JPEG":
        save_kwargs = {"quality": int(jpeg_quality), "optimize": True}
    elif output_format == "WEBP":
        save_kwargs = {"quality": int(jpeg_quality), "method": 6}
    elif output_format == "PNG":
        save_kwargs = {"optimize": True}
    if exif_bytes:
        # PNG/JPEG/WEBP all support EXIF in Pillow; the Orientation tag was
        # already removed by exif_transpose().
        save_kwargs["exif"] = exif_bytes
    if dpi and output_format in ("PNG", "JPEG"):
        save_kwargs["dpi"] = dpi
    try:
        out_pil.save(out_path, output_format, **save_kwargs)
    except (OSError, TypeError, ValueError):
        # A malformed EXIF payload must not turn a finished upscale into an
        # error: retry without metadata, otherwise re-raise the real failure.
        if "exif" not in save_kwargs:
            raise
        save_kwargs.pop("exif")
        out_pil.save(out_path, output_format, **save_kwargs)

    out_bytes = out_path.stat().st_size
    # re-open to get true saved format info
    with Image.open(out_path) as chk:
        chk.load()
        new_info = _image_info(chk, out_bytes, out_name)

    total = round(time.time() - t_start, 2)
    ratio = round(new_info["width"] / orig_info["width"], 2) if orig_info["width"] else 0
    download_url = f"/api/download/{quote(disk_name, safe='')}"

    mp = orig_info["width"] * orig_info["height"] / 1e6
    _job_log.info(
        "upscale %dx%d -> %dx%d | model=%s scale=%s tile=%s fp16=%s face=%s "
        "| %s in %.2fs total=%.2fs (%.2f MP/s)",
        orig_info["width"], orig_info["height"],
        new_info["width"], new_info["height"],
        model, scale, tile, use_fp16, face_enhance,
        method, elapsed, total, mp / max(elapsed, 0.001),
    )

    return {
        "job_id": job_id,
        "download_url": download_url,
        # Raw on-disk name: used by /api/batch-zip (the URL is percent-encoded).
        "download_file": disk_name,
        "preview_url": download_url,
        "original": orig_info,
        "upscaled": new_info,
        "settings": {
            "model": model, "model_label": MODEL_REGISTRY[model]["label"],
            "requested_scale": scale, "achieved_scale": ratio,
            "tile": tile, "fp16": use_fp16, "face_enhance": face_enhance,
            "output_format": output_format, "jpeg_quality": jpeg_quality,
            "denoise": denoise, "sharpen": sharpen,
        },
        "engine": {"method": method, "device": device,
                   "device_label": dev.get("device_label"),
                   "gpu": dev.get("gpu_name"),
                   "inference_s": round(elapsed, 2), "total_s": total},
        "fallback_note": err_note,
    }


@app.get("/api/download/{fname}")
def download(fname: str):
    p = OUTPUTS / Path(fname).name
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(str(p), filename=_public_name(p.name))


@app.post("/api/batch-zip")
def batch_zip(payload: dict, _token: None = Depends(_require_desktop_token)):
    """Body: {files: [download_fname, ...]} -> returns zip."""
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise HTTPException(400, "No files")

    paths = []
    for entry in files[:50]:
        p = OUTPUTS / Path(str(entry)).name
        if p.is_file():
            paths.append(p)
    if not paths:
        raise HTTPException(400, "No matching files")

    zid = uuid.uuid4().hex[:8]
    zpath = OUTPUTS / f"batch_{zid}.zip"
    # Generated ZIPs are throwaway downloads; drop old ones so the outputs
    # folder cannot grow without bound when users never clean it up.
    cutoff = time.time() - 6 * 3600
    for old in OUTPUTS.glob("batch_*.zip"):
        try:
            if old.stat().st_mtime < cutoff:
                old.unlink()
        except OSError:
            pass
    used: set = set()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            arc = _public_name(p.name)
            # de-duplicate entry names when originals share a filename
            n = 1
            base, suffix = arc.rsplit(".", 1) if "." in arc else (arc, "")
            while arc in used:
                n += 1
                arc = f"{base} ({n}).{suffix}" if suffix else f"{base} ({n})"
            used.add(arc)
            z.write(p, arcname=arc)
    return {"zip_url": f"/api/download/{quote(zpath.name, safe='')}"}
