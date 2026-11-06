"""Upscaler engine: device detection, Real-ESRGAN loading, fallback classical upscale."""
from __future__ import annotations
import os
import shutil
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

# Compatibility: basicsr 1.4.2 imports torchvision.transforms.functional_tensor,
# removed in torchvision>=0.19. Inject a shim so AI imports work on new torch.
try:
    import sys as _sys_compat
    if "torchvision.transforms.functional_tensor" not in _sys_compat.modules:
        try:
            import torchvision.transforms.functional_tensor  # noqa: F401
        except Exception:
            try:
                from types import ModuleType as _ModuleType
                import torchvision.transforms.functional as _tvf
                _shim = _ModuleType("torchvision.transforms.functional_tensor")
                _shim.rgb_to_grayscale = _tvf.rgb_to_grayscale
                _sys_compat.modules["torchvision.transforms.functional_tensor"] = _shim
            except Exception:
                pass
except Exception:
    pass

WEIGHTS_DIR = Path(
    os.environ.get("PIXELFORGE_WEIGHTS_DIR")
    or (Path(__file__).resolve().parent.parent / "weights")
)

MODEL_REGISTRY = {
    "realesrgan-x4plus": {
        "label": "Real-ESRGAN x4plus (General Photo)",
        "file": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "netscale": 4,
        "model_class": "RRDBNet",
        "params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
        "default_tile": 256,
        "description": "Best all-round. People, landscapes, product shots.",
    },
    "realesrnet-x4plus": {
        "label": "Real-ESRNet x4plus (Faithful, less hallucination)",
        "file": "RealESRNet_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
        "netscale": 4,
        "model_class": "RRDBNet",
        "params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
        "default_tile": 256,
        "description": "More faithful to original, fewer AI details.",
    },
    "realesrgan-x4plus-anime": {
        "label": "Real-ESRGAN x4 Anime (Anime / Illustration)",
        "file": "RealESRGAN_x4plus_anime_6B.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_anime_6B.pth",
        "netscale": 4,
        "model_class": "SRVGGNetCompact",
        "params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu"),
        "default_tile": 256,
        "description": "Optimized for anime, cartoons, line art.",
    },
    "realesrgan-x2plus": {
        "label": "Real-ESRGAN x2plus (2x, cleaner)",
        "file": "RealESRGAN_x2plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "netscale": 2,
        "model_class": "RRDBNet",
        "params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
        "default_tile": 256,
        "description": "Native 2x model, sharper for small boosts.",
    },
}


def get_device_info() -> dict:
    info: dict = {"device": "cpu", "device_label": "CPU", "cuda_available": False,
                  "mps_available": False, "gpu_name": None, "vram_total_mb": None,
                  "torch_version": None, "cuda_version": None, "half_precision_ok": False}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if hasattr(torch.backends, "mps"):
            try:
                info["mps_available"] = torch.backends.mps.is_available()
            except Exception:
                pass
        if info["cuda_available"]:
            info["device"] = "cuda"
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                info["vram_total_mb"] = round(props.total_memory / 1024 / 1024)
            except Exception:
                pass
            info["cuda_version"] = getattr(torch.version, "cuda", None)
            info["device_label"] = f"CUDA GPU ({info.get('gpu_name') or 'GPU'})"
            info["half_precision_ok"] = True
        elif info["mps_available"]:
            info["device"] = "mps"
            info["device_label"] = "Apple MPS GPU"
        else:
            info["device"] = "cpu"
            info["device_label"] = "CPU"
    except Exception as e:
        info["error"] = str(e)
    return info


def ensure_weights(model_key: str) -> Path:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = MODEL_REGISTRY[model_key]
    dest = WEIGHTS_DIR / cfg["file"]
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    # download with progress
    tmp = dest.with_suffix(".downloading")
    print(f"[upscaler] downloading {cfg['file']} ...")
    try:
        # urlretrieve has no timeout, so a hung connection would block the only
        # upscale worker (and every queued job) forever. A socket timeout makes
        # the request fail, the endpoint fall back to classical and a later
        # attempt retry cleanly.
        with urllib.request.urlopen(cfg["url"], timeout=60) as response, open(
            tmp, "wb"
        ) as handle:
            shutil.copyfileobj(response, handle, 1024 * 256)
        if tmp.stat().st_size < 1_000_000:
            raise OSError(f"{cfg['file']} download looks truncated")
        tmp.replace(dest)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return dest


_model_cache: dict = {}


def _redirect_runtime_downloads() -> None:
    """Make GFPGAN/facexlib write their weights into our writable data dir.

    Both libraries normally download into their installed package directory,
    which may be read-only (and is inside the PyInstaller bundle when frozen).
    """
    root = str(WEIGHTS_DIR.parent)
    try:
        import gfpgan.utils as _gfpgan_utils
        _gfpgan_utils.ROOT_DIR = root
    except Exception:
        pass
    try:
        import facexlib.utils.misc as _facexlib_misc
        _facexlib_misc.ROOT_DIR = root
    except Exception:
        pass


def _build_torch_model(model_key: str, device: str, fp16: bool):
    """Build RealESRGANer instance (cached per model+device+fp16)."""
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.archs.srvgg_arch import SRVGGNetCompact
    from realesrgan import RealESRGANer

    cfg = MODEL_REGISTRY[model_key]
    weight_path = ensure_weights(model_key)

    if cfg["model_class"] == "RRDBNet":
        model = RRDBNet(**cfg["params"])
    else:
        model = SRVGGNetCompact(**cfg["params"])

    use_half = bool(fp16 and device == "cuda")
    upsampler = RealESRGANer(
        scale=cfg["netscale"],
        model_path=str(weight_path),
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=use_half,
        device=device,
    )
    return upsampler


def get_upsampler(model_key: str, device: str, fp16: bool):
    key = (model_key, device, fp16)
    if key not in _model_cache:
        _model_cache[key] = _build_torch_model(model_key, device, fp16)
    return _model_cache[key]


def upscale_ai(img_bgr: np.ndarray, model_key: str, tile: int, device: str, fp16: bool,
               face_enhance: bool = False):
    """Run Real-ESRGAN. Returns (output_bgr, elapsed_s). Raises on failure (caller falls back)."""
    import torch
    upsampler = get_upsampler(model_key, device, fp16)
    # realesrgan renamed this attribute to `tile_size` (older releases used
    # `tile`). Setting only the old name silently disables tiling, which makes
    # the model process the whole image at once -> VRAM exhaustion on 4-6 GB
    # cards and a very slow shared-memory fallback. Set whichever exists (and
    # both if both do, which is harmless).
    requested_tile = int(tile or 0)
    if hasattr(upsampler, "tile_size"):
        upsampler.tile_size = requested_tile
    if hasattr(upsampler, "tile"):
        upsampler.tile = requested_tile
    # fp16 switch requires rebuild if changed
    face_enhancer = None
    if face_enhance:
        from gfpgan import GFPGANer
        _redirect_runtime_downloads()
        # lazy GFPGAN, reuse bg upsampler. Constructing it may download the
        # GFPGAN/facexlib weights; keep that out of the reported inference time.
        gfpgan_key = ("gfpgan", device)
        if gfpgan_key not in _model_cache:
            _model_cache[gfpgan_key] = GFPGANer(
                model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                upscale=4, arch="clean", channel_multiplier=2,
                bg_upsampler=upsampler, device=device)
        face_enhancer = _model_cache[gfpgan_key]
        face_enhancer.bg_upsampler = upsampler
    t0 = time.time()
    with torch.inference_mode():
        if face_enhancer is not None:
            _, _, output = face_enhancer.enhance(img_bgr, has_aligned=False,
                                                 only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(img_bgr, outscale=None)
    return output, time.time() - t0


def upscale_classical(pil_img: Image.Image, scale: float, denoise: float, sharpen: float) -> Image.Image:
    """High-quality non-AI fallback: denoise + Lanczos + unsharp."""
    img = pil_img.convert("RGB")
    if denoise and denoise > 0:
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h = int(np.clip(denoise * 10, 1, 10))
        arr = cv2.fastNlMeansDenoisingColored(arr, None, h, h, 7, 21)
        img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    w, h = img.size
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    if sharpen and sharpen > 0:
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=int(sharpen * 150), threshold=2))
    return img


def post_process_bgr(img_bgr: np.ndarray, target_scale: float, netscale: int,
                     denoise: float, sharpen: float) -> np.ndarray:
    """Resize netscale output to exact target_scale + optional denoise/sharpen."""
    out = img_bgr
    if abs(target_scale - netscale) > 1e-6:
        # relative resize
        h, w = out.shape[:2]
        factor = target_scale / netscale
        out = cv2.resize(out, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_LANCZOS4)
    if denoise and denoise > 0:
        h = int(np.clip(denoise * 10, 1, 10))
        out = cv2.fastNlMeansDenoisingColored(out, None, h, h, 7, 21)
    if sharpen and sharpen > 0:
        # unsharp via PIL for stability
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil = pil.filter(ImageFilter.UnsharpMask(radius=2, percent=int(sharpen * 150), threshold=2))
        out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return out
