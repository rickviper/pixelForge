# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the PixelForge portable Windows build.

Build (on Windows, from this folder):

    uv venv --python 3.12 .venv-win
    uv pip install --python .venv-win\\Scripts\\python.exe torch torchvision ^
        --index-url https://download.pytorch.org/whl/cu130
    uv pip install --python .venv-win\\Scripts\\python.exe -r requirements-desktop.txt
    .venv-win\\Scripts\\python.exe -m PyInstaller --noconfirm --clean pixelforge.spec

Output: dist\\PixelForge\\PixelForge.exe (portable folder, no installer).
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata

# basicsr 1.4.2 still imports torchvision.transforms.functional_tensor, which
# was removed in torchvision >= 0.19. The app installs a runtime shim in
# backend/upscaler_engine.py; install the same shim at build time so
# collect_all() can import basicsr/realesrgan and discover their submodules.
import sys
import types

try:
    import torchvision.transforms.functional as _tv_functional

    if "torchvision.transforms.functional_tensor" not in sys.modules:
        _compat = types.ModuleType("torchvision.transforms.functional_tensor")
        _compat.rgb_to_grayscale = _tv_functional.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = _compat
except Exception:
    pass

ROOT = Path(SPECPATH).resolve()

datas = []
binaries = []
hiddenimports = []

# These packages load architectures dynamically and ship small data files.
for package in ("basicsr", "realesrgan", "gfpgan", "facexlib"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden
    try:
        datas += copy_metadata(package)
    except Exception:
        pass

# torchvision >= 0.19 loads its compiled extension by path
# (torch.ops.load_library("torchvision/_C_stable.pyd")), so PyInstaller's
# import graph never sees it. Without these files every custom op is missing
# and basicsr/realesrgan fail with "operator torchvision::nms does not exist".
try:
    import torchvision as _torchvision

    _tv_dir = Path(_torchvision.__file__).resolve().parent
    for _item in _tv_dir.iterdir():
        if _item.is_file() and _item.suffix.lower() not in {".py", ".pyc", ".pyo"}:
            binaries.append((str(_item), "torchvision"))
except Exception:
    pass

# Native window implementation + its .NET bridge.
try:
    webview_datas, webview_binaries, webview_hidden = collect_all("webview")
    datas += webview_datas
    binaries += webview_binaries
    hiddenimports += webview_hidden
    datas += copy_metadata("pywebview")
except Exception:
    pass

hiddenimports += [
    # Uvicorn selects these at runtime.
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    # FastAPI form parsing.
    "multipart",
    "python_multipart",
    # pywebview Windows backends.
    "webview.platforms.edgechromium",
    "webview.platforms.winforms",
    "webview.platforms.mshtml",
    "clr",
    "pythonnet",
    "proxy_tools",
    "bottle",
    # Torch / torchvision pieces referenced only indirectly.
    "torchvision.ops",
    "torchvision.transforms",
    "torchvision.transforms.functional",
    "torchvision.transforms.functional_tensor",
]

# The app's own assets live next to the executable in the portable folder.
# They are copied into _internal and found through sys._MEIPASS at runtime.
datas += [
    (str(ROOT / "frontend"), "frontend"),
    (str(ROOT / "weights"), "weights"),
    (str(ROOT / "assets"), "assets"),
]

# pywebview's .NET bridge (pythonnet) ships its own PyInstaller hook.
hook_dirs = []
try:
    import pythonnet

    hook_dirs.append(str(Path(pythonnet.__file__).resolve().parent / "_pyinstaller"))
except Exception:
    pass

a = Analysis(
    [str(ROOT / "desktop.py")],
    pathex=[str(ROOT), str(ROOT / "backend")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=hook_dirs,
    hooksconfig={},
    runtime_hooks=[str(ROOT / "assets" / "runtime_hook.py")],
    excludes=[
        "tkinter",
        "matplotlib",
        "pandas",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
        "sphinx",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "gi",
        "cefpython3",
        "torch.utils.tensorboard",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PixelForge",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / "pixelforge.ico"),
    version=str(ROOT / "assets" / "version_info.txt"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="PixelForge",
)
