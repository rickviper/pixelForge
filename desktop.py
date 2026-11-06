#!/usr/bin/env python3
"""PixelForge desktop entry point.

Starts the bundled FastAPI backend on a free localhost port and opens it in a
native WebView2 window (pywebview). If a native window cannot be created, the
default browser is used instead.

The same file works for development (``python desktop.py``) and for the
PyInstaller build (``PixelForge.exe``).
"""
from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import secrets
import shutil
import socket
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Optional

APP_NAME = "PixelForge"
APP_VERSION = "1.0.0"
HOST = "127.0.0.1"

# Disabled for headless runs (--no-window/--selftest/--no-browser): a blocking
# MessageBox would otherwise hang the process with nobody to click OK.
DIALOGS_ENABLED = True

log = logging.getLogger("pixelforge")
_LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path:
    """Folder containing the executable (or this script in development)."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def resource_dir() -> Path:
    """Folder containing bundled data (frontend, weights, ...)."""
    # PyInstaller onedir puts data in <app>/_internal and exposes it as _MEIPASS.
    return Path(getattr(sys, "_MEIPASS", app_dir())).resolve()


def _is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".pixelforge-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:
        return False


def resolve_data_dir(override: Optional[str]) -> Path:
    """Pick a writable folder for outputs, weights, logs and caches.

    Portable builds keep everything next to the executable. If that folder is
    read-only (e.g. Program Files), fall back to %LOCALAPPDATA%\\PixelForge.
    """
    if override:
        path = Path(override).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    if not is_frozen():
        return app_dir()
    portable = app_dir() / "PixelForge-Data"
    if _is_writable(portable):
        return portable
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home())
    fallback = Path(base) / APP_NAME
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def seed_bundled_weights(resource: Path, data: Path) -> None:
    """Copy model weights shipped with the app into the writable folder."""
    src_dir = resource / "weights"
    dst_dir = data / "weights"
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not src_dir.is_dir():
        return
    for src in src_dir.glob("*.pth"):
        dst = dst_dir / src.name
        try:
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                continue
            log.info("Installing bundled weight %s", src.name)
            shutil.copy2(src, dst)
        except Exception:
            log.exception("Could not copy bundled weight %s", src)


# ---------------------------------------------------------------------------
# Startup / logging
# ---------------------------------------------------------------------------
def setup_logging(data_dir: Path) -> Path:
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pixelforge.log"

    # PyInstaller --windowed has no console: stdout/stderr are None. Redirect
    # them to the log so stray library prints never crash the process.
    stderr_was_none = sys.stderr is None
    if sys.stdout is None or sys.stderr is None:
        stream = open(log_file, "a", encoding="utf-8", buffering=1)
        if sys.stdout is None:
            sys.stdout = stream
        if sys.stderr is None:
            sys.stderr = stream

    handlers: list[logging.Handler] = []
    try:
        from logging.handlers import RotatingFileHandler

        handlers.append(
            RotatingFileHandler(
                log_file, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
        )
    except Exception:
        pass
    # Avoid logging every line twice when stderr was redirected to the same file.
    if sys.stderr is not None and not stderr_was_none:
        handlers.append(logging.StreamHandler(sys.stderr))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    # Uvicorn's own loggers propagate to our root handlers.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    return log_file


def show_error(message: str) -> None:
    log.error("%s", message)
    if DIALOGS_ENABLED and sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, message, APP_NAME, 0x10)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def http_json(url: str, timeout: float = 2.0) -> dict[str, Any]:
    # Bypass system/env proxies: a corporate proxy must never intercept or
    # delay requests to our own loopback server.
    with _LOCAL_OPENER.open(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_server(
    port: int,
    timeout: float = 180.0,
    *,
    server: Any = None,
    server_errors: Optional[list[BaseException]] = None,
    thread: Optional[threading.Thread] = None,
) -> dict[str, Any]:
    """Poll /api/ping until the local backend answers.

    The deadline is generous because the first import of torch/CUDA and the
    first model download can take minutes on slow machines. The requests
    themselves are short so a busy server can still answer between jobs.
    """
    url = f"http://{HOST}:{port}/api/ping"
    deadline = time.monotonic() + timeout
    last_error: Optional[BaseException] = None
    while time.monotonic() < deadline:
        if server_errors:
            raise RuntimeError(f"Backend server failed to start: {server_errors[0]!r}")
        if getattr(server, "should_exit", False):
            raise RuntimeError("Backend server was asked to stop during startup")
        if thread is not None and not thread.is_alive():
            raise RuntimeError("Backend server thread stopped before it was ready")
        try:
            payload = http_json(url, timeout=3.0)
            if payload.get("app") == "pixelforge":
                return payload
        except Exception as exc:  # connection refused while starting
            last_error = exc
        time.sleep(0.15)
    raise RuntimeError(f"PixelForge server did not start in time ({last_error})")


def read_instance(data_dir: Path) -> Optional[dict[str, Any]]:
    """Return the running instance if one is already serving this data folder."""
    path = data_dir / "instance.json"
    if not path.exists():
        return None
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
        if info.get("app") != APP_NAME:
            return None
        payload = http_json(f"http://{HOST}:{int(info['port'])}/api/ping", timeout=0.75)
        if payload.get("instance") == info.get("token"):
            return info
    except Exception:
        pass
    return None


def write_instance(data_dir: Path, port: int, token: str) -> Path:
    path = data_dir / "instance.json"
    payload = json.dumps(
        {
            "app": APP_NAME,
            "version": APP_VERSION,
            "pid": os.getpid(),
            "port": port,
            "token": token,
            "url": f"http://{HOST}:{port}/",
            "started": time.time(),
        },
        indent=2,
    )
    # Write-then-rename so a second launch never reads a half-written file.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    return path


def clear_instance(data_dir: Path, token: str) -> None:
    path = data_dir / "instance.json"
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
        if info.get("token") == token:
            path.unlink()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Window / browser
# ---------------------------------------------------------------------------
class DesktopApi:
    """Small JS <-> Python bridge used by the Quit button."""

    def __init__(
        self,
        data_dir: Path,
        token: str,
        port: int,
        shutdown_event: threading.Event,
    ) -> None:
        self._data_dir = data_dir
        self._token = token
        self._port = port
        self._shutdown_event = shutdown_event

    def quit(self) -> bool:
        log.info("Quit requested from the UI")
        # Ask the server to stop (this also works when this window belongs to
        # a second launch and the real server runs in another process).
        try:
            request = urllib.request.Request(
                f"http://{HOST}:{self._port}/api/shutdown",
                method="POST",
                headers={"X-PixelForge-Token": self._token},
            )
            _LOCAL_OPENER.open(request, timeout=2).read()
        except Exception:
            pass
        self._shutdown_event.set()
        return True

    def info(self) -> dict[str, Any]:
        return {
            "app": APP_NAME,
            "version": APP_VERSION,
            "data_dir": str(self._data_dir),
        }


def _watch_shutdown(webview_module: Any, shutdown_event: threading.Event) -> None:
    """Destroy open windows when a quit is requested from the UI."""
    while not shutdown_event.wait(0.25):
        pass
    # Window.destroy() marshals to the GUI thread; retry briefly in case the
    # window is still finishing a navigation.
    for _ in range(40):
        windows = list(getattr(webview_module, "windows", []))
        if not windows:
            return
        for window in windows:
            try:
                window.destroy()
            except Exception:
                pass
        time.sleep(0.1)


def open_native_window(
    url: str,
    data_dir: Path,
    token: str,
    port: int,
    shutdown_event: threading.Event,
) -> bool:
    """Open a WebView2 window. Returns False when unavailable."""
    try:
        import webview
    except Exception:
        log.exception("pywebview is not available; falling back to the browser")
        return False

    # pywebview cancels every download unless this is enabled (default is
    # False). Without it the Download buttons in the native window silently do
    # nothing: the click never reaches the browser and no file is saved.
    try:
        webview.settings["ALLOW_DOWNLOADS"] = True
        log.info("Native downloads enabled: %s", webview.settings.get("ALLOW_DOWNLOADS"))
    except Exception:
        log.exception("Could not enable WebView downloads")

    try:
        api = DesktopApi(data_dir, token, port, shutdown_event)
        webview.create_window(
            APP_NAME,
            url,
            width=1600,
            height=1000,
            min_size=(960, 640),
            maximized=True,
            background_color="#070b16",
            js_api=api,
        )
        threading.Thread(
            target=_watch_shutdown,
            args=(webview, shutdown_event),
            name="pixelforge-window-watch",
            daemon=True,
        ).start()
        icon = resource_dir() / "assets" / "pixelforge.ico"
        start_kwargs: dict[str, Any] = {
            "debug": False,
            "private_mode": False,
            "storage_path": str(data_dir / "webview"),
        }
        if icon.exists():
            start_kwargs["icon"] = str(icon)
        try:
            webview.start(**start_kwargs)
        except TypeError:
            # Older pywebview versions have a smaller start() signature.
            webview.start(debug=False)
        return True
    except Exception:
        log.exception("Could not create the native window; falling back to the browser")
        return False


def wait_forever(shutdown_event: Optional[threading.Event] = None) -> None:
    """Block until Ctrl+C, a UI shutdown request, or process termination."""
    try:
        while shutdown_event is None or not shutdown_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# Self-test (used by the build script to validate the frozen bundle)
# ---------------------------------------------------------------------------
def run_selftest(port: int, data_dir: Path, upscale: bool, token: str = "") -> int:
    import httpx

    base = f"http://{HOST}:{port}"
    result: dict[str, Any] = {"ok": False, "app": APP_NAME, "version": APP_VERSION}
    try:
        headers = {"X-PixelForge-Token": token} if token else {}
        with httpx.Client(
            base_url=base, timeout=600.0, headers=headers, trust_env=False
        ) as client:
            result["ping"] = client.get("/api/ping").json()
            system = client.get("/api/system").json()
            result["system"] = system
            result["frozen"] = is_frozen()
            result["executable"] = sys.executable
            result["data_dir"] = str(data_dir)

            if upscale:
                from io import BytesIO

                from PIL import Image

                buf = BytesIO()
                Image.new("RGB", (48, 48), (90, 130, 190)).save(buf, "PNG")
                files = {"file": ("selftest.png", buf.getvalue(), "image/png")}
                form = {
                    "model": "realesrgan-x4plus",
                    "scale": "4",
                    "tile": "128",
                    "use_fp16": "true",
                    "face_enhance": "false",
                    "output_format": "PNG",
                    "jpeg_quality": "92",
                    "denoise": "0",
                    "sharpen": "0",
                    "keep_exif": "false",
                    "force_classical": "false",
                }
                t0 = time.time()
                response = client.post("/api/upscale", files=files, data=form)
                response.raise_for_status()
                up = response.json()
                result["upscale"] = {
                    "engine": up.get("engine"),
                    "upscaled": up.get("upscaled"),
                    "fallback_note": up.get("fallback_note"),
                    "seconds": round(time.time() - t0, 2),
                }
                method = str(up.get("engine", {}).get("method", ""))
                result["used_cuda"] = up.get("engine", {}).get("device") == "cuda"
                result["used_ai"] = method.startswith("realesrgan")
                if not result["used_ai"]:
                    # A silent classical fallback means the bundled AI engine is
                    # broken (e.g. missing torchvision ops); fail the self-test.
                    result["error"] = f"AI engine was not used (method={method!r})"
            result["ok"] = not result.get("error")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()

    out = data_dir / "selftest.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"[selftest] written to {out}")
    return 0 if result.get("ok") else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PixelForge desktop launcher")
    parser.add_argument("--port", type=int, default=0, help="0 = pick a free port")
    parser.add_argument("--data-dir", default=None, help="Writable data folder")
    parser.add_argument("--no-window", action="store_true", help="Run the server only")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser")
    parser.add_argument("--selftest", action="store_true", help="Check server/device, then exit")
    parser.add_argument(
        "--selftest-upscale",
        action="store_true",
        help="Also run one AI upscale during the self-test (slow)",
    )
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {APP_VERSION}")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    global DIALOGS_ENABLED

    args = parse_args(argv)
    DIALOGS_ENABLED = not (
        args.no_window or args.no_browser or args.selftest or args.selftest_upscale
    )
    try:
        data_dir = resolve_data_dir(args.data_dir)
    except Exception as exc:
        show_error(
            "PixelForge could not create its data folder.\n\n"
            f"{args.data_dir or ''}\n\n{type(exc).__name__}: {exc}"
        )
        return 2
    log_file = setup_logging(data_dir)

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTHONUTF8", "1")

    resource = resource_dir()
    frontend = resource / "frontend"
    if not frontend.is_dir():
        frontend = app_dir() / "frontend"

    seed_bundled_weights(resource, data_dir)
    (data_dir / "outputs").mkdir(parents=True, exist_ok=True)

    token = secrets.token_urlsafe(18)
    os.environ["PIXELFORGE_DESKTOP"] = "1"
    os.environ["PIXELFORGE_DATA_DIR"] = str(data_dir)
    os.environ["PIXELFORGE_OUTPUTS_DIR"] = str(data_dir / "outputs")
    os.environ["PIXELFORGE_WEIGHTS_DIR"] = str(data_dir / "weights")
    os.environ["PIXELFORGE_FRONTEND_DIR"] = str(frontend)
    os.environ["PIXELFORGE_TOKEN"] = token
    os.chdir(data_dir)

    # Reuse an already running instance (for example: user double-clicks twice).
    existing = read_instance(data_dir)
    if existing:
        url = f"http://{HOST}:{int(existing['port'])}/"
        log.info("PixelForge is already running at %s", url)
        if not args.no_window:
            if not open_native_window(
                url,
                data_dir,
                str(existing.get("token", "")),
                int(existing["port"]),
                threading.Event(),
            ) and not args.no_browser:
                webbrowser.open(url)
        return 0

    log.info("%s %s starting", APP_NAME, APP_VERSION)
    log.info("executable=%s", sys.executable)
    log.info("resource_dir=%s", resource)
    log.info("data_dir=%s", data_dir)
    log.info("log_file=%s", log_file)

    port = args.port or find_free_port()
    if not (1 <= port <= 65535):
        show_error(f"Invalid port: {port}")
        return 2

    sys.path.insert(0, str(app_dir()))
    sys.path.insert(0, str(resource))
    sys.path.insert(0, str(resource / "backend"))

    try:
        import uvicorn

        from backend.app import SHUTDOWN_EVENT, app as fastapi_app
    except Exception as exc:
        show_error(
            "PixelForge could not load its backend.\n\n"
            f"{type(exc).__name__}: {exc}\n\nLog: {log_file}"
        )
        return 1

    config = uvicorn.Config(
        fastapi_app,
        host=HOST,
        port=port,
        log_level="warning",
        log_config=None,
        access_log=False,
        lifespan="on",
        timeout_graceful_shutdown=5,
    )
    server = uvicorn.Server(config)
    # The server runs in a worker thread; signal handlers must stay on main.
    try:
        server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
    except Exception:
        pass

    server_errors: list[BaseException] = []
    server_ready = threading.Event()

    def serve() -> None:
        try:
            server.run()
        except BaseException as exc:  # reported to the main thread
            server_errors.append(exc)
            log.exception("Backend server crashed")
        finally:
            server_ready.set()

    thread = threading.Thread(target=serve, name="pixelforge-server", daemon=True)
    thread.start()

    started_at = time.monotonic()
    try:
        wait_for_server(
            port, server=server, server_errors=server_errors, thread=thread
        )
    except Exception as exc:
        detail = f"{exc}"
        if server_errors:
            detail += f"\n\n{server_errors[0]}"
        show_error(
            "PixelForge could not start its local server.\n\n"
            f"{detail}\n\nLog: {log_file}"
        )
        try:
            server.should_exit = True
        except Exception:
            pass
        return 1

    url = f"http://{HOST}:{port}/"
    write_instance(data_dir, port, token)
    log.info("Server ready: %s (%.1fs)", url, time.monotonic() - started_at)

    def cleanup() -> None:
        try:
            server.should_exit = True
        except Exception:
            pass
        try:
            thread.join(timeout=6)
        except Exception:
            pass
        # Only removes the file when it still belongs to this instance.
        clear_instance(data_dir, token)

    atexit.register(cleanup)

    exit_code = 0
    try:
        if args.selftest or args.selftest_upscale:
            exit_code = run_selftest(
                port, data_dir, upscale=args.selftest_upscale, token=token
            )
        elif args.no_window:
            log.info("Running headless (Ctrl+C to stop)")
            wait_forever(SHUTDOWN_EVENT)
        else:
            if not open_native_window(url, data_dir, token, port, SHUTDOWN_EVENT):
                if args.no_browser:
                    log.info("Native window unavailable; running headless (--no-browser)")
                    wait_forever(SHUTDOWN_EVENT)
                else:
                    log.info("Opening the default browser instead")
                    webbrowser.open(url)
                    wait_forever(SHUTDOWN_EVENT)
    except KeyboardInterrupt:
        log.info("Interrupted")
    except Exception as exc:
        log.exception("Unhandled launcher error")
        show_error(f"PixelForge hit an unexpected error.\n\n{exc}\n\nLog: {log_file}")
        exit_code = 1
    finally:
        cleanup()
        log.info("%s stopped", APP_NAME)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
