"""PyInstaller runtime hook: make bundled DLL folders discoverable.

Runs before torch is imported. Without this Windows can fail to find the
CUDA/cuDNN DLLs that ship inside torch/lib, especially on machines where
CUDA is not installed system-wide.
"""
import os
import sys

if getattr(sys, "frozen", False):
    root = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    candidates = [
        root,
        os.path.join(root, "torch", "lib"),
        os.path.join(root, "torch", "bin"),
        os.path.join(root, "torchvision"),
        os.path.join(root, "cv2"),
    ]
    # The directory is removed again when add_dll_directory's return value is
    # garbage collected, so keep every cookie alive for the whole process.
    _dll_cookies = []
    for folder in candidates:
        if os.path.isdir(folder):
            try:
                _dll_cookies.append(os.add_dll_directory(folder))
            except Exception:
                pass
    # Some libraries still resolve DLLs through PATH.
    os.environ["PATH"] = os.pathsep.join(
        [p for p in candidates if os.path.isdir(p)] + [os.environ.get("PATH", "")]
    )

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")


def _install_dynamo_stub() -> None:
    """Avoid a PyInstaller-only crash while importing torchvision.

    torchvision imports ``torch._dynamo.utils`` (in roi_align) purely to ask
    whether torch.compile is supported. Importing torch._dynamo pulls in
    ``torch._numpy._ufuncs``, whose module-level ``for name in ...:
    vars()[name] = ...`` pattern raises ``NameError: name 'name' is not
    defined`` inside PyInstaller's frozen importer.

    PixelForge only runs inference and never calls torch.compile, so a small
    stub for torch._dynamo is safe and keeps torchvision importable.
    """
    import types
    from importlib.abc import Loader, MetaPathFinder
    from importlib.machinery import ModuleSpec

    def _identity_decorator(fn=None, *args, **kwargs):
        if fn is None:
            return lambda f: f
        return fn

    def _noop(*args, **kwargs):
        return None

    class _StubModule(types.ModuleType):
        def __getattr__(self, name):
            # Keep common decorator usage working (e.g. torch.compiler.disable).
            if name in ("disable", "allow_in_graph", "assume_constant_result", "skip"):
                return _identity_decorator
            if name == "is_compiling":
                return lambda *a, **k: False
            if name == "is_compile_supported":
                return lambda *a, **k: False
            return _noop

    class _StubLoader(Loader):
        def create_module(self, spec):
            module = _StubModule(spec.name)
            module.__path__ = []  # lets submodules (torch._dynamo.utils) import
            module.__doc__ = "PixelForge inference-only torch._dynamo stub"
            return module

        def exec_module(self, module):
            pass

    class _StubFinder(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "torch._dynamo" or fullname.startswith("torch._dynamo."):
                return ModuleSpec(fullname, _StubLoader(), is_package=True)
            return None

    sys.meta_path.insert(0, _StubFinder())


if os.name == "nt":
    _install_dynamo_stub()
