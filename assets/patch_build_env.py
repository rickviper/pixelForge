"""Patch the Windows build environment before PyInstaller runs.

Both changes work around library code that runs at module import time and
breaks only inside PyInstaller's frozen importer. They are applied to the
disposable build virtualenv, never to the user's system Python.
"""
from __future__ import annotations

import os


def patch_torchvision() -> None:
    """Restore torchvision.transforms.functional_tensor for basicsr 1.4.2.

    basicsr still imports this module, which torchvision removed in 0.19.
    The runtime shim lives in backend/upscaler_engine.py; this real module is
    needed because PyInstaller's isolated workers import basicsr while
    collecting its submodules.
    """
    import torchvision

    path = os.path.join(
        os.path.dirname(torchvision.__file__), "transforms", "functional_tensor.py"
    )
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("from .functional import rgb_to_grayscale\n")
        print("wrote torchvision compatibility module:", path)
    else:
        print("torchvision compatibility module already present:", path)


def patch_scipy_stats() -> None:
    """Fix scipy.stats under PyInstaller.

    ``scipy/stats/_distn_infrastructure.py`` ends a module-level cleanup loop
    with ``del obj``. Inside the frozen importer the list comprehension can be
    empty, leaving ``obj`` unbound and raising
    ``NameError: name 'obj' is not defined``. basicsr imports scipy.stats
    eagerly, so that crash would break every AI upscale.
    """
    import scipy

    path = os.path.join(
        os.path.dirname(scipy.__file__), "stats", "_distn_infrastructure.py"
    )
    with open(path, encoding="utf-8") as handle:
        source = handle.read()

    old = "    exec('del ' + obj)\ndel obj\n"
    new = "    exec('del ' + obj)\nglobals().pop('obj', None)\n"
    if old in source:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(source.replace(old, new, 1))
        print("patched scipy for the frozen importer:", path)
    else:
        print("scipy patch not needed (already patched or upstream changed):", path)


if __name__ == "__main__":
    patch_torchvision()
    patch_scipy_stats()
