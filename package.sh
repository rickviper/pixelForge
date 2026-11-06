#!/bin/bash
# Build a downloadable zip bundle of PixelForge (excludes venv, weights, results).
set -e
cd "$(dirname "$0")"
VERSION="${1:-1.0.0}"
NAME="pixelforge-upscaler-v${VERSION}"
TMP="/tmp/${NAME}"

rm -rf "$TMP" "${NAME}.zip"
mkdir -p "$TMP" uploads outputs
touch uploads/.gitkeep outputs/.gitkeep
cp -r backend frontend assets pyproject.toml README.md run.sh desktop.py \
  pixelforge.spec build_windows.ps1 build_windows.sh requirements-desktop.txt \
  Dockerfile docker-compose.yml .dockerignore "$TMP/"
mkdir -p "$TMP/weights" && touch "$TMP/weights/.gitkeep"
cp uploads/.gitkeep "$TMP/uploads/" 2>/dev/null || true
cp outputs/.gitkeep "$TMP/outputs/" 2>/dev/null || true
find "$TMP" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
# Zip with python stdlib (no `zip` binary needed)
"$(command -v python3)" - "$TMP" "${NAME}.zip" <<'EOF'
import sys, zipfile
from pathlib import Path
src, out = Path(sys.argv[1]), Path(sys.argv[2])
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(src.rglob('*')):
        z.write(p, p.relative_to(src.parent))
print('files:', len(z.namelist()))
EOF
rm -rf "$TMP"
echo "Built ${NAME}.zip ($(du -h ${NAME}.zip | cut -f1))"
echo "Contents: backend + frontend + Dockerfile + run/install scripts."
echo "Weights (~65MB/model) auto-download on first upscale; not bundled."
