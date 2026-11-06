#!/usr/bin/env bash
# Build the portable Windows app from WSL/Linux.
#
# Windows tooling (PyInstaller) cannot run on Linux, so this script copies the
# project to a real Windows folder, runs build_windows.ps1 there, and copies the
# finished portable folder back to ./dist/PixelForge-Windows.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd /mnt/c 2>/dev/null || cd "$ROOT"

if ! command -v cmd.exe >/dev/null 2>&1; then
  echo "ERROR: cmd.exe not found. Run this from WSL on Windows." >&2
  exit 1
fi
if ! command -v powershell.exe >/dev/null 2>&1; then
  echo "ERROR: powershell.exe not found. Run this from WSL on Windows." >&2
  exit 1
fi

WIN_TEMP="$(cmd.exe /c "echo %TEMP%" | tr -d '\r')"
BUILD_DIR="$(wslpath -u "$WIN_TEMP")/pixelforge-build"
echo "Windows build folder: $(wslpath -w "$BUILD_DIR")"

# Keep .venv-win between runs if you want fast rebuilds. Set PF_FRESH=1 to
# start over (for example after changing the torch/CUDA version).
if [ "${PF_FRESH:-0}" = "1" ]; then
  rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"

echo "Copying project to the Windows build folder..."
tar -cf - \
  --exclude='./.venv' \
  --exclude='./.venv-win' \
  --exclude='./dist' \
  --exclude='./outputs' \
  --exclude='./uploads' \
  --exclude='./PixelForge-Data' \
  --exclude='./.git' \
  --exclude='./__pycache__' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.zip' \
  -C "$ROOT" . | tar -xf - -C "$BUILD_DIR"

echo "Running Windows build (first run downloads several GB of wheels)..."
powershell.exe -NoProfile -ExecutionPolicy Bypass \
  -File "$(wslpath -w "$BUILD_DIR/build_windows.ps1")"

DEST="$ROOT/dist/PixelForge-Windows"
mkdir -p "$ROOT/dist"
echo "Copying portable build back to WSL..."
rm -rf "$DEST"
cp -a "$BUILD_DIR/dist/PixelForge" "$DEST"

echo
echo "Portable folder: $DEST"
echo "Windows path:    $(wslpath -w "$DEST")"
echo "Executable:      $(wslpath -w "$DEST/PixelForge.exe")"
du -sh "$DEST"
