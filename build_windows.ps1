param(
    [string]$TorchIndex = $(if ($env:TORCH_INDEX) { $env:TORCH_INDEX } else { "https://download.pytorch.org/whl/cu130" }),
    [switch]$SkipInstall,
    [switch]$SkipSelftest
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Write-Step([string]$Text) {
    Write-Host ""
    Write-Host "=== $Text ===" -ForegroundColor Cyan
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv was not found on PATH. Install it from https://astral.sh/uv and retry."
}

$Venv = Join-Path $PSScriptRoot ".venv-win"
$Py = Join-Path $Venv "Scripts\python.exe"

Write-Step "Python 3.12 environment"
if (-not (Test-Path $Py)) {
    uv venv --python 3.12 $Venv
}
& $Py --version

if (-not $SkipInstall) {
    Write-Step "Installing CUDA PyTorch ($TorchIndex)"
    uv pip install --python $Py torch torchvision --index-url $TorchIndex
    if ($LASTEXITCODE -ne 0) { throw "Could not install torch/torchvision" }

    Write-Step "Installing application + packaging dependencies"
    uv pip install --python $Py -r requirements-desktop.txt
    if ($LASTEXITCODE -ne 0) { throw "Could not install requirements-desktop.txt" }

    # basicsr/gfpgan declare torch as a dependency, so the resolver above can
    # replace the CUDA wheel with the CPU wheel from PyPI. Put the CUDA build
    # back (it is already in uv's cache, so this is quick).
    Write-Step "Re-asserting the CUDA PyTorch build"
    uv pip install --python $Py --upgrade torch torchvision --index-url $TorchIndex
    if ($LASTEXITCODE -ne 0) { throw "Could not restore the CUDA torch build" }
}

Write-Step "Checking native window + GPU support"
& $Py -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))"
& $Py -c "import webview; print('pywebview', getattr(webview, '__version__', 'ok'))"

# basicsr 1.4.2 imports torchvision.transforms.functional_tensor, which was
# removed in torchvision >= 0.19. PyInstaller's isolated workers also import
# scipy.stats, which needs a frozen-importer fix. Both patches only touch this
# disposable build virtualenv.
Write-Step "Patching build environment for the frozen importer"
& $Py (Join-Path $PSScriptRoot "assets\patch_build_env.py")
if ($LASTEXITCODE -ne 0) { throw "Could not patch the build environment" }

Write-Step "Building portable folder with PyInstaller"
& $Py -m PyInstaller --noconfirm --clean pixelforge.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$Dist = Join-Path $PSScriptRoot "dist\PixelForge"
$Exe = Join-Path $Dist "PixelForge.exe"
if (-not (Test-Path $Exe)) { throw "Build finished but $Exe is missing" }

Copy-Item (Join-Path $PSScriptRoot "README.md") (Join-Path $Dist "README.md") -Force
@"
PixelForge portable build
========================

Double-click PixelForge.exe to start the app.

- No installation needed. Keep this whole folder together.
- Outputs, downloaded model weights and logs are stored in PixelForge-Data\
  next to the executable.
- The app opens in its own desktop window (WebView2). If WebView2 is missing,
  it falls back to your default browser.
- Windows may show a SmartScreen warning the first time because this build is
  not code-signed. Choose "More info" -> "Run anyway".
- GPU: an NVIDIA driver with CUDA support is required for CUDA inference.
  Without one, the app falls back to CPU.
"@ | Set-Content -Path (Join-Path $Dist "RUNNING.txt") -Encoding UTF8

Write-Host ""
Write-Host "Portable build: $Dist" -ForegroundColor Green
$size = (Get-ChildItem $Dist -Recurse | Measure-Object -Property Length -Sum).Sum
Write-Host ("Size: {0:N2} GB" -f ($size / 1GB))

if (-not $SkipSelftest) {
    Write-Step "Self-test (server + device + one GPU upscale)"
    $TestData = Join-Path $PSScriptRoot "dist\selftest-data"
    if (Test-Path $TestData) { Remove-Item -Recurse -Force $TestData }
    $proc = Start-Process -FilePath $Exe -ArgumentList "--selftest-upscale --data-dir `"$TestData`"" -PassThru
    if (-not $proc.WaitForExit(600000)) {
        $proc.Kill()
        throw "Self-test timed out after 10 minutes"
    }
    $ResultFile = Join-Path $TestData "selftest.json"
    if (Test-Path $ResultFile) {
        $result = Get-Content $ResultFile -Raw | ConvertFrom-Json
        $result | ConvertTo-Json -Depth 8
        if ($result.used_cuda -ne $true) {
            throw "Self-test ran but CUDA was not used. The bundled torch/CUDA libraries may be incomplete."
        }
        if ($result.used_ai -ne $true) {
            throw "Self-test fell back to the classical engine; the AI runtime is incomplete."
        }
    } else {
        throw "Self-test did not produce selftest.json (the app crashed before testing)"
    }
    if ($proc.ExitCode -ne 0) {
        $log = Join-Path $TestData "logs\pixelforge.log"
        if (Test-Path $log) { Get-Content $log -Tail 80 }
        throw "Self-test failed with exit code $($proc.ExitCode)"
    }
    Write-Host "Self-test passed." -ForegroundColor Green
}

Write-Host ""
Write-Host "Done. Run: $Exe" -ForegroundColor Green
