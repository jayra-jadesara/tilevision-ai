# Install CUDA-enabled PyTorch for TileVision AI (Windows).
# Run from the project root:
#   powershell -ExecutionPolicy Bypass -File scripts/install_pytorch_cuda.ps1
#
# Requires: NVIDIA GPU + NVIDIA driver (CUDA toolkit not required).

$ErrorActionPreference = "Stop"

Write-Host "=== TileVision AI - CUDA PyTorch Installer ===" -ForegroundColor Cyan

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) {
    Write-Host ""
    Write-Host "ERROR: nvidia-smi not found." -ForegroundColor Red
    Write-Host "This PC needs an NVIDIA GPU with the NVIDIA driver installed." -ForegroundColor Yellow
    Write-Host "AMD / Intel graphics cannot run CUDA PyTorch." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Detected graphics adapters:" -ForegroundColor Cyan
    Get-CimInstance Win32_VideoController | ForEach-Object {
        Write-Host ("  - {0}" -f $_.Name)
    }
    Write-Host ""
    Write-Host "TileVision will continue to use CPU on this machine." -ForegroundColor Yellow
    exit 1
}

Write-Host "NVIDIA driver detected:" -ForegroundColor Green
& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

Write-Host ""
Write-Host "Removing CPU-only torch/torchvision (if present)..." -ForegroundColor Yellow
python -m pip uninstall -y torch torchvision torchaudio 2>$null | Out-Null

Write-Host "Installing CUDA 12.4 PyTorch wheels..." -ForegroundColor Yellow
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Write-Host ""
Write-Host "Verifying installation..." -ForegroundColor Cyan
python dev_tools/check_gpu.py
$checkExit = $LASTEXITCODE

if ($checkExit -eq 0) {
    Write-Host ""
    Write-Host "SUCCESS - restart TileVision AI to use GPU." -ForegroundColor Green
    exit 0
}

Write-Host ""
Write-Host "GPU still not active. Check driver and re-run: python dev_tools/check_gpu.py" -ForegroundColor Red
exit 1
