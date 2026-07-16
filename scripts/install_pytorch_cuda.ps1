# Install CUDA-enabled PyTorch for TileVision AI (Windows).
# Run from the project root:
#   powershell -ExecutionPolicy Bypass -File scripts/install_pytorch_cuda.ps1
#
# Requires: NVIDIA GPU + up-to-date driver (no separate CUDA toolkit needed).

$ErrorActionPreference = "Stop"

Write-Host "=== TileVision AI — CUDA PyTorch Installer ===" -ForegroundColor Cyan

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) {
    Write-Host "WARNING: nvidia-smi not found. Install the latest NVIDIA driver first." -ForegroundColor Yellow
} else {
    Write-Host "NVIDIA driver detected:" -ForegroundColor Green
    & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
}

Write-Host ""
Write-Host "Removing CPU-only torch/torchvision (if present)..." -ForegroundColor Yellow
python -m pip uninstall -y torch torchvision torchaudio 2>$null

Write-Host "Installing CUDA 12.4 PyTorch wheels..." -ForegroundColor Yellow
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Write-Host ""
Write-Host "Verifying installation..." -ForegroundColor Cyan
python dev_tools/check_gpu.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "SUCCESS — restart TileVision AI to use GPU." -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "GPU still not active. Check driver and re-run check_gpu.py." -ForegroundColor Red
    exit 1
}
