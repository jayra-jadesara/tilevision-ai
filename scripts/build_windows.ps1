#Requires -Version 5.1
<#
.SYNOPSIS
    Build TileVision AI for Windows: PyInstaller + Inno Setup installer.

.DESCRIPTION
    1. Downloads DINOv2 weights if missing (~1 GB, needs internet once)
    2. Runs PyInstaller (one-folder build)
    3. Compiles Inno Setup installer when ISCC.exe is available

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "=== TileVision AI — Windows Release Build ===" -ForegroundColor Cyan

if ($env:TILEVISION_DEV_MODE) {
    Write-Warning "TILEVISION_DEV_MODE is set — unset it before a production build."
    exit 1
}

Write-Host "`n[1/4] Checking Python dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip | Out-Null
python -m pip install -r requirements.txt pyinstaller | Out-Null

Write-Host "`n[2/4] Ensuring DINOv2 model weights..." -ForegroundColor Yellow
$ModelDir = Join-Path $ProjectRoot "model_weights\dinov2-large"
$ConfigFile = Join-Path $ModelDir "config.json"
if (-not (Test-Path $ConfigFile)) {
    Write-Host "  Downloading DINOv2 (~1 GB) — this may take several minutes..."
    python scripts/download_dinov2_model.py
} else {
    Write-Host "  Model weights already present at $ModelDir"
}

# Optional SAM2 Precise Crop weights (lab). Default auto = ONNX on Windows.
if (-not $env:TILEVISION_BUNDLE_SAM2) {
    $env:TILEVISION_BUNDLE_SAM2 = "auto"
}
$bundleSam2 = $env:TILEVISION_BUNDLE_SAM2.ToLowerInvariant()
if ($bundleSam2 -in @("1", "true", "yes", "on", "auto")) {
    Write-Host "`n[2b/4] Ensuring ONNX SAM2 Precise Crop weights (Windows CPU)..." -ForegroundColor Yellow
    $OnnxDir = Join-Path $ProjectRoot "model_weights\sam2.1-hiera-tiny-onnx"
    $OnnxEnc = Join-Path $OnnxDir "sam2.1_hiera_tiny.encoder.onnx"
    if (-not (Test-Path $OnnxEnc)) {
        Write-Host "  Downloading ONNX SAM2 tiny (~126 MB)..."
        python scripts/download_sam2_onnx_model.py
    } else {
        Write-Host "  ONNX SAM2 weights already present at $OnnxDir"
    }
    Write-Host "`n[2c/4] Ensuring Transformers SAM2 weights (optional upgrade path)..." -ForegroundColor Yellow
    $Sam2Dir = Join-Path $ProjectRoot "model_weights\sam2.1-hiera-tiny"
    $Sam2Config = Join-Path $Sam2Dir "config.json"
    if (-not (Test-Path $Sam2Config)) {
        Write-Host "  Downloading SAM2 tiny safetensors (~150 MB)..."
        python scripts/download_sam2_model.py
    } else {
        Write-Host "  Transformers SAM2 weights already present at $Sam2Dir"
    }
}

Write-Host "`n[3/4] Running PyInstaller..." -ForegroundColor Yellow
if (Test-Path "dist\TileVisionAI") {
    Remove-Item -Recurse -Force "dist\TileVisionAI"
}
if (Test-Path "build") {
    Remove-Item -Recurse -Force "build"
}
pyinstaller packaging/tilevision.spec --clean --noconfirm

$ExePath = Join-Path $ProjectRoot "dist\TileVisionAI\TileVisionAI.exe"
if (-not (Test-Path $ExePath)) {
    Write-Error "PyInstaller build failed — TileVisionAI.exe not found."
}

Write-Host "  Built: $ExePath" -ForegroundColor Green

Write-Host "`n[4/5] Building Inno Setup installer..." -ForegroundColor Yellow
$IsccCandidates = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
)
$Iscc = $IsccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $Iscc) {
    Write-Warning @"
Inno Setup 6 not found — skipping installer compilation.
Install from: https://jrsoftware.org/isinfo.php
Then run:  iscc packaging\tilevision_setup.iss

PyInstaller output is ready at: dist\TileVisionAI\
"@
    exit 0
}

$InstallerDir = Join-Path $ProjectRoot "dist\installer"
New-Item -ItemType Directory -Force -Path $InstallerDir | Out-Null
& $Iscc "packaging\tilevision_setup.iss"

$Installer = Get-ChildItem -Path $InstallerDir -Filter "TileVisionAI-Setup-*.exe" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($Installer) {
    Write-Host "`nDone." -ForegroundColor Green
    Write-Host "  App folder:  dist\TileVisionAI\" -ForegroundColor Green
    Write-Host "  Installer:   $($Installer.FullName)" -ForegroundColor Green
} else {
    Write-Warning "Inno Setup ran but no installer .exe was found in dist\installer\"
}

Write-Host "`n[5/5] Building vendor admin tool (Windows only, do not ship to customers)..." -ForegroundColor Yellow
if (Test-Path "dist\TileVisionAI-Admin") {
    Remove-Item -Recurse -Force "dist\TileVisionAI-Admin"
}
pyinstaller packaging/tilevision_admin.spec --clean --noconfirm

$AdminExePath = Join-Path $ProjectRoot "dist\TileVisionAI-Admin\TileVisionAI-Admin.exe"
if (-not (Test-Path $AdminExePath)) {
    Write-Warning "Admin PyInstaller build failed — TileVisionAI-Admin.exe not found."
    exit 0
}

Write-Host "  Built admin: $AdminExePath" -ForegroundColor Green

if ($Iscc) {
    & $Iscc "packaging\tilevision_admin_setup.iss"
    $AdminInstaller = Get-ChildItem -Path $InstallerDir -Filter "TileVisionAI-Admin-VENDOR-ONLY-*.exe" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($AdminInstaller) {
        Write-Host "  Admin installer (vendor only): $($AdminInstaller.FullName)" -ForegroundColor Green
    }
}
