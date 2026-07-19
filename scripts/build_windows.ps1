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

Write-Host "`n[4/4] Building Inno Setup installer..." -ForegroundColor Yellow
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
