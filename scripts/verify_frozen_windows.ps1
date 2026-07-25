# Post-PyInstaller checks for Windows onedir bundle (PyInstaller 6+ uses _internal/).
param(
    [string]$AppDir = "dist"
)

$ErrorActionPreference = "Stop"

function Get-BundleRoot([string]$ExePath) {
    $dir = Split-Path $ExePath -Parent
    $internal = Join-Path $dir "_internal"
    if (Test-Path $internal) { return $internal }
    return $dir
}

$exe = Get-ChildItem -Path $AppDir -Recurse -Filter "TileVisionAI.exe" -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $exe) {
    Write-Error "TileVisionAI.exe not found under $AppDir/"
    exit 1
}
Write-Host "exe: $($exe.FullName)"

$root = Get-BundleRoot $exe.FullName

$model = Get-ChildItem -Path $AppDir -Recurse -Filter "config.json" -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match "dinov2-large" } |
    Select-Object -First 1
if (-not $model) {
    Write-Error "DINOv2 model (dinov2-large/config.json) not bundled"
    exit 1
}
Write-Host "model: $($model.FullName)"

$torchDir = Join-Path $root "torch"
if (-not (Test-Path $torchDir)) {
    Write-Error "torch package missing from bundle ($torchDir)"
    exit 1
}
Write-Host "torch: $torchDir"

$cudaDir = Join-Path $torchDir "cuda"
if (-not (Test-Path $cudaDir)) {
    Write-Error "torch.cuda missing from bundle ($cudaDir) - app will crash on startup"
    exit 1
}
Write-Host "torch.cuda: $cudaDir"

Write-Host "Windows bundle OK"
