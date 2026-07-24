# Post-PyInstaller checks for Windows onedir bundle (PyInstaller 6+ uses _internal/).
param(
    [string]$AppDir = "dist"
)

$ErrorActionPreference = "Stop"

$exe = Get-ChildItem -Path $AppDir -Recurse -Filter "TileVisionAI.exe" -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $exe) {
    Write-Error "TileVisionAI.exe not found under $AppDir/"
    Get-ChildItem -Path $AppDir -Recurse -ErrorAction SilentlyContinue | Select-Object FullName
    exit 1
}
Write-Host "exe: $($exe.FullName)"

$model = Get-ChildItem -Path $AppDir -Recurse -Filter "config.json" -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match "dinov2-large" } |
    Select-Object -First 1
if (-not $model) {
    Write-Error "DINOv2 model (dinov2-large/config.json) not bundled under $AppDir/"
    Get-ChildItem -Path $AppDir -Recurse -ErrorAction SilentlyContinue | Select-Object FullName
    exit 1
}
Write-Host "model: $($model.FullName)"
Write-Host "Windows bundle OK"
