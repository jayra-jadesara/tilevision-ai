# Post-PyInstaller checks for Windows onedir bundle (PyInstaller 6+ uses _internal/).
$ErrorActionPreference = "Stop"

$exe = Get-ChildItem -Path dist -Recurse -Filter "TileVisionAI.exe" -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $exe) {
    Write-Error "TileVisionAI.exe not found under dist/"
    Get-ChildItem -Path dist -Recurse | Select-Object FullName
    exit 1
}
Write-Host "exe: $($exe.FullName)"

$model = Get-ChildItem -Path dist -Recurse -Filter "config.json" -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match "dinov2-large" } |
    Select-Object -First 1
if (-not $model) {
    Write-Error "DINOv2 model (dinov2-large/config.json) not bundled under dist/"
    Get-ChildItem -Path dist -Recurse | Select-Object FullName
    exit 1
}
Write-Host "model: $($model.FullName)"
Write-Host "Windows bundle OK"
