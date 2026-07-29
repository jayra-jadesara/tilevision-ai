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

$bundleSam2 = if ($env:TILEVISION_BUNDLE_SAM2) { $env:TILEVISION_BUNDLE_SAM2.ToLowerInvariant() } else { "" }
$expectSam2 = $bundleSam2 -in @("1", "true", "yes", "on", "auto")
if ($expectSam2) {
    $sam2Onnx = Get-ChildItem -Path $AppDir -Recurse -Filter "*.encoder.onnx" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match "sam2\.1-hiera-tiny-onnx" } |
        Select-Object -First 1
    if (-not $sam2Onnx) {
        Write-Error "TILEVISION_BUNDLE_SAM2=$bundleSam2 but ONNX SAM2 encoder not bundled"
        exit 1
    }
    Write-Host "sam2 onnx: $($sam2Onnx.FullName)"
    $sam2 = Get-ChildItem -Path $AppDir -Recurse -Filter "config.json" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match "sam2\.1-hiera-tiny[\\/]" } |
        Select-Object -First 1
    if ($sam2) {
        Write-Host "sam2 transformers: $($sam2.FullName)"
    } else {
        Write-Host "sam2 transformers optional missing (ONNX still present)"
    }
} else {
    Write-Host "sam2 bundle skipped (TILEVISION_BUNDLE_SAM2=$bundleSam2)"
}

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
