# Smoke-test TileVision AI Windows installer BEFORE sending v1.0.1 to customers.
#
# Usage (after installing from release):
#   powershell -ExecutionPolicy Bypass -File scripts/smoke_test_windows.ps1
#
# Or test the portable build folder without installing:
#   powershell -ExecutionPolicy Bypass -File scripts/smoke_test_windows.ps1 -AppDir "C:\path\to\TileVisionAI"

param(
    [string]$AppDir = "",
    [switch]$NonInteractive
)

$ErrorActionPreference = "Stop"

function Find-TileVisionExe {
    param([string]$Root)
    if ($Root -and (Test-Path $Root)) {
        $hit = Get-ChildItem -Path $Root -Recurse -Filter "TileVisionAI.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($hit) { return $hit }
    }
    $candidates = @(
        "${env:ProgramFiles}\TileVision AI\TileVisionAI.exe",
        "${env:ProgramFiles(x86)}\TileVision AI\TileVisionAI.exe",
        "${env:LocalAppData}\Programs\TileVision AI\TileVisionAI.exe"
    )
    foreach ($path in $candidates) {
        if (Test-Path $path) { return Get-Item $path }
    }
    return $null
}

Write-Host "=== TileVision AI - Windows pre-release smoke test ===" -ForegroundColor Cyan
Write-Host ""

$exe = Find-TileVisionExe -Root $AppDir
if (-not $exe) {
    Write-Host "FAIL: TileVisionAI.exe not found." -ForegroundColor Red
    Write-Host "Install from TileVisionAI-Setup-1.0.1.exe first, or pass -AppDir to dist\TileVisionAI"
    exit 1
}
Write-Host "[OK] exe: $($exe.FullName)"

$root = $exe.Directory.FullName
$model = Get-ChildItem -Path $root -Recurse -Filter "config.json" -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match "dinov2-large" } |
    Select-Object -First 1
if (-not $model) {
    Write-Host "FAIL: DINOv2 model not bundled (dinov2-large/config.json missing)" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] model: $($model.FullName)"

Write-Host ""
Write-Host "Automated checks passed." -ForegroundColor Green
Write-Host ""
Write-Host "=== Manual steps (do these before sending to ANY client) ===" -ForegroundColor Yellow
Write-Host @"

1. LAUNCH
   - Double-click TileVision AI (or run the exe above)
   - App opens without 'Python not found' or crash

2. LICENSE
   - Open License / Activation screen
   - Copy Machine ID (64 hex chars)
   - Generate trial key in admin_tool for THAT Machine ID
   - Paste key -> app shows licensed / trial active

3. INDEX (offline OK after install)
   - Create a folder with 10-20 tile photos (jpg/png)
   - Index that folder -> wait until complete, no error dialog

4. SEARCH
   - Use one tile photo as query
   - Results show similar tiles from indexed folder

5. PDF EXPORT
   - Export search results or catalogue to PDF
   - Open PDF -> images visible

6. RESTART
   - Close app, reopen -> still licensed, index still there

If ALL 6 pass on your Windows PC, the Windows build is OK for customers.

"@

$launch = "n"
if (-not $NonInteractive) {
    $launch = Read-Host "Launch TileVision AI now for manual testing? (y/n)"
}
if ($launch -eq "y") {
    Start-Process -FilePath $exe.FullName
    Write-Host "App launched. Complete steps 2-6 above."
}
