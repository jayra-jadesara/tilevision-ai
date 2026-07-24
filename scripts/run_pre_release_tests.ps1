# Full pre-release test suite for TileVision AI v1.0.1+
#
# Runs automated checks you can do on Windows before sending to clients.
# Mac Intel must be validated separately using packaging/MAC_BETA_TEST.txt
# on the client's iMac (or any Intel Mac).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/run_pre_release_tests.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/run_pre_release_tests.ps1 -SkipPytest
#   powershell -ExecutionPolicy Bypass -File scripts/run_pre_release_tests.ps1 -InstallerPath "C:\path\TileVisionAI-Setup-1.0.1.exe"

param(
    [switch]$SkipPytest,
    [switch]$SkipInstall,
    [string]$InstallerPath = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $MyInvocation.MyCommand.Path -Parent) -Parent
Set-Location $Root

$ReleaseDir = Join-Path $Root "test_release"
$InstallDir = Join-Path $ReleaseDir "installed"
$DefaultInstaller = Join-Path $ReleaseDir "TileVisionAI-Setup-1.0.2.exe"
$ReleaseUrl = "https://github.com/jayra-jadesara/tilevision-ai/releases/download/v1.0.2/TileVisionAI-Setup-1.0.2.exe"

$results = @()

function Record-Result($Name, $Passed, $Detail = "") {
    $script:results += [PSCustomObject]@{ Step = $Name; Pass = $Passed; Detail = $Detail }
    $color = if ($Passed) { "Green" } else { "Red" }
    $mark = if ($Passed) { "PASS" } else { "FAIL" }
    Write-Host "[$mark] $Name" -ForegroundColor $color
    if ($Detail) { Write-Host "       $Detail" }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " TileVision AI - Pre-release test run" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. CI config validation
try {
    python scripts/validate_ci_build.py | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "validate_ci_build failed" }
    Record-Result "CI build config validation" $true
} catch {
    Record-Result "CI build config validation" $false $_.Exception.Message
}

# 2. Preflight (dev environment)
try {
    python scripts/preflight_check.py | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "preflight_check failed" }
    Record-Result "Dev preflight check" $true
} catch {
    Record-Result "Dev preflight check" $false $_.Exception.Message
}

# 3. Pytest
if (-not $SkipPytest) {
    try {
        Write-Host "Running pytest (not slow) - may take several minutes..." -ForegroundColor Yellow
        python -m pytest tests/ -q -m "not slow" --tb=line
        if ($LASTEXITCODE -ne 0) { throw "pytest exit $LASTEXITCODE" }
        Record-Result "Unit/integration tests (pytest)" $true
    } catch {
        Record-Result "Unit/integration tests (pytest)" $false $_.Exception.Message
    }
} else {
    Record-Result "Unit/integration tests (pytest)" $true "skipped"
}

# 4. Release test files exist in repo
$required = @(
    "packaging/MAC_BETA_TEST.txt",
    "packaging/MAC_INSTALL.txt",
    "scripts/smoke_test_windows.ps1",
    "scripts/verify_frozen_windows.ps1"
)
$missing = @($required | Where-Object { -not (Test-Path $_) })
if ($missing.Count -eq 0) {
    Record-Result "Release helper files present" $true
} else {
    Record-Result "Release helper files present" $false ($missing -join ", ")
}

# 5. Download / locate Windows installer
if (-not $InstallerPath) { $InstallerPath = $DefaultInstaller }
if (-not (Test-Path $InstallerPath)) {
    Write-Host "Installer not found locally. Downloading from GitHub release..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $ReleaseDir | Out-Null
    try {
        Invoke-WebRequest -Uri $ReleaseUrl -OutFile $InstallerPath -UseBasicParsing
        Record-Result "Download v1.0.1 Windows installer" $true $InstallerPath
    } catch {
        Record-Result "Download v1.0.1 Windows installer" $false $_.Exception.Message
        $InstallerPath = ""
    }
} else {
    $sizeGb = [math]::Round((Get-Item $InstallerPath).Length / 1GB, 2)
    Record-Result "Windows installer on disk" $true "${InstallerPath} (${sizeGb} GB)"
}

# 6. Silent install + bundle smoke test
if ($InstallerPath -and (Test-Path $InstallerPath) -and -not $SkipInstall) {
    try {
        if (Test-Path $InstallDir) { Remove-Item -Recurse -Force $InstallDir }
        New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
        Write-Host "Installing silently to $InstallDir ..." -ForegroundColor Yellow
        $proc = Start-Process -FilePath $InstallerPath -ArgumentList @(
            "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
            "/DIR=$InstallDir"
        ) -Wait -PassThru
        if ($proc.ExitCode -ne 0) { throw "Installer exit code $($proc.ExitCode)" }
        Record-Result "Silent install Windows build" $true $InstallDir
    } catch {
        Record-Result "Silent install Windows build" $false $_.Exception.Message
    }

    try {
        powershell -ExecutionPolicy Bypass -File scripts/smoke_test_windows.ps1 -AppDir $InstallDir -NonInteractive
        if ($LASTEXITCODE -ne 0) { throw "smoke test failed" }
        Record-Result "Windows bundle smoke test (exe + model)" $true
    } catch {
        Record-Result "Windows bundle smoke test (exe + model)" $false $_.Exception.Message
    }
} elseif ($SkipInstall) {
    Record-Result "Silent install + bundle smoke" $true "skipped"
} else {
    Record-Result "Silent install + bundle smoke" $false "no installer"
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
$passed = @($results | Where-Object { $_.Pass }).Count
$failed = @($results | Where-Object { -not $_.Pass }).Count
$results | Format-Table -AutoSize
Write-Host "Passed: $passed  Failed: $failed" -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Yellow" })

Write-Host ""
Write-Host "MANUAL - still required before client delivery:" -ForegroundColor Yellow
Write-Host @"
  Windows (you, ~15 min):
    1. Launch installed app from: $InstallDir
    2. Activate trial license (admin_tool + Machine ID)
    3. Index 10-20 tile photos -> search -> export PDF -> restart app

  Mac Intel (client iMac 2020, ~20 min):
    1. Send: TileVisionAI-macOS-Intel-1.0.2.dmg + packaging/MAC_BETA_TEST.txt
    2. Client completes all 7 steps and reports back
    3. Only then send final license / go-live

  Release page: https://github.com/jayra-jadesara/tilevision-ai/releases/tag/v1.0.2
"@

if ($failed -gt 0) { exit 1 }
exit 0
