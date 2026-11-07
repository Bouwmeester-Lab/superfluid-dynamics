<#
.SYNOPSIS
    Runs the gtest suite and the Python golden-regression tests.

.DESCRIPTION
    Uses the build output of scripts/build.ps1. Python is taken from
    CUSH_PYTHON if set, otherwise from PATH. Set CUSH_SHOW_PLOTS=1 to open
    figure windows from visual tests; by default nothing blocks.

.EXAMPLE
    ./scripts/test.ps1
    ./scripts/test.ps1 -Build
    ./scripts/test.ps1 -Rebuild     # full rebuild first; use after editing headers
    ./scripts/test.ps1 -GTestFilter "Radial*"
    ./scripts/test.ps1 -SkipGTest
#>
param(
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release",
    [switch]$Build,
    [switch]$Rebuild,
    [switch]$SkipGTest,
    [switch]$SkipPython,
    [string]$GTestFilter = "*",
    [string]$PytestArgs = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$outDir = Join-Path $repoRoot "CuSuperHelium\x64\$Configuration"

if ($Build -or $Rebuild) {
    & (Join-Path $PSScriptRoot "build.ps1") -Configuration $Configuration -Rebuild:$Rebuild
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$failed = @()

if (-not $SkipGTest) {
    $gtest = Join-Path $outDir "CuSuperHelium.Tests.exe"
    if (-not (Test-Path $gtest)) { throw "Test executable not found: $gtest. Run scripts/build.ps1 first." }
    Write-Host "`n=== gtest ($GTestFilter) ==="
    # Run from the output folder so the test executable finds its DLLs.
    # Native stderr must not stop the script (Windows PowerShell 5.1 turns it into errors); the exit code decides.
    Push-Location $outDir
    $ErrorActionPreference = "Continue"
    try { & $gtest "--gtest_filter=$GTestFilter" 2>&1 | ForEach-Object { "$_" } } finally { Pop-Location; $ErrorActionPreference = "Stop" }
    if ($LASTEXITCODE -ne 0) { $failed += "gtest" }
}

if (-not $SkipPython) {
    $python = $env:CUSH_PYTHON
    if (-not $python) {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if (-not $cmd) { throw "Python not found. Put it on PATH or set CUSH_PYTHON to python.exe." }
        $python = $cmd.Source
    }
    $dll = Join-Path $outDir "CuSuperHelium.dll"
    if (-not (Test-Path $dll)) { throw "DLL not found: $dll. Run scripts/build.ps1 first." }
    $env:CUSUPERHELIUM_DLL = $dll
    Write-Host "`n=== pytest (DLL: $dll) ==="
    $testsDir = Join-Path $repoRoot "CuSuperHelium\Python\tests"
    $extra = if ($PytestArgs) { $PytestArgs -split " " } else { @() }
    $ErrorActionPreference = "Continue"
    try { & $python -m pytest $testsDir -p no:cacheprovider @extra 2>&1 | ForEach-Object { "$_" } } finally { $ErrorActionPreference = "Stop" }
    if ($LASTEXITCODE -ne 0) { $failed += "pytest" }
}

if ($failed.Count -gt 0) {
    Write-Host "`nFailed: $($failed -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host "`nAll tests passed."
