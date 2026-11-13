<#
.SYNOPSIS
    Builds the CuSuperHelium solution (DLL, tests and app).

.DESCRIPTION
    Finds MSBuild with vswhere, preferring the newest Visual Studio including
    previews, because the projects use the v145 toolset (Visual Studio 18).
    Set CUSH_MSBUILD to the full path of MSBuild.exe to override the search.

    Use -Rebuild after editing headers. The CUDA build tracks header
    dependencies through nvcc's generated .deps files, and these miss some
    included .cuh files, so an incremental build can leave stale binaries.

.EXAMPLE
    ./scripts/build.ps1
    ./scripts/build.ps1 -Rebuild
    ./scripts/build.ps1 -Configuration Debug
#>
param(
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release",
    [string]$Platform = "x64",
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$solution = Join-Path $repoRoot "CuSuperHelium\CuSuperHelium.sln"

function Find-MSBuild {
    if ($env:CUSH_MSBUILD) {
        if (-not (Test-Path $env:CUSH_MSBUILD)) { throw "CUSH_MSBUILD points to a missing file: $env:CUSH_MSBUILD" }
        return $env:CUSH_MSBUILD
    }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found. Install Visual Studio or set CUSH_MSBUILD." }
    $found = & $vswhere -latest -prerelease -products * -requires Microsoft.Component.MSBuild -find "MSBuild\**\Bin\MSBuild.exe" | Select-Object -First 1
    if (-not $found) { throw "MSBuild not found by vswhere. Set CUSH_MSBUILD." }
    return $found
}

$msbuild = Find-MSBuild
Write-Host "MSBuild: $msbuild"
$target = if ($Rebuild) { "Rebuild" } else { "Build" }
Write-Host "$target $Configuration|$Platform"

# Native stderr must not stop the script (Windows PowerShell 5.1 turns it into errors); the exit code decides.
$ErrorActionPreference = "Continue"
& $msbuild $solution /m /nologo /verbosity:minimal "/t:$target" "/p:Configuration=$Configuration" "/p:Platform=$Platform" 2>&1 | ForEach-Object { "$_" }
$ErrorActionPreference = "Stop"
if ($LASTEXITCODE -ne 0) {
    $code = $LASTEXITCODE
    Write-Host "Build failed with exit code $code" -ForegroundColor Red
    exit $code
}
Write-Host "Build succeeded."
