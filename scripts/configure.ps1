<#
.SYNOPSIS
    Writes CuSuperHelium/Directory.Build.user.props: the machine-specific build settings.

.DESCRIPTION
    Run once per machine (and again after changing Python or Visual Studio). It records:
      - the Python used by the Tests and App for their matplotlib-cpp plots
        (installation root, numpy include directory, import library name);
      - an MSVC pin (VCToolsVersion) when needed. CUDA 13.3's cudafe++ crashes
        (0xC0000005) with MSVC 14.50 and newer, so the newest installed MSVC
        older than 14.50 is pinned.
    It also checks that CUDA and vcpkg (VCPKG_ROOT) can be found.

.PARAMETER Python
    python.exe to use. Default: CUSH_PYTHON if set, otherwise python on PATH.
    It needs numpy (and matplotlib to run the plots).

.PARAMETER VCToolsVersion
    Pin this MSVC version instead of choosing one. Pass "none" to never pin.

.EXAMPLE
    ./scripts/configure.ps1
    ./scripts/configure.ps1 -Python C:\Users\me\anaconda3\python.exe
#>
param(
    [string]$Python = "",
    [string]$VCToolsVersion = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$solutionDir = Join-Path $repoRoot "CuSuperHelium"
$userProps = Join-Path $solutionDir "Directory.Build.user.props"
$cudaVersion = "13.3"
$warnings = @()

# ---------------------------------------------------------------- Python
if (-not $Python) { $Python = $env:CUSH_PYTHON }
if (-not $Python) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) { throw "Python not found. Pass -Python C:\path\to\python.exe or set CUSH_PYTHON." }
    $Python = $cmd.Source
}
if (-not (Test-Path $Python)) { throw "Python not found at $Python" }

# No double quotes in the code: Windows PowerShell 5.1 strips them when passing arguments to native programs.
$query = 'import sys, numpy; print(sys.base_prefix); print(numpy.get_include()); print(''python%d%d'' % sys.version_info[:2])'
$ErrorActionPreference = "Continue"
$lines = & $Python -c $query 2>&1 | ForEach-Object { "$_" }
$ErrorActionPreference = "Stop"
if ($LASTEXITCODE -ne 0) { throw "Could not query $Python (is numpy installed?):`n$($lines -join "`n")" }
$pythonRoot, $numpyInclude, $pythonLib = $lines[0].Trim(), $lines[1].Trim(), $lines[2].Trim()

if (-not (Test-Path (Join-Path $pythonRoot "include\Python.h"))) { throw "Python.h not found under $pythonRoot\include" }
if (-not (Test-Path (Join-Path $pythonRoot "libs\$pythonLib.lib"))) { throw "$pythonLib.lib not found under $pythonRoot\libs" }
if (-not (Test-Path (Join-Path $numpyInclude "numpy\arrayobject.h"))) { throw "numpy headers not found under $numpyInclude" }

# ---------------------------------------------------------------- CUDA
$cudaVar = "CUDA_PATH_V" + $cudaVersion.Replace(".", "_")
$cudaRoot = [Environment]::GetEnvironmentVariable($cudaVar)
if (-not $cudaRoot) { $cudaRoot = $env:CUDA_PATH }
if (-not $cudaRoot -or -not (Test-Path (Join-Path $cudaRoot "bin\nvcc.exe"))) {
    $warnings += "CUDA $cudaVersion not found ($cudaVar and CUDA_PATH are not set). Install the CUDA $cudaVersion toolkit."
}
elseif (-not (Test-Path (Join-Path $cudaRoot "extras\visual_studio_integration\MSBuildExtensions\CUDA $cudaVersion.props"))) {
    $warnings += "CUDA's MSBuild integration is missing under $cudaRoot\extras\visual_studio_integration."
}

# ---------------------------------------------------------------- vcpkg
if (-not $env:VCPKG_ROOT -or -not (Test-Path (Join-Path $env:VCPKG_ROOT "scripts\buildsystems\msbuild\vcpkg.props"))) {
    $warnings += "VCPKG_ROOT is not set to a vcpkg checkout. Clone https://github.com/microsoft/vcpkg, run bootstrap-vcpkg.bat, set VCPKG_ROOT, and open a new PowerShell."
}
else {
    # Manifest mode reads the versions at vcpkg.json's builtin-baseline from vcpkg's git history, so VCPKG_ROOT must
    # be a git clone (not a downloaded zip or a copy) that contains that commit.
    $baseline = (Get-Content (Join-Path $solutionDir "vcpkg.json") -Raw | ConvertFrom-Json)."builtin-baseline"
    if (-not (Test-Path (Join-Path $env:VCPKG_ROOT ".git"))) {
        $warnings += "VCPKG_ROOT ($env:VCPKG_ROOT) is not a git clone, so vcpkg cannot read the baseline $baseline and the first build fails. Clone https://github.com/microsoft/vcpkg (git clone, not a zip), run bootstrap-vcpkg.bat, and point VCPKG_ROOT at it."
    }
    elseif (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        $warnings += "git is not on PATH; vcpkg needs it to read the baseline $baseline."
    }
    else {
        $ErrorActionPreference = "Continue"
        & git -C $env:VCPKG_ROOT cat-file -e "$baseline^{commit}" 2>$null
        $found = ($LASTEXITCODE -eq 0)
        $ErrorActionPreference = "Stop"
        if (-not $found) {
            $warnings += "The vcpkg clone at $env:VCPKG_ROOT does not contain the baseline commit $baseline from vcpkg.json. Run: git -C `"$env:VCPKG_ROOT`" fetch origin"
        }
    }
}

# ---------------------------------------------------------------- MSVC pin
# The projects use the v145 toolset, which only Visual Studio 18 provides. Another Visual Studio (2022) must not be
# picked even when it has the C++ tools: its default MSVC says nothing about the one v145 will use.
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
function Get-VsInstall([switch]$AnyWorkload) {
    if (-not (Test-Path $vswhere)) { return $null }
    $required = if ($AnyWorkload) { @() } else { @("-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64") }
    return & $vswhere -version "[18.0,19.0)" -latest -prerelease -products * @required -property installationPath | Select-Object -First 1
}

$pin = $null
if ($VCToolsVersion -eq "none") {
    $pin = $null
}
elseif ($VCToolsVersion) {
    $pin = $VCToolsVersion
}
else {
    $vs = Get-VsInstall
    if (-not $vs) {
        if (Get-VsInstall -AnyWorkload) {
            $warnings += "Visual Studio 18 is installed without the C++ tools. Add the 'Desktop development with C++' workload and the 'MSVC v14.44' build tools in the Visual Studio Installer, then run this script again."
        }
        else {
            $warnings += "Visual Studio 18 with the C++ tools was not found (vswhere). Install Visual Studio 18 with 'Desktop development with C++' and the 'MSVC v14.44' build tools."
        }
    }
    else {
        Write-Host "Visual Studio: $vs"
        # v145 projects use the v145 default when it exists, otherwise the general default.
        $defaultFile = @("Microsoft.VCToolsVersion.v145.default.txt", "Microsoft.VCToolsVersion.default.txt") |
            ForEach-Object { Join-Path $vs "VC\Auxiliary\Build\$_" } | Where-Object { Test-Path $_ } | Select-Object -First 1
        $default = if ($defaultFile) { (Get-Content $defaultFile -Raw).Trim() } else { "" }
        if (-not $default) {
            $warnings += "Could not read the default MSVC version of $vs (no Microsoft.VCToolsVersion*.default.txt). Check the C++ workload, or pass -VCToolsVersion."
        }
        $installed = Get-ChildItem (Join-Path $vs "VC\Tools\MSVC") -Directory -ErrorAction SilentlyContinue |
            ForEach-Object { $_.Name } | Where-Object { $_ -match '^\d+\.\d+\.\d+$' } | Sort-Object { [version]$_ }
        if ($default -and [version]$default -ge [version]"14.50") {
            $pin = $installed | Where-Object { [version]$_ -lt [version]"14.50" } | Select-Object -Last 1
            if (-not $pin) {
                $warnings += "MSVC $default is the default and no MSVC older than 14.50 is installed. CUDA $cudaVersion's cudafe++ crashes with it: install the 'MSVC v14.44' build tools from the Visual Studio Installer and run this script again."
            }
        }
    }
}

# ---------------------------------------------------------------- write
$pinXml = if ($pin) {
    "    <!-- CUDA $cudaVersion's cudafe++ crashes with MSVC 14.50+; pin an older host compiler. -->`r`n    <VCToolsVersion>$pin</VCToolsVersion>`r`n"
} else { "" }
$content = @"
<Project>
  <!-- Written by scripts/configure.ps1 on $(Get-Date -Format yyyy-MM-dd). Machine-specific; not tracked by git. -->
  <PropertyGroup>
$pinXml    <CushPythonRoot>$pythonRoot</CushPythonRoot>
    <CushNumpyInclude>$numpyInclude</CushNumpyInclude>
    <CushPythonLib>$pythonLib</CushPythonLib>
  </PropertyGroup>
</Project>
"@
Set-Content -Path $userProps -Value $content -Encoding utf8

Write-Host "Wrote $userProps"
Write-Host "  Python        $Python ($pythonLib)"
Write-Host "  numpy include $numpyInclude"
Write-Host ("  MSVC pin      " + $(if ($pin) { $pin } else { "none" }))
foreach ($w in $warnings) { Write-Host "Warning: $w" -ForegroundColor Yellow }
if ($warnings.Count -eq 0) { Write-Host "Ready: run ./scripts/build.ps1" }
