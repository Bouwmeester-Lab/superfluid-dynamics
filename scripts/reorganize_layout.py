"""Phase 2 of docs/reorganization-plan.md: move the DLL sources into the layered layout.

Mechanical and reproducible. For every file in MOVES it
  1. runs `git mv`,
  2. rewrites every `#include "..."` / `#include <...>` that points at a moved project header,
     in all three projects, to a path relative to the DLL project directory (the single include
     root), e.g. `#include "Core/CudaChecks.cuh"`; files moved into the App include each other by
     bare name,
  3. updates the item paths in the three .vcxproj files and regenerates the DLL's .filters file
     from the folder structure.

Run from anywhere: `python scripts/reorganize_layout.py`. It refuses to run on a dirty tree.
"""
from __future__ import annotations

import os
import posixpath
import re
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SLN = os.path.join(ROOT, "CuSuperHelium")
LIB = os.path.join(SLN, "CuSuperHelium")
APP = os.path.join(SLN, "CuSuperHelium.App")
TESTS = os.path.join(SLN, "CuSuperHelium.Tests")

# old path (relative to the DLL project) -> new path (relative to the DLL project).
# Paths starting with "../CuSuperHelium.App/" leave the DLL project.
TARGETS = {
    "Core": [
        "Common/CudaChecks.cuh", "Common/VectorKernels.cuh", "Common/DeviceBuffer.cuh", "constants.cuh",
        "cuDoubleComplexOperators.cuh", "PrecisionMath.cuh", "Math/ComplexMath.cuh", "Reductions.cuh",
        "VectorUtilities.cuh", "DevicePointers.cuh",
    ],
    "LinearAlgebra": ["MatrixSolver.cuh", "Common/DenseLuSolver.cuh", "LinearAlgebra.cuh"],
    "Math": ["Math/FftKernels.cuh", "Derivatives.cuh", "DerivativeCalculator.cuh", "EllipticIntegrals.cuh",
             "GLCoefficients.hpp"],
    "Ode": ["OdeSolver.h", "RK4_Time_Dependent.cuh", "AutonomousRungeKuttaStepper.cuh", "RK45.cuh",
            "RK45_Kernels.cuh", "GaussLegendre.cuh", "SimpleEuler.cuh", "JacobianCalculator.cuh", "RK4Options.h",
            "AutonomousProblem.h"],
    "BoundaryIntegral/Periodic2D": [
        "BoundaryProblem.cuh", "BaseBoundaryIntegrator.cuh", "TimedBoundaryIntegrator.cuh",
        "TimedBoundaryProblem.cuh", "RealBoundaryIntegralCalculator.cuh", "WaterVelocities.cuh", "Energies.cuh",
        "BoundaryKernels/MatrixKernels.cuh", "BoundaryKernels/PhiRhsKernels.cuh",
    ],
    "BoundaryIntegral/Axisymmetric": [
        "Radial/BoundaryNodes.cuh", "Radial/SelfTerms.cuh", "RadialModels.h", "RadialVelocityCalculator.cuh",
        "BoundaryKernels/RadialMatrixKernels.cuh",
    ],
    "BoundaryIntegral/Axisymmetric/GreenFunctions": ["Math/BesselGreenFunctions.cuh", "EllipticGreenFunctions.cuh"],
    "Physics": ["HeliumBoundaryProblem.cuh", "WaterBoundaryProblem.cuh", "ProblemProperties.hpp",
                "SimulationOptions.h"],
    "Optomechanics": [
        "LightIntensity.cuh", "DelayedIntensityIntegrator.cuh", "DelayedIntensityTerm.cuh",
        "OptomechanicalVariables.h", "HeliumDrivenAutonomousProblem.cuh",
        "HeliumRampedIntensityAutonomousProblem.cuh", "HeliumSineAutonomousProblem.cuh",
        "HeliumWithDrivingBoundaryProblem.cuh", "AugmentedBoundaryIntegrator.cuh",
        "AugmentedIntensityBoundaryIntegrator.cuh", "GeneralAugmentedBoundaryIntegrator.cuh",
    ],
    "IO": ["DataLogger.cuh", "ValueLogger.h", "TrajectoryLogger.cuh", "StateFile.h"],
    "Export": ["ExportTypes.cuh", "DeviceParticleLoading.cuh"],
    "../CuSuperHelium.App": ["SimulationRunner.cuh", "SimulationFunctions.cuh", "VideoMaking.h", "SolitonPeak.h"],
}
MOVES = {old: posixpath.join(d, posixpath.basename(old)) for d, olds in TARGETS.items() for old in olds
         if posixpath.join(d, posixpath.basename(old)) != old}
APP_PREFIX = "../CuSuperHelium.App/"

SOURCE_EXT = (".cu", ".cuh", ".h", ".hpp", ".cpp")
INCLUDE_RE = re.compile(r'^(?P<pre>[ \t]*#[ \t]*include[ \t]*)(?P<open>["<])(?P<path>[^">]+)(?P<close>[">])', re.M)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", ROOT, *args], check=True, capture_output=True, text=True).stdout


def read(path: str) -> tuple[str, bool]:
    """Byte-preserving: the sources mix UTF-8 (with and without BOM) and cp1252, and only the ASCII
    #include lines and project-file paths are edited, so decode as latin-1 and keep any BOM bytes
    inside the text."""
    return open(path, "rb").read().decode("latin-1"), False


def write(path: str, text: str, bom: bool) -> None:
    # Editors and virus scanners can hold a file open for a moment on Windows; retry briefly.
    for attempt in range(20):
        try:
            with open(path, "wb") as f:
                f.write(text.encode("latin-1"))
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.25)


def lib_rel(abs_path: str) -> str:
    return os.path.relpath(abs_path, LIB).replace(os.sep, "/")


def source_files() -> list[str]:
    files = []
    for top in (LIB, APP, TESTS):
        for d, dirs, names in os.walk(top):
            dirs[:] = [x for x in dirs if x not in ("x64", "indicators", ".vs")]
            files += [os.path.join(d, n) for n in names if n.endswith(SOURCE_EXT)]
    return files


def resolve(including_old: str, inc: str) -> str | None:
    """Old DLL-relative path of the project header that `inc` refers to, or None for external headers."""
    inc = inc.replace("\\", "/")
    base = posixpath.dirname(including_old) if including_old is not None else None
    candidates = []
    if base is not None:
        candidates.append(posixpath.normpath(posixpath.join(base, inc)))
    candidates.append(posixpath.normpath(inc))  # include root = DLL project dir
    for c in candidates:
        if c in MOVES or os.path.isfile(os.path.join(LIB, c)):
            return c
    return None


def main() -> int:
    if git("status", "--porcelain", "--untracked-files=no").strip():
        print("working tree has changes; commit or stash first")
        return 1

    # 1. Rewrite includes (before moving, so every old path still resolves on disk).
    changed = 0
    for path in source_files():
        old_rel = lib_rel(path)
        in_lib = not old_rel.startswith("..")
        own_new = MOVES.get(old_rel, old_rel) if in_lib else old_rel
        text, bom = read(path)

        def fix(m: re.Match) -> str:
            target_old = resolve(old_rel if in_lib else None, m.group("path"))
            if target_old is None:
                return m.group(0)
            target_new = MOVES.get(target_old, target_old)
            if target_new.startswith(APP_PREFIX):
                own_dir_new = posixpath.dirname(own_new) if in_lib else old_rel.rsplit("/", 1)[0]
                if not (own_new.startswith(APP_PREFIX) or own_dir_new == "../CuSuperHelium.App"):
                    raise SystemExit(f"{old_rel} includes App-only header {target_old}")
                new_inc = posixpath.basename(target_new)
            else:
                new_inc = target_new
            return f'{m.group("pre")}"{new_inc}"'

        new_text = INCLUDE_RE.sub(fix, text)
        if new_text != text:
            write(path, new_text, bom)
            changed += 1
    print(f"rewrote includes in {changed} files")

    # 2. git mv.
    for old, new in MOVES.items():
        src, dst = os.path.join(LIB, old), os.path.normpath(os.path.join(LIB, new))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        git("mv", src, dst)
    for d in ("Common", "Radial", "BoundaryKernels"):
        p = os.path.join(LIB, d)
        if os.path.isdir(p) and not os.listdir(p):
            os.rmdir(p)
    print(f"moved {len(MOVES)} files")

    # 3. Project files.
    win = lambda p: p.replace("/", "\\")
    proj = os.path.join(LIB, "CuSuperHelium.vcxproj")
    text, bom = read(proj)
    nl = "\r\n" if "\r\n" in text else "\n"
    app_items = []
    for old, new in MOVES.items():
        pat = re.compile(r'([ \t]*)<(ClInclude|CudaCompile|ClCompile|None) Include="' + re.escape(win(old)) + r'"( />|>.*?</\2>)\r?\n', re.S)
        m = pat.search(text)
        if m is None:
            print(f"  note: {old} not listed in the DLL project")
            continue
        if new.startswith(APP_PREFIX):
            text = text[:m.start()] + text[m.end():]
            app_items.append(posixpath.basename(new))
        else:
            text = text[:m.start()] + m.group(0).replace(f'Include="{win(old)}"', f'Include="{win(new)}"') + text[m.end():]
    write(proj, text, bom)

    app_proj = os.path.join(APP, "CuSuperHelium.App.vcxproj")
    text, bom = read(app_proj)
    anl = "\r\n" if "\r\n" in text else "\n"
    group = anl.join(["  <ItemGroup>"] + [f'    <ClInclude Include="{n}" />' for n in app_items] + ["  </ItemGroup>"]) + anl
    i = text.rindex("  <Import Project=\"$(VCTargetsPath)\\Microsoft.Cpp.targets\" />")
    write(app_proj, text[:i] + group + text[i:], bom)

    # Regenerate the DLL .filters from the folder of each item.
    text, _ = read(proj)
    items = re.findall(r'<(ClInclude|CudaCompile|ClCompile|None|Text) Include="([^"]+)"', text)
    folders = sorted({posixpath.dirname(p.replace("\\", "/")) for _, p in items} - {""})
    all_folders = sorted({"/".join(f.split("/")[:i]) for f in folders for i in range(1, f.count("/") + 2)})
    out = ['<?xml version="1.0" encoding="utf-8"?>',
           '<Project ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">',
           "  <ItemGroup>"]
    for f in all_folders:
        out += [f'    <Filter Include="{win(f)}">', f"    </Filter>"]
    out.append("  </ItemGroup>")
    for kind in ("ClInclude", "CudaCompile", "ClCompile", "None", "Text"):
        kind_items = sorted(p for k, p in items if k == kind)
        if not kind_items:
            continue
        out.append("  <ItemGroup>")
        for p in kind_items:
            folder = posixpath.dirname(p.replace("\\", "/"))
            if folder:
                out += [f'    <{kind} Include="{p}">', f"      <Filter>{win(folder)}</Filter>", f"    </{kind}>"]
            else:
                out.append(f'    <{kind} Include="{p}" />')
        out.append("  </ItemGroup>")
    out.append("</Project>")
    write(os.path.join(LIB, "CuSuperHelium.vcxproj.filters"), "\xef\xbb\xbf" + nl.join(out) + nl, False)
    print("updated project files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
