"""Finding and loading CuSuperHelium.dll.

The DLL is looked up in this order:
1. the path given to `load_library` / `SimulationManager`;
2. the CUSUPERHELIUM_DLL environment variable;
3. the build output of this checkout, CuSuperHelium/x64/<Release|Debug>/CuSuperHelium.dll (Release first).

Since Python 3.8, Windows no longer resolves a DLL's dependencies through PATH. The DLL's own folder (hdf5.dll and
the other vcpkg DLLs are copied there) and CUDA's bin folders (cuBLAS, cuSOLVER, cuFFT) are therefore added to
the search path explicitly.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

from .bindings import bind

# Python/cusuperhelium/library.py -> parents[2] is the solution directory CuSuperHelium/.
_SOLUTION_DIR = Path(__file__).resolve().parents[2]
_CUDA_VERSION = "13.3"
_dll_directories = []  # handles returned by os.add_dll_directory must stay alive


def default_dll_candidates() -> list[Path]:
    return [_SOLUTION_DIR / "x64" / configuration / "CuSuperHelium.dll" for configuration in ("Release", "Debug")]


def find_dll(path: str | os.PathLike | None = None) -> Path:
    """Returns the DLL to load, or raises FileNotFoundError saying where it looked."""
    if path is not None:
        candidates = [Path(path)]
    elif os.environ.get("CUSUPERHELIUM_DLL"):
        candidates = [Path(os.environ["CUSUPERHELIUM_DLL"])]
    else:
        candidates = default_dll_candidates()
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    looked = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"CuSuperHelium.dll not found (looked at: {looked}). Build it with scripts/build.ps1, "
                            "or set CUSUPERHELIUM_DLL to its path.")


def _cuda_bin_directories() -> list[Path]:
    roots = [os.environ.get("CUDA_PATH_V" + _CUDA_VERSION.replace(".", "_")), os.environ.get("CUDA_PATH")]
    directories = []
    for root in filter(None, roots):
        for sub in ("bin/x64", "bin"):  # CUDA 13 keeps its DLLs in bin\x64
            directory = Path(root) / sub
            if directory.is_dir() and directory not in directories:
                directories.append(directory)
    return directories


def _add_dll_directory(directory: Path) -> None:
    if hasattr(os, "add_dll_directory") and directory.is_dir():
        _dll_directories.append(os.add_dll_directory(str(directory)))


def load_library(path: str | os.PathLike | None = None) -> ctypes.CDLL:
    """Loads CuSuperHelium.dll and sets the signature of every export."""
    if os.name != "nt":
        raise OSError("CuSuperHelium.dll is a Windows library; this package only runs on Windows.")
    dll = find_dll(path)
    _add_dll_directory(dll.parent)
    for directory in _cuda_bin_directories():
        _add_dll_directory(directory)
    return bind(ctypes.CDLL(str(dll)))
