# superfluid-dynamics
A repo aimed at simulating the dynamics of fluids, in particular superfluid helium using boundary integral methods.

This repo will use CUDA to accelarate solving the dynamics of superfluid helium using boundary integral methods.

The repo will use the method developped by [Roberts (1983)](https://doi.org/10.1093/imamat/31.1.13) to solve the dynamics of superfluid helium.

First we will implement the method as if it was related to fluids subject to gravity as originally done in the paper, then we will adapt it to the case of superfluid helium.

CUDA will be used throughout the repo to accelerate the computations and hopefully make it possible to simulate many particles in a more reasonable time.

## 2D cartesian flow

The method used here is mainly applicable to 2D Cartesian flow. This is not yet generalized to 3D radially symmetric flow.

## Building

The solution `CuSuperHelium/CuSuperHelium.sln` builds three projects: the library `CuSuperHelium.dll` (used from Python), the gtest suite `CuSuperHelium.Tests`, and the console App `CuSuperHelium.App`. Only Windows x64 is supported. The layout is described in [docs/architecture.md](docs/architecture.md).

### Requirements

- **Visual Studio 18** (2026) with the *Desktop development with C++* workload, which provides the v145 toolset.
  - Also install the **MSVC v14.44** build tools (Visual Studio Installer, *Individual components*). CUDA 13.3's front end (`cudafe++`) crashes with MSVC 14.50 and newer, so the build pins 14.44. `scripts/configure.ps1` does this automatically.
- **CUDA Toolkit 13.3.** The installer sets `CUDA_PATH_V13_3`, which the build uses. The GPU code is built for sm_86 (RTX 30xx) and sm_120 (RTX 50xx).
- **vcpkg.** Clone [microsoft/vcpkg](https://github.com/microsoft/vcpkg), run `bootstrap-vcpkg.bat`, and set the `VCPKG_ROOT` environment variable to the clone.
  - The dependencies are listed in `CuSuperHelium/vcpkg.json` and installed automatically into `CuSuperHelium/vcpkg_installed/` on the first build. That first build takes a while, mostly for boost, hdf5 and opencv.
  - `vcpkg integrate install` is not needed.
  - Keep the clone path short (for example `D:\repos\superfluid-dynamics`), or enable Windows long paths. Some boost header paths are long, and vcpkg fails when a full path exceeds 260 characters.
- **Python 3** with numpy, matplotlib and pytest (Anaconda works). The Tests and App draw plots with matplotlib-cpp, and the Python tests drive the DLL.

### Steps

From the repository root, in PowerShell:

```powershell
./scripts/configure.ps1 -Python C:\path\to\python.exe   # once per machine: writes CuSuperHelium/Directory.Build.user.props
./scripts/build.ps1                                     # Release; add -Configuration Debug for Debug
./scripts/test.ps1                                      # gtest, then the Python tests (CUSH_PYTHON or python on PATH)
```

- `configure.ps1` records the Python installation, the numpy include directory and the MSVC pin in a git-ignored file. Run it again after changing Python or Visual Studio.
- After editing headers, use `./scripts/build.ps1 -Rebuild` or `./scripts/test.ps1 -Rebuild`. nvcc's dependency tracking misses some included `.cuh` files.
- The DLL is written to `CuSuperHelium/x64/<Configuration>/CuSuperHelium.dll`. The Python bindings are in `CuSuperHelium/Python/integration/rhs.py`.

