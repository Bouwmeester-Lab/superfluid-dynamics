# CuSuperHelium architecture

CuSuperHelium is a CUDA library (`CuSuperHelium.dll`) that computes the dynamics of thin superfluid helium films and gravity water waves with boundary-integral methods. Python drives it through a C ABI. A console App and a gtest suite use the same sources.

```
CuSuperHelium/                       Visual Studio solution
  CuSuperHelium/                     the DLL; this directory is the only include root
  CuSuperHelium.App/                 simulation runner, videos, matplotlib plots
  CuSuperHelium.Tests/               gtest suite
  Python/                            ctypes bindings, analysis scripts, notebooks, pytest
scripts/                             build.ps1, test.ps1, one-off maintenance scripts
docs/                                this file, reorganization-plan.md
```

Includes are always written from the include root, for example `#include "Core/CudaChecks.cuh"`, never with `../`.

## Layers

Each folder of the DLL is a layer. A layer may include only the layers below it.

```
                 Export            C ABI: ExportApi.cuh, ExportTypes.cuh, *Exports.cu
                    |
            Optomechanics          light intensity, delayed-intensity terms, driven problems
                    |
                 Physics           problem definitions and physical parameters
                    |
            BoundaryIntegral       Periodic2D (Roberts method), Axisymmetric (radial)
              /     |     \
           Ode    Math   LinearAlgebra
             \      |      /
                   IO              loggers, HDF5 state files
                    |
                  Core             CUDA checks, device buffers, constants, complex math
```

Plotting lives only in `CuSuperHelium.App` and `CuSuperHelium.Tests`, never in the DLL.

| Folder | Purpose |
|---|---|
| `Core/` | CUDA basics with no physics: error checks, `DeviceBuffer<T>`, constants, complex operators, double-double arithmetic, vector kernels and reductions |
| `LinearAlgebra/` | Dense LU solvers (`MatrixSolver<N, batch>`, `DenseLuSolver`) and other linear-algebra kernels |
| `Math/` | FFT kernels, spectral and finite-difference derivatives, elliptic integrals, Gauss-Legendre coefficients |
| `Ode/` | Time integrators behind `OdeSolver` (RK4, RK45, Gauss-Legendre, Euler), the `AutonomousProblem` interface, Jacobians, solver options |
| `BoundaryIntegral/Periodic2D/` | Roberts-method solver for a periodic 2D interface: boundary problem, integral calculators, matrix and potential kernels, velocities, energies |
| `BoundaryIntegral/Axisymmetric/` | Axisymmetric (radial) solver: `BoundaryNodes`, self terms, matrix assembly, the velocity calculator. `GreenFunctions/` holds the Bessel-series kernel and the elliptic ring kernel (option A) |
| `Physics/` | Helium-film and water problem definitions; `PhysicalParameters` (SI inputs, including the caller's material), `Scales`, `DimensionlessParameters`, and `NonDimensionalization`, the one place SI inputs become dimensionless; simulation options |
| `Optomechanics/` | Light intensity, delayed-intensity terms and integrators, driven and augmented problems |
| `IO/` | `DataLogger`, `ValueLogger`, `TrajectoryLogger`, HDF5 state files |
| `Export/` | The C ABI and its helpers: entry points, ABI types, size dispatch, host-to-device loading |
| `indicators/` | Third-party header-only progress bars (vendored) |

## Known layering violations

These are the include edges that break the rules above. There were 23 after the Phase 2 move. Phase 3 removed the 8 caused by the parameter struct, which leaves 15. Each remaining one is assigned to a later phase.

**The rules should change to fit the code:**
- `Ode` uses `Math` (Gauss-Legendre coefficients, FFT kernels) and `IO` (loggers). Both are legitimate dependencies, so the layer order above already puts `IO` below `Ode` and lets `Ode` see `Math`.
- `BoundaryIntegral` implements `Ode/AutonomousProblem.h`. That interface could move to `Core/`, or `BoundaryIntegral` could be allowed to see `Ode`. Decide in Phase 4, when the steppers are unified.

**Fixed in Phase 3 (parameters):**
- No file below `Physics/` includes a parameter struct any more.
  - The periodic-2D solver defines what it reads in `BoundaryIntegral/Periodic2D/InterfaceParameters.h`, and `Physics/DimensionlessParameters` extends it.
  - `Math/Derivatives` takes ρ and U as numbers.
  - `IO/StateFile` takes an attribute map.
  - The `Ode/` steppers lost an include they did not use.

**The code should change (Phase 4, duplication):**
- `Ode/RK4_Time_Dependent.cuh` includes `BoundaryIntegral/Periodic2D/` integrators. The stepper should depend only on the `OdeSolver`/problem interfaces.
- `Ode/GaussLegendre.cuh` includes `Export/ExportTypes.cuh` for `c_double`, an ABI type leaking into the numerics.
- `BoundaryIntegral/Periodic2D/RealBoundaryIntegralCalculator.cuh` includes `Optomechanics/DelayedIntensityTerm.cuh`.
