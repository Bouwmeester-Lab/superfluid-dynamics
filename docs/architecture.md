# CuSuperHelium architecture

CuSuperHelium is a CUDA library (`CuSuperHelium.dll`) that computes the dynamics of thin superfluid helium films and gravity water waves with boundary-integral methods. Python drives it through a C ABI. A console App and a gtest suite use the same sources.

```
CuSuperHelium/                       Visual Studio solution
  CuSuperHelium/                     the DLL; this directory is the only include root
  CuSuperHelium.App/                 simulation runner, videos, matplotlib plots
  CuSuperHelium.Tests/               gtest suite
  Python/                            the cusuperhelium package (pip install -e), analysis/, notebooks/, tests/ (pytest)
  Plotting/                          plotting support shared by Tests and App (PythonInclude.h)
  vcpkg.json, vcpkg-ports/           dependencies (manifest mode) and the patched matplotlib-cpp port
  Directory.Build.props/.targets     settings shared by the three projects
scripts/                             configure.ps1, build.ps1, test.ps1, one-off maintenance scripts
docs/                                this file, reorganization-plan.md
```

Includes are always written from the include root, for example `#include "Core/CudaChecks.cuh"`, never with `../`.

## Layers

Each folder of the DLL is a layer. A layer may include only the layers below it. On the middle row, `Ode` may also use `Math` and `LinearAlgebra` (Gauss-Legendre coefficients, FFT kernels, the LU solver and cuBLAS wrappers). `Math` and `LinearAlgebra` do not include each other or `Ode`.

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
| `LinearAlgebra/` | `DenseLuSolver` (runtime size, single or batched LU), type-generic cuBLAS wrappers (`Blas.cuh`) and other linear-algebra kernels |
| `Math/` | FFT kernels, spectral and finite-difference derivatives, elliptic integrals, Gauss-Legendre coefficients |
| `Ode/` | Time integrators behind `OdeSolver<T>` (`RungeKutta4`, RK45, Gauss-Legendre), the `OdeProblem<T>` interface they all integrate (`AutonomousProblem` for time-independent problems), Jacobians, solver options |
| `BoundaryIntegral/Periodic2D/` | Roberts-method solver for a periodic 2D interface: boundary problem, integral calculators, matrix and potential kernels, velocities, energies |
| `BoundaryIntegral/Axisymmetric/` | Axisymmetric (radial) solver: `BoundaryNodes`, self terms, matrix assembly, the velocity calculator. `GreenFunctions/` holds the Bessel-series kernel and the elliptic ring kernel (option A) |
| `Physics/` | Helium-film and water problem definitions; `PhysicalParameters` (SI inputs, including the caller's material), `Scales`, `DimensionlessParameters`, and `NonDimensionalization`, the one place SI inputs become dimensionless; simulation options |
| `Optomechanics/` | Light intensity, delayed-intensity terms and integrators, driven and augmented problems |
| `IO/` | `StepObserver` (what a solver reports after each step), `TrajectoryLogger`, `ProgressBarObserver`, `DataLogger`, `ValueLogger`, HDF5 state files |
| `Export/` | The C ABI and its helpers: entry points, ABI types, host-to-device loading. Every export takes its sizes (`N`, batch) as arguments |
| (vcpkg) | Third-party libraries come from `CuSuperHelium/vcpkg.json`: boost-math, gtest, hdf5, highfive, indicators, matplotlib-cpp, opencv4 |

## Layering status

No include edge breaks the rules above. There were 23 violations after the Phase 2 move. Phase 3 removed 8, and Phase 4 removed the rest by fixing the code or settling the rule:

**Rules settled in Phase 4:**
- `Ode` sees `Math`, `LinearAlgebra` and `IO`. Observers (`IO/StepObserver.h`) live in `IO`, so loggers do not depend on `Ode`.
- `BoundaryIntegral` sees `Ode`. Its calculators are `OdeProblem`s (`Ode/OdeProblem.h`, through `Ode/AutonomousProblem.h`). No interface moved to `Core/`.

**Fixed in Phase 3 (parameters):**
- No file below `Physics/` includes a parameter struct any more.
  - The periodic-2D solver defines what it reads in `BoundaryIntegral/Periodic2D/InterfaceParameters.h`, and `Physics/DimensionlessParameters` extends it.
  - `Math/Derivatives` takes ρ and U as numbers.
  - `IO/StateFile` takes an attribute map.
  - The `Ode/` steppers lost an include they did not use.

**Fixed in Phase 4 (duplication):**
- The RK4 stepper that included the `BoundaryIntegral/Periodic2D/` integrators is gone. `Ode/RungeKutta4.cuh` depends only on `OdeProblem` and `OdeSolver`.
- `Ode/GaussLegendre.cuh` no longer includes `Export/ExportTypes.cuh`. The conversion from the C ABI options lives in `Export/IntegrationExports.cu`.
- `BoundaryIntegral/Periodic2D/RealBoundaryIntegralCalculator.cuh` dropped its unused includes of `Optomechanics/DelayedIntensityTerm.cuh` and `TimedBoundaryIntegrator.cuh`.

## Time integration

Every integrator implements `OdeSolver<T>` and integrates an `OdeProblem<T>` (dy/dt = f(y, t)), so a problem can be run with any solver that supports its state type:

```cpp
HeliumBoundaryProblem heliumProblem(properties, N);
BaseBoundaryIntegralCalculator problem(properties, heliumProblem);          // an OdeProblem<std_complex>
RungeKutta4<std_complex> solver(problem, dt);
auto trajectory = std::make_shared<TrajectoryLogger<std_complex>>();
solver.addObserver(trajectory);                                             // what is recorded is the caller's choice
solver.initialize(devState, true);                                          // copied: devState is not modified
solver.runEvolution(t0, t1);                                                // the result is solver.state()
```

- `rhs(y, t, dydt)` is a pure function of the state and the time. Solvers evaluate it at trial stages, reject steps and iterate (Newton), so a problem must not keep anything from one call to the next. Whatever evolves belongs in the state: the optomechanical delayed intensity D is part of the augmented state, with `dD/dt = (I - D) / tau`.
- Options stay specific to each scheme (`RK4Options`, `RK45_Options`, `GaussLegendre2Options`).
- Gauss-Legendre is implicit and needs a `JacobianProvider`. `FiniteDifferenceJacobian` works for any `OdeProblem<double>`, and `JacobianCalculator` is the fast batched Jacobian of the periodic boundary integral. `ComplexAsRealProblem` presents a `std_complex` problem as a `double` one.
- Output goes through `StepObserver<T>`: `TrajectoryLogger` and `ProgressBarObserver` are the observers provided.
- `CuSuperHelium.Tests/DelayedIntensityTests.cuh` runs one problem through RK4, RK45 and Gauss-Legendre against an exact solution. It is the check that a problem written against `OdeProblem` works with every solver.
