# CuSuperHelium reorganization plan

Status: accepted, decisions recorded in section 6 · Branch it was written on: `radial/boundary-nodes-refactor` · November 2026

## 1. Why

CuSuperHelium has grown from a 2D Roberts-method solver into a library with several solvers, three ODE integrators, optomechanical drives, an axisymmetric solver, a C export layer and Python analysis code. The pieces work and are validated against experiment, but the layout makes them hard to reuse or maintain:

- **Flat root.** About 60 headers sit in `CuSuperHelium/CuSuperHelium/`, mixing CUDA utilities, linear algebra, ODE solvers, boundary-integral calculators, physics problems, optomechanics, logging and plotting. Only `Common/`, `Math/`, `BoundaryKernels/`, `Export/`, `Radial/` and `Optomechanics/` have started to impose structure.
- **Hidden dependencies.** Core numerics headers include `matplotlibcpp.h` (about 150 references), so the DLL links `python312.lib` and every numerical header depends on a Python install. The gtest `main` calls `plt::show()`.
- **Non-portable build.** The `.vcxproj` files hard-code `C:\Users\emore\anaconda3`, `C:\vcpkg\packages\...`, and a **second checkout** (`C:\Users\emore\source\repos\Bouwmeester-Lab\superfluid-dynamics\...`) on the include path, so a header can silently resolve from the wrong copy. Debug builds use C++23 and Release C++20. The project needs the v145 toolset (VS 18); VS 2022 cannot build it.
- **Duplication.** There are two dense solvers (`MatrixSolver<N, batch>`, `Common/DenseLuSolver`), two RK4 steppers (`RK4_Time_Dependent.cuh`, `AutonomousRungeKuttaStepper.cuh`), two derivative families (`Derivatives.cuh` FFT, `DerivativeCalculator.cuh` finite differences) and three parameter structs (`ProblemProperties`, `SimProperties`, `RadialProperties`). Nondimensionalization lives in `Export/ExportProperties.cu`.
- **Compile-time sizes.** Most classes are templated on `N`, so every export carries a `switch` over supported sizes (about 70 `case` labels across `Export/*.cu`). The radial solver has already moved to runtime sizes.
- **Naming drift.**
  - Spelling errors: `Eliptical*` for "Elliptic", and the class `RealBoundaryItegralCalculator`.
  - Generic periodic-2D code is prefixed "Water" (`WaterVelocities.cuh` holds `VelocityCalculator`).
  - File names don't match the classes they hold: `BaseBoundaryIntegrator.cuh` defines `BaseBoundaryIntegralCalculator`.
  - `WaterBoundaryIntegralCalculator.cuh` and `utilities.cuh` are umbrella headers that only include other headers.
- **Dead code.** `PingPongBuffer.cuh` and `createM.cuh` are included nowhere. `BoundaryKernels/RadialKernels.cuh` is an unused early elliptic kernel.
- **Python.** The bindings (`integration/rhs.py`) find the DLL through the current working directory, and scripts insert `sys.path` entries by hand.
- **Binaries in git.** About 400 MB of simulation output is tracked: roughly 30 videos (`.avi` and extension-less `video*` files, 5–28 MB each) under `CuSuperHelium/CuSuperHelium/temp/`, and `.h5` files under `temp/` and `Python/`. They are all still present at `HEAD`. The `.git` folder is 412 MB, almost all of it these files.

The goal is a layout where each layer can be understood, tested and reused on its own, and where a new maintainer can build the project on a fresh machine.

## 2. Principles

1. **Layers with one-way dependencies.** A layer may include only layers below it. Numerics never include physics, physics never includes export or plotting.
2. **Behavior-preserving steps.** Every phase is a separate PR that leaves numerical output unchanged, checked against recorded golden output, as was done for the radial refactor (worst difference 6e-13).
3. **Mechanical moves separate from logic changes.** File moves and renames go in their own commits, so diffs stay reviewable and `git log --follow` keeps history.
4. **Runtime sizes by default.** Template on sizes only where it measurably helps, such as small fixed quadrature orders. The Cartesian solver moves to runtime sizes only after the reorganization is finished (Phase 7).
5. **The C ABI is the only public surface.** Everything behind `Export/` can change freely.
6. **Plots are a verification tool, not a library feature.** The library never plots. Tests may plot, so a person can check results by eye as well as by assertion (see "Visual tests" below).

## 3. Target layout

```
CuSuperHelium/CuSuperHelium/
  Core/              CUDA basics, no physics
                     CudaChecks, DeviceBuffer<T> (new, RAII), constants, complex operators,
                     PrecisionMath (double-double), VectorKernels, Reductions
  LinearAlgebra/     DenseLuSolver (single + batched, runtime size), LinearAlgebra kernels
  Math/              FFT kernels, spectral/finite-difference derivatives, elliptic integrals,
                     Gauss-Legendre coefficients
  Ode/               OdeSolver interface, RK4, RK45, implicit Gauss-Legendre, Euler,
                     JacobianCalculator, solver options
  BoundaryIntegral/
    Periodic2D/      Roberts-method solver: BoundaryProblem, BoundaryIntegralCalculator,
                     VelocityCalculator, matrix/phi kernels, energies
    Axisymmetric/    BoundaryNodes, SelfTerms, RadialMatrixKernels, GreenFunctions/
                     (Bessel series, half-space ring = option A), RadialVelocityCalculator
  Physics/           Problem definitions (gravity water, helium film, finite/infinite depth),
                     PhysicalParameters + nondimensionalization
  Optomechanics/     LightIntensity, delayed-intensity terms, drive kernels,
                     augmented/driven problems and integrators
  IO/                DataLogger, ValueLogger, TrajectoryLogger, HDF5 output
  Export/            C ABI only: ExportApi.cuh, ExportTypes.cuh, *Exports.cu, size dispatch
CuSuperHelium.App/   Simulation runner, video making, matplotlib plots (only place they live)
CuSuperHelium.Tests/ One test file per layer (Core, LinearAlgebra, Ode, Periodic2D, Axisymmetric, Export)
Python/
  cusuperhelium/     Installable package: _bindings.py (ctypes), dll discovery, helpers
  analysis/          Scripts (phase diagrams, radial checks, thin film)
  notebooks/
  tests/             pytest: golden regression, mode tests
docs/                architecture.md, numerics notes, this plan
```

Allowed dependencies, top to bottom:

```
Export  ->  Optomechanics, Physics, BoundaryIntegral, Ode, IO
Optomechanics -> Physics, BoundaryIntegral, Ode
Physics -> BoundaryIntegral
BoundaryIntegral -> Math, LinearAlgebra, Core
Ode -> LinearAlgebra, Core
Math, LinearAlgebra -> Core
App, Tests -> anything;   plotting -> App and Tests only, never the DLL
```

### Visual tests

Plots in tests are encouraged, but they must never stop an automated run.

- **Separate from assertions.** A visual test still asserts what it can, such as convergence order, symmetry or conservation. The figure is an extra check for a person, not the pass/fail criterion.
- **Save, don't show.** Figures are written as PNGs to `test-output/<suite>/<test>.png` (git-ignored), so they can be compared across runs and attached to a PR.
- **Opt in to windows.** A figure window opens only when `CUSH_SHOW_PLOTS=1` is set. Without it nothing blocks, so `scripts/test.ps1` and future CI run unattended.
- **Where they live.** In C++, a small `Tests/Plotting.h` wrapper is the only test code that includes `matplotlibcpp.h`, and test files call the wrapper. In Python, pytest tests use matplotlib with the `Agg` backend unless `CUSH_SHOW_PLOTS=1`.
- **Naming.** Visual tests are tagged (gtest name prefix `Visual_`, pytest marker `@pytest.mark.visual`) so they can be run on their own, for example `--gtest_filter=*Visual_*`.

Good first candidates are the radial Bessel-mode check, the Nb convergence sweep, and the option A vs Bessel vs option B comparison.

Includes are written relative to the project root (`#include "Core/CudaChecks.cuh"`), with the project directory as the only include root, instead of `../` paths.

### Where the current files go

| Current file(s) | Target |
|---|---|
| `Common/CudaChecks.cuh`, `Common/VectorKernels.cuh`, `constants.cuh`, `cuDoubleComplexOperators.cuh`, `PrecisionMath.cuh`, `Math/ComplexMath.cuh`, `Reductions.cuh`, `VectorUtilities.cuh`, `DevicePointers.cuh` | `Core/` |
| `MatrixSolver.cuh`, `Common/DenseLuSolver.cuh`, `LinearAlgebra.cuh` | `LinearAlgebra/` (merged, see Phase 4) |
| `Math/FftKernels.cuh`, `Derivatives.cuh`, `DerivativeCalculator.cuh`, `EllipticIntegrals.cuh`, `GLCoefficients.hpp` | `Math/` |
| `OdeSolver.h`, `RK4_Time_Dependent.cuh`, `AutonomousRungeKuttaStepper.cuh`, `RK45.cuh`, `RK45_Kernels.cuh`, `GaussLegendre.cuh`, `SimpleEuler.cuh`, `JacobianCalculator.cuh`, `RK4Options.h`, `AutonomousProblem.h` | `Ode/` |
| `BoundaryProblem.cuh`, `BaseBoundaryIntegrator.cuh`, `TimedBoundaryIntegrator.cuh`, `TimedBoundaryProblem.cuh`, `RealBoundaryIntegralCalculator.cuh`, `WaterVelocities.cuh`, `Energies.cuh`, `BoundaryKernels/MatrixKernels.cuh`, `BoundaryKernels/PhiRhsKernels.cuh` | `BoundaryIntegral/Periodic2D/` |
| `Radial/*`, `RadialModels.h`, `RadialVelocityCalculator.cuh`, `BoundaryKernels/RadialMatrixKernels.cuh`, `Math/BesselGreenFunctions.cuh`, `EllipticGreenFunctions.cuh` | `BoundaryIntegral/Axisymmetric/` |
| `HeliumBoundaryProblem.cuh`, `WaterBoundaryProblem.cuh`, `ProblemProperties.hpp`, `SimulationOptions.h` | `Physics/` |
| `LightIntensity.cuh`, `DelayedIntensity*.cuh`, `OptomechanicalVariables.h`, `Helium*AutonomousProblem.cuh`, `HeliumWithDrivingBoundaryProblem.cuh`, `Augmented*Integrator.cuh`, `GeneralAugmentedBoundaryIntegrator.cuh`, `Optomechanics/*` | `Optomechanics/` |
| `DataLogger.cuh`, `ValueLogger.h`, `TrajectoryLogger.cuh` | `IO/` |
| `SimulationRunner.cuh`, `SimulationFunctions.cuh`, `VideoMaking.h`, `SolitonPeak.h`, `kernel.cu` | `CuSuperHelium.App/` |
| `PingPongBuffer.cuh`, `createM.cuh`, `BoundaryKernels/RadialKernels.cuh`, umbrella headers `utilities.cuh`, `WaterBoundaryIntegralCalculator.cuh`, `Export.cuh` | delete (after a build confirms no users) |

## 4. Phases

Each phase is one PR. Each must pass the gtest suite and the golden regression from Phase 0 before merging.

### Phase 0 — Safety net (do first)

- Move the golden-regression approach used for the radial refactor into the repo as `Python/tests/test_golden.py`, extended to the Cartesian exports: RHS from vectors, Jacobian, and a short RK4/Gauss-Legendre integration, each at two sizes. Record goldens from `main`.
- Add `scripts/build.ps1` (MSBuild path, Release x64) and `scripts/test.ps1` (gtest + pytest).
- Make the gtest executable non-interactive: `plt::show()` in `main` runs only when `CUSH_SHOW_PLOTS=1` (see "Visual tests").
- **Done when:** one command builds, runs gtest and compares goldens on a clean checkout of this machine.
- **Status: done** on branch `reorg/phase-0-safety-net`.
  - 17 golden cases in `CuSuperHelium/Python/tests/golden/` (about 600 KB). They cover every export that does work: RHS (N and fixed-256, batched), vorticities, FFT derivative, Jacobian, Gauss-Legendre, optomechanical and augmented RK4 (constant and ramped), augmented RHS, and the radial Bessel solver.
  - Run-to-run output on the same GPU is bit-identical. The tolerances (1e-10 relative, 1e-9 for integrations) only leave room for harmless reordering of floating-point operations.
  - `scripts/test.ps1` runs 26 gtests and 17 golden cases in about 15 seconds.
- **Found while doing Phase 0:**
  - Fixed:
    - the empty `integrateSimulationRK4` stub was removed from the ABI;
    - the uninitialized `i` in `RK45.cuh` is fixed;
    - missing return paths in `RK45.cuh` and `AutonomousRungeKuttaStepper.cuh` are fixed (`OdeSolverResult::UnknownError`).
  - Fixed in the tests:
    - `ComplexFunctionsTests` used `x, y = ...` (comma operator), so `x` was never set. The points now lie on the diagonal x = y, as the other tests in that file use. Sine and cosine now pass `EXPECT_DOUBLE_EQ` on real inputs. Cotangent uses a tolerance relative to |cot z|, which reaches about 20 near the pole.
    - `theoreticalMMatrix` throws for entries outside the N = 2 closed form.
  - Not an issue: the `pwsh.exe` post-build call is a fallback and works as intended.
  - **Incremental builds miss header changes.** nvcc's generated `.deps` files leave out some included `.cuh` files (for example `ComplexFunctionsTests.cuh` and `RK45.cuh` for the test project), so MSBuild can keep stale objects after a header-only edit. Until Phase 5 looks at this, run `scripts/test.ps1 -Rebuild` (about 2 minutes) before trusting a golden comparison after editing headers.
  - Fixed after the first full rebuild: the `switch (N)` exports `calculateJacobian`, `calculateRHSFromVectors` and `calculateRHS256FromVectorsBatched` fell off the end for an unsupported N. They now return -1, which `Python/tests/test_export_errors.py` checks. A full rebuild now has no C4700 or C4715 warnings.
  - Deferred (Phase 6): in `rhs.py`, several bindings declare structs by value while the C functions take pointers. This works because the Windows x64 calling convention passes large structs by reference, and the code only runs on Windows for now. Marked `TODO(struct-pointers)` in `rhs.py`.
  - The radial golden output for a curved surface is not converged in N, which is the known Bessel-truncation error. The golden freezes it so refactors stay neutral. Option A will replace these values with intended changes.

### Phase 1 — Hygiene, no moves

- Remove plotting from library headers. Debug plots that are still useful become visual tests or App code; the rest are deleted. Drop `python312.lib` from the DLL link (the Tests and App projects keep it).
- Delete the dead files listed above.
- Fix the spelling errors (`Elliptic*`, `RealBoundaryIntegralCalculator`), with one commit per rename.
- Add `Core/DeviceBuffer.cuh` (RAII device array) and use it in the exports in place of manual `cudaMalloc`/`cudaFree` pairs, which leak on exceptions today.
- Stop tracking generated data: add `*.h5`, `*.avi`, `CuSuperHelium/CuSuperHelium/temp/`, `Python/figures/` and `test-output/` to `.gitignore`, and `git rm --cached` the existing files. The files stay on disk, and none of them need to be kept: they can be regenerated with the solver.
- **Done when:** the DLL builds without Python libraries and goldens match.
- **Status: done** on branch `reorg/phase-1-hygiene`. 29 gtests and 20 pytest tests pass after a full rebuild, and the goldens match.
  - **Plotting.** The DLL now imports only the CUDA libraries, `hdf5.dll` and the CRT. Python and OpenCV are gone from its link line and include paths.
    - The loaders `loadDataToDevice` and `freeDeviceData` moved from `SimulationRunner.cuh` to `DeviceParticleLoading.cuh`, and the HDF5 helpers moved to `StateFile.h`.
    - `SimulationRunner.cuh`, `SimulationFunctions.cuh` and `VideoMaking.h` are now App-only, and Phase 2 moves them to the App.
    - The never-enabled `DEBUG_*` plotting blocks were deleted. None of them was worth keeping as a visual test.
  - **Removed exports:**
    - `dispertionTest256`, which ran the full plotting/video runner, had no caller and no Python binding;
    - `integrateSimulationRK4_freeMemory`, together with the unfinished `integrateSimulationRK4_N` template.
  - **Dead files:**
    - deleted: the files listed above, plus `CuSuperHelium/kernel.cu`, an old copy of the App's `kernel.cu` that no project compiled;
    - each include of an umbrella header was replaced by the headers that umbrella pulled in;
    - kept: `EllipticGreenFunctions.cuh` and `EllipticIntegrals.cuh`, renamed, as the starting point for option A.
  - **`DeviceBuffer`** lives in `Common/` next to `CudaChecks.cuh` for now. Phase 2 moves both to `Core/`.
  - **Found while doing Phase 1:**
    - **Toolchain.** CUDA 13.3's `cudafe++` crashes (0xC0000005) with MSVC 14.52, the version that came with the November VS 18 Insiders update.
      - `CuSuperHelium/Directory.Build.props` imports a git-ignored `Directory.Build.user.props`. On this machine that file pins `VCToolsVersion` to 14.44.35207.
      - Remove the pin once a CUDA release supports 14.52. Phase 5 builds on this hook.
    - **Fixed:**
      - `calculateRhsAugmentedOptomechanical_N` freed uninitialized pointers when it threw early, and returned 0 on failure;
      - `calculateDerivativeFFT256` also returned 0 on failure.
      - Both now return -1.
    - **Open (Phase 3):**
      - `checkCuda(result)`, the single-argument form used throughout the library, calls `exit()`, so a CUDA error kills the Python process. The converted export call sites use `CHECK_CUDA`, which throws, but the library still uses `checkCuda` widely.
      - `loadDataToDevice` returns a raw `DeviceParticleData` that leaks if the export throws before `freeDeviceData`.

### Phase 1b — Remove binaries from history

Done once, right after Phase 1 merges and before Phase 2. Rewriting history changes every commit hash, so it is easiest while few branches are in flight.

1. **Backup.** `git clone --mirror` the GitHub repository to a folder outside the repo, and keep it until the rewrite has been checked.
2. **Inventory.** Run `git filter-repo --analyze` and review the list of large blobs and paths.
3. **Remove by path, never by size.** Use `git filter-repo --invert-paths` with explicit paths and globs: the `temp/` directory, `*.h5` and `*.avi`. A size threshold (`--strip-blobs-bigger-than`) could also remove large source files or notebooks, so it is not used. Notebooks (`.ipynb`) are code and are kept. To keep them small from now on, strip outputs before commit (`nbstripout`).
4. **Verify no code was lost.** For every branch, compare the file list of the old tip (from the mirror) with the new tip. The only files missing may be ones matching the removal patterns. Also check that the number of commits that touch `*.cu`, `*.cuh`, `*.h`, `*.hpp`, `*.py` or `*.ipynb` is unchanged. A script in `scripts/verify_history_rewrite.py` does both checks.
5. **Publish.** Force-push all branches and tags. Every other clone must then be deleted and cloned again, including the second checkout at `C:\Users\emore\source\repos\Bouwmeester-Lab\superfluid-dynamics`. GitHub may keep the old objects reachable through closed PRs until GitHub support runs a garbage collection. That does not matter for a private repository.

- **Done when:** a fresh clone is a few MB, and the check in step 4 passes for every branch.
- **Status: done** on 2026-11-13.
  - **Backup:** a mirror of GitHub from before the rewrite, at `D:\repos\superfluid-dynamics-backup-2026-11-13.git` (208 MB).
  - **Removed:** `*.avi`, `*.h5`, `CuSuperHelium/CuSuperHelium/temp/` and `CuSuperHelium/Python/figures/`, with `git filter-repo --invert-paths`. Packed size went from 208 MB to 4 MB.
  - **Verified:** `scripts/verify_history_rewrite.py` passes for all 17 branches. The script also fails, as it should, when run without the removal patterns.
  - **Published:** all 17 branches were force-pushed, and GitHub's branch tips match the rewritten mirror.
  - **Local checkout:** the checkout at `D:\repos\superfluid-dynamics` was updated in place instead of re-cloned. Each local branch was moved to its rewritten commit through filter-repo's commit map, which keeps the local stashes. The second checkout at `C:\Users\emore\source\repos\...` no longer exists.
  - **The repository is public, not private as this plan assumed.**
    - GitHub keeps the old objects reachable through the `refs/pull/*` refs until GitHub Support runs a garbage collection.
    - Fresh clones only fetch branches, so they are small anyway.
    - Everyone with an older clone must re-clone before pushing, or the binaries come back.

### Phase 2 — Directory layout (mechanical)

- Apply the mapping in section 3 with a script (`git mv` plus include rewriting plus regenerated `.vcxproj` and `.filters` entries), so the PR is reproducible and reviewable.
- Use one include root: the project directory, with no `../` paths.
- Remove the stale second-checkout path from `<IncludePath>`.
- Add `docs/architecture.md` with the layer diagram and a one-line purpose for each folder.
- **Done when:** the build is green, goldens match, and no file is left at the project root except the `.vcxproj` files.
- **Timing:** land this while no large feature branch is open, because it conflicts with everything. In particular, do it before or right after the option A (half-space kernel) work.
- **Status: done** on branch `reorg/phase-2-layout`. After a full rebuild, 29 gtests and 20 pytest tests pass, and the goldens match.
  - **The move.** `scripts/reorganize_layout.py` moved 68 files, and git detects all 68 as renames. The script also rewrote the includes and the project files.
  - **Placement of files the table doesn't list:**
    - `Common/DeviceBuffer.cuh` → `Core/`; `Common/DenseLuSolver.cuh` → `LinearAlgebra/`;
    - `StateFile.h` → `IO/`;
    - `DeviceParticleLoading.cuh` and `ExportTypes.cuh` → `Export/`;
    - the Bessel and elliptic Green functions → `BoundaryIntegral/Axisymmetric/GreenFunctions/`;
    - `indicators/` (third-party) stays at the project root.
  - **Include roots.** Each project now uses the DLL directory as its only include root. The absolute `D:\` path and the stale second-checkout path were removed from all three projects.
  - **`docs/architecture.md`** lists the 23 include edges that still break the layer rules. Each one is assigned to a later phase.

### Phase 3 — Parameters and the export boundary

The per-export `switch (N)` blocks stay as they are. They exist only in `Export/`, and Phase 7 deletes them, so a shared size-dispatch helper would be throwaway work.

**Parameters**
- Split `ProblemProperties` into two parts:
  - `PhysicalParameters`: the SI inputs, which are L, depth, ρ, κ, U and the **Hamaker constant**;
  - `Scales`: the derived characteristic values, which are base length, time, acceleration and energy.
- Put the conversion in one `Physics/NonDimensionalization` module, replacing the `adimensionalize*` functions in `Export/ExportProperties.cu`.
- `SimProperties` (the C ABI struct) converts to these types only in `Export/`, and `RadialProperties` derives from them.
- **The Hamaker constant is an input, not a constant.** It determines the material being simulated, so the caller chooses it.
  - It becomes a field of `SimProperties` and of the Python bindings.
  - `alpha_hamaker_d` is removed from `Core/constants.cuh`.
  - The App's hard-coded `2.6e-24` copies of the nondimensionalization, in `SimulationFunctions.cuh` and `kernel.cu`, call the shared module with an explicit value instead.
  - The goldens pass the value they were recorded with (3.5e-24), so the results don't change.
- Move the three `adimensionalize*` declarations out of the public `Export/ExportApi.cuh`. They are internal C++ functions declared inside `extern "C"`.

**Layering**
- Nothing below `Physics/` includes the parameter structs. Today `ProblemProperties.hpp` is included by eight such files: `Math/Derivatives`, `IO/StateFile`, three `Ode/` steppers and three `Periodic2D/` files. Each will take only the scalars it uses.

**Error checking: one standard, in `Core/CudaChecks.cuh`**
- **`CUSH_CHECK(expr)`** wraps every call that returns a status.
  - It has overloads for `cudaError_t`, `cublasStatus_t`, `cusolverStatus_t` and `cufftResult`. cuFFT results are not checked at all today.
  - All of them throw `cush::CudaError`, derived from `std::runtime_error`. The message gives the library, the status name, the expression text, and the file and line.
  - The `CUSH_` prefix (CuSuperHelium) keeps the macro from colliding with the `CUDA_CHECK`/`CHECK_CUDA` macros that other headers often define. Macros ignore namespaces, so only a prefix prevents a collision.
- **`CUSH_CHECK_NOTHROW(expr)`** is the variant for destructors and other `noexcept` code. It logs instead of throwing.
- **`CUSH_CHECK_LAUNCH()`** goes after every kernel launch.
  - It checks `cudaGetLastError()`.
  - Built with `CUSH_SYNC_CHECKS`, it also calls `cudaDeviceSynchronize()`, so that errors during kernel execution are reported at the launch that caused them. This is for debug builds.
- **Numerical failures** reported through cuSOLVER's `devInfo` (for example a singular matrix in `getrf`) throw the same `cush::CudaError`. `MatrixSolver::checkSolverInfo` and `DenseLuSolver` both switch to it.
- **No `exit()` anywhere in the library.** Every export catches `std::exception` and returns -1.
- **Destructors never throw.** They log errors only, as `DeviceBuffer::reset` does.
- **Remove the old helpers and replace all their call sites:**

  | Old helper | Uses | Behaviour today |
  |---|---|---|
  | `checkCuda(result)` (both forms, about 124 uses in total) | | the one-argument form calls `exit()` |
  | `CHECK_CUDA` | 97 | throws |
  | `checkCudaErrors(from)` | 7 | prints only |
  | `checkCusolver` | 6 | calls `exit()` |
  | `checkCublas` | 7 | calls `exit()` |

  The roughly 37 hand-written `cudaGetLastError()` checks become `CUSH_CHECK_LAUNCH()`.
- `DeviceParticleData` owns its memory through `DeviceBuffer`, so `loadDataToDevice` no longer hands out a raw pointer that leaks when an export throws.

**Done when:**
- the non-dimensionalization exists once, in `Physics/NonDimensionalization`;
- the Hamaker constant is passed in by every caller;
- nothing below `Physics/` includes the parameter structs;
- no library code calls `exit()`, and `CUSH_CHECK` / `CUSH_CHECK_LAUNCH` are the only error-check helpers;
- the goldens match.

**Status: done** on branch `reorg/phase-3-parameters`.
- **Tests.** After a full rebuild, 34 gtests and 22 pytest tests pass, and the goldens match. The suite also passes in a build with `CUSH_SYNC_CHECKS` on, so no kernel was failing silently.
- **Error checking.**
  - About 300 call sites now go through the new macros. 126 calls that were not checked before (allocations, copies, cuBLAS and cuFFT calls) now are.
  - About 50 kernel launches gained `CUSH_CHECK_LAUNCH()`.
  - Destructors use a third macro, **`CUSH_CHECK_NOTHROW`**, which logs instead of throwing, because a destructor that throws ends the process.
  - The cuSOLVER and cuBLAS info checks throw `cush::CudaError`.
  - The RK4 stage updates throw on a failed cuBLAS call, where before they printed and carried on.
  - `ExportProperties.cu` had a second `exit(1)`, for an invalid drive type; it now throws `std::invalid_argument`.
  - 5 new gtests cover the macros.
- **Parameters.**
  - `PhysicalParameters` holds the SI inputs. Besides the Hamaker constant, the **film density** (the 150 kg/m³ that was hard-coded) is now also a caller input, because it is a material property too.
  - `Scales` holds the derived scales. `DimensionlessParameters` (renamed from `ProblemProperties`) carries its `Scales` as a member, because the optomechanical kernels need the scales on the device.
  - `NonDimensionalization` throws `std::invalid_argument` for non-positive inputs. A missing material therefore fails loudly instead of producing NaNs.
  - The optomechanical conversion lives in `Optomechanics/OptomechanicalNonDimensionalization.h`, because Physics may not include Optomechanics. It only applies the `Scales` that Physics computes.
- **ABI.**
  - `SimProperties` gains `hamaker_constant` and `film_density`, at the end of the struct.
  - The 8 exports that took `L, rho, kappa, depth` as separate doubles now take `SimProperties*`, so the next new parameter will not change every signature.
  - Python keeps its signatures. `HELIUM_FILM` is the single place for the Python defaults. `CSimulationProperties` fills the material from it, and the scalar-argument methods take keyword-only `hamaker_constant` / `film_density`. The C++ library has no defaults.
- **Layering.**
  - `BoundaryIntegral/Periodic2D/InterfaceParameters.h` is what the periodic solver reads, and `DimensionlessParameters` extends it.
  - The boundary-problem virtual methods take `InterfaceParameters&`. Problems that need the scales use their own copy of the full parameters.
- **To check (physics):** `LightIntensity::get_current_intensity_drive_strength` divides by the dimensionless `rho`, which is the density ratio, on top of a `base_energy` that already contains the film density. This is unchanged from before. The goldens use ρ equal to the film density, so they would not show a difference.

### Phase 4 — Remove duplication

A refactor only: the same arithmetic in the same order, so the goldens stay bit-identical. Changes that alter numerics belong to Phase 8.

- **RK4 steppers.** `RungeKuttaStepper` (`Ode/RK4_Time_Dependent.cuh`) and `AutonomousRungeKuttaStepper` are near-copies of each other, and each specialises `step()` for `std_complex`, `cuDoubleComplex` and `double`. Replace them with one `RungeKutta4<T>`:
  - One problem interface, `OdeProblem<T>` in `Ode/`, with `size()`, `setStream()` and `rhs(const T* y, double t, T* dydt, Evaluation kind)`. Time becomes an argument instead of `setCurrentTime` before each stage. `AutonomousProblem<T>` remains as a thin subclass that ignores `t`.
  - `Evaluation { AcceptedPoint, Stage }` replaces `setSaveProgress`. RK4 passes `AcceptedPoint` for stage 1 and `Stage` for stages 2 to 4, which is exactly what the flag does today. The drive kernel is not touched.
  - `OdeSolver<T>` gains `initialize`, `state()`, `time()`, `setStream` and `addObserver`. Options stay per algorithm (`RK4Options`, `RK45_Options`, `GaussLegendre2Options`).
  - Output goes through `StepObserver<T>::onAcceptedStep(t, y, n, stream)`: `TrajectoryLogger`, a `DeviceTrajectoryRecorder` (the timed stepper's `devTimes`/`devYs` and `copy*ToHost`), and an adapter for `DataLogger`. The progress bar becomes an opt-in observer.
  - The six `step()` specialisations become one `axpy` overload set in `LinearAlgebra/` (`cublasZaxpy` / `cublasDaxpy`). This also deletes the `cuDoubleComplex` copy, which writes k1 and k2 into the wrong stage buffers. Only the `ODE_Solvers.RK4` gtest used it, and its 1e-2 tolerance hid the error. No export or golden is affected.
  - Steppers take the size from `problem.size()`, not from a template parameter, which saves Phase 7 that work.
  - `RK45` and `GaussLegendre2` change only their constructor (`OdeProblem<T>&`) and their output (observers). Their numerics are unchanged.
- **Linear solvers.** Merge `MatrixSolver<N, batch>` and `DenseLuSolver` into one runtime-size LU solver. Both make the same `getrf`/`getrs` calls. The batched path is used: the Jacobian and Gauss-Legendre exports build `BaseBoundaryIntegralCalculator<N, 3N>`, so the merged solver keeps it (cuBLAS `getrfBatched`/`getrsBatched`). Benchmark the Cartesian RHS before and after.
- **Derivatives.** The FFT (complex, periodic) and finite-difference (real, non-uniform) derivatives are different algorithms on different data, so they get no common interface. Delete the unused `FftDerivativeBase` and `ComplexFftDerivative`.
- **Dead code.** Delete `Euler` (`Ode/SimpleEuler.cuh`), `IntegrableProblem` and `BatchedAutonomousProblem`.
- **Layering.** Fix the three violations that `docs/architecture.md` assigns to this phase. The RK4 include of the Periodic2D integrators goes away with the merge. `GaussLegendre` stops including `Export/ExportTypes.cuh`, and `RealBoundaryIntegralCalculator` stops including `Optomechanics/DelayedIntensityTerm.cuh`.
- **Done when:** each concept has one implementation, nothing unused remains in `Ode/`, and the goldens are bit-identical.

**Status: done** on branch `reorg/phase-4-dedup`. After a full rebuild, 35 gtests and 22 pytest tests pass. All 47 golden output arrays are bit-identical to a build of `main` (each case was also run three times to confirm it is reproducible). The RHS, Jacobian, Gauss-Legendre and RK4 export cases run at the same speed or slightly faster.

What changed against the plan above:
- **Interfaces.** `Ode/OdeProblem.h` holds `OdeProblem<T>` and `Evaluation`, and `OdeProblem` also has `start(t0)`, which replaces `TimedProblem::setStartingTime`. `AutonomousProblem<T, N>` keeps its size template, so the twelve existing problems are unchanged. Its `rhs()` forwards to the old `run(T*, T*)`, which is not yet const-correct (Phase 8). `TimedBoundaryIntegrator` now overrides `rhs()` and `start()` directly.
- **Observers.** `IO/StepObserver.h` has `onStart` (default no-op) and `onAcceptedStep`. It lives in `IO/` so that loggers do not depend on `Ode/`. There is no separate `DeviceTrajectoryRecorder`. `TrajectoryLogger<T>` covers both uses, sizes itself at runtime, and records the initial state only when asked, which Gauss-Legendre has always done. `IO/ProgressBarObserver.h` is the progress bar.
  - The exports decide what is returned. `Export/ExportInternal.cuh` gains `copyIntegrationResultToHost`: it returns the trajectory, or only the final state when `returnTrajectory` is off.
  - No `DataLogger` adapter was needed. RK45 was its only solver user, and RK45 now reports through observers like the others.
- **Options.** `returnTrajectory` left `GaussLegendre2Options`. The C ABI's `GaussLegendreOptions` keeps it, and `createOptionsFromCOptions` moved to `Export/IntegrationExports.cu`.
- **Solvers.** `LinearAlgebra/Blas.cuh` (`cush::blas::axpy`) holds the overload set. `DenseLuSolver(n, batch = 1)` replaces `MatrixSolver`. RK45 passes the Fehlberg stage times to `rhs()`, and Gauss-Legendre passes t + c1 h and t + c2 h, so time-dependent problems receive correct times too.
- **Deleted:** `Ode/RK4_Time_Dependent.cuh`, `Ode/AutonomousRungeKuttaStepper.cuh`, `Ode/SimpleEuler.cuh`, `LinearAlgebra/MatrixSolver.cuh`, `Core/VectorUtilities.cuh` (`appendToVector`), `TimedProblem`, `IntegrableProblem`, `BatchedAutonomousProblem`, `FftDerivativeBase`, `ComplexFftDerivative`.
- **New test:** `ODE_Solvers.RungeKutta4IsFourthOrderForEveryStateType` integrates z' = iz exactly and checks that halving the step divides the error by 16, for `std_complex` and `cuDoubleComplex`.
- **Golden note:** three augmented cases differ from the Phase 0 goldens (recorded at `9346c56`) by at most 1.4e-16 relative. The difference predates this phase, since `main` gives the same values, and it is within the test tolerance.
- **Not run:** the suite under `CUSH_SYNC_CHECKS`, because the scripts have no switch for it.

### Phase 5 — Portable build

The build stays on MSBuild.

- Add a vcpkg manifest (`vcpkg.json`: gtest, boost-math, hdf5/highfive, opencv for the App only) and a `Directory.Build.props` holding CUDA architectures, the C++ standard (one value for Debug and Release) and include roots. This removes every user-specific absolute path.
- The Python include and lib paths (needed by Tests and App for plotting) come from one property, `CushPythonRoot`, set in a user-local `Directory.Build.user.props` that git ignores.
- Document the required toolset (v145 / VS 18) in the README, or retarget the solution.
- **Done when:** a fresh clone builds on another Windows machine following the README alone.

### Phase 6 — Python package

The package stays private for now: it is installed from the local checkout and not published. Publishing on PyPI can be revisited closer to open-sourcing, which would also need a license and a check of what lab data the repository contains.

- Make `Python/cusuperhelium/` an installable package (`pip install -e .`). It finds the DLL via a `CUSUPERHELIUM_DLL` environment variable, falling back to the build output, instead of `os.getcwd()`.
- Split `rhs.py` into a bindings layer (ctypes signatures, set once at load time) and a convenience API.
- Move analysis scripts and notebooks under `analysis/` and `notebooks/`, and remove the `sys.path` hacks.
- **Done when:** `pytest Python/tests` runs from any directory.

### Phase 7 — Runtime sizes for the Cartesian solver (after the reorganization)

- Remove the `N` template parameter from the periodic-2D classes, following the radial solver's pattern: sizes passed to constructors, buffers allocated at runtime.
- Delete the per-export `switch (N)` blocks. Every export then takes `N` directly, as `calculateVelocitiesRadialSymmetryModes` does.
- Benchmark the Cartesian RHS and a short integration before and after. A slowdown of a few percent is acceptable. A larger one is investigated before merging, for example by keeping fixed-size kernels for the FFT step.
- **Done when:** no class is templated on `N`, any `N` the user passes is accepted, and goldens match.

### Phase 8 — Time-integration interface (before new solvers)

Phase 4 gives every integrator one interface but keeps one leak: `Evaluation::AcceptedPoint` tells the problem when the solver is at an accepted state. That works for RK4, where stage 1 always sits at an accepted point, but not for adaptive or implicit schemes: RK45 re-evaluates a rejected step at the same point, and a first-same-as-last scheme never evaluates there. This phase makes the integrators reusable by any problem, without the caller needing to know how a scheme stages its evaluations. It comes before any new integrator or solver is added.

- **Race test first.** `DelayedIntensityTermDevice::save_value` has every thread write `*prev_time`, while every thread also reads it in `calculate_new_delayed_intensity`. With N > 256 there are several blocks, and CUDA does not order them, so a late block can read the new time, take the `t == prev_time` branch, and return and save 0 for its nodes. Write a test that shows whether this happens (several blocks, a known intensity history, compare against a host reference) before changing anything.
- **Fix the race.** Update `prev_time` once per step outside the per-node kernel, or pass the previous time in as an argument. Results may change for runs where the race fired, so report the difference in the PR.
- **`acceptStep` instead of `Evaluation`.** Problems get `acceptStep(const T* y, double t)`, which the solver calls once per accepted step, plus a call at the start. `rhs` loses its `kind` argument. The delayed-intensity history moves from the rhs kernel into that hook:
  - Stage evaluations read the saved D and no longer recompute it at the accepted point.
  - The `t == prev_time` branch, which returns 0, is replaced by reading the saved value.
  - The first step keeps today's behaviour (D = 0 at `initial_time`) or changes it on purpose. Record which in the PR.
- **Audit the rest of the call surface** for scheme details leaking into callers. Examples: `initialize(y, onDevice)` flags, per-scheme `setOptions` calls in the exports, steppers that need the problem's internal buffers.
- **Done when:** any `OdeProblem` runs unchanged with RK4, RK45 and Gauss-Legendre; the race test passes; and the goldens match, or each difference is explained by the race fix.

## 5. Suggested order and size

| Phase | Risk | Rough size | Blocks |
|---|---|---|---|
| 0 Safety net | low | ½–1 day | everything else |
| 1 Hygiene | low | 1–2 days | 1b |
| 1b History rewrite | low if verified; needs re-cloning everywhere | ½ day | 2 |
| 2 Layout | medium (merge conflicts) | 1 day + review | 3, 4 |
| 3 Parameters/dispatch | medium | 1–2 days | — |
| 4 Deduplication | medium (bit-identical refactor) | 2–4 days | 7, 8 |
| 5 Build | low–medium | 1–2 days | — |
| 6 Python | low | 1 day | — |
| 7 Runtime sizes | medium (performance) | 2–3 days | — |
| 8 Time-integration interface | medium–high (numerics: race fix, history hook) | 2–3 days | new integrators and solvers |

Recommended sequence: **0 → 1 → 1b → 2 → 3 → 4 → 5 → 6 → 7 → 8 → option A (half-space kernel).** Phase 4 moved ahead of 5 and 6 once it was scoped as a bit-identical refactor. Work on the axisymmetric solver (option A, then option B) waits until the reorganization is finished, so that it lands on the common interfaces. The goal is plug and play: the same problem can be solved with a different integrator or boundary-integral solver without changing the problem's code. Phase 4 is kept to a refactor that leaves the goldens bit-identical. Every change that alters numerics in the time integration (the `prev_time` race, the delayed-intensity history hook) waits for Phase 8. Phase 7 comes after 4 because merging the solvers removes much of the templated code it would otherwise have to convert. Phase 8 comes before any new integrator or solver is added.

## 6. Decisions

Recorded November 2026.

1. **Build system:** keep MSBuild, made portable with a vcpkg manifest and `Directory.Build.props` (Phase 5). CMake is not planned.
2. **Sizes:** the Cartesian solver moves to runtime sizes, but only after the reorganization is complete (Phase 7). Until then, the per-export `switch (N)` blocks stay as they are, with no interim dispatch helper.
3. **Python package:** private, installed from the checkout. Publishing on PyPI may be revisited later.
4. **History:** large binaries are removed from history by path, with a mirror backup and a check that no code file is lost (Phase 1b). `.h5` files and other simulation output are not tracked from now on.
5. **Plotting:** allowed in tests and the App, never in the DLL. Tests save figures and open windows only on request (section 3, "Visual tests").
6. **Hamaker constant:** a caller-supplied parameter, because it determines the simulated material. It is not a library constant.
7. **Error checking:** one throwing macro family, `CUSH_CHECK` / `CUSH_CHECK_LAUNCH`, with the project prefix `CUSH_` and the namespace `cush::` for the error type. The library never calls `exit()` (Phase 3).

## 7. Open questions (after the phases)

Things to check once the reorganization is finished. They do not block any phase.

1. **Optomechanical drive strength and the density ratio.** `LightIntensity::get_current_intensity_drive_strength` returns hbar G / (E0 t0 rho sigma^2), with `rho` the *dimensionless* density ratio rho_upper / rho_film. The energy scale E0 = 3 rho_film alpha L0^4 / h0^4 already contains the film density, so the extra division by the ratio may be wrong. Every current run uses rho = rho_film, which makes the ratio 1, so no result is affected yet. Check the derivation, and add a test with rho != rho_film before relying on that regime. (Found in Phase 3.)
2. **Delayed-intensity history under adaptive stepping.** The history is a fixed-step construction: the class comment says it "works only with fixed time steps". Once Phase 8 has moved it to `acceptStep`, decide whether the driven problems should run with RK45, and if so, how D should be updated over a variable step. (Found while planning Phase 4.)
