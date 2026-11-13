# cusuperhelium

Python bindings and tools for `CuSuperHelium.dll`. The package is private: install it from this checkout.

```powershell
pip install -e CuSuperHelium/Python            # from the repository root; add [test] for pytest, [analysis] for the scripts
```

The DLL is found through `CUSUPERHELIUM_DLL`, otherwise this checkout's build output
(`CuSuperHelium/x64/Release`, then `Debug`). Build it with `scripts/build.ps1`.

```python
import cusuperhelium as cush

sim = cush.SimulationManager()
status, vx, vy, dphi = sim.calculate_rhs_from_vectors(x, y, phi, L, rho, kappa, depth)
```

| Folder | Contents |
|---|---|
| `cusuperhelium/` | the package: C ABI types, ctypes bindings, DLL loading, `SimulationManager`, and shared tools (`fourier_series`, `gauss_legendre`, `finite_differences`, `thin_film`, `plots`) |
| `tests/` | pytest: golden regression tests of every export and the error paths. `pytest CuSuperHelium/Python/tests` works from any directory |
| `analysis/` | scripts that use the package (phase diagrams, radial velocities, studies) |
| `notebooks/` | Jupyter notebooks |
