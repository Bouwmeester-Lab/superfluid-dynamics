"""Golden regression tests for the C exports of CuSuperHelium.dll.

Each case calls one export with fixed inputs and compares every output array
with a recorded golden file in tests/golden/<case>.npz. The reorganization
must not change these numbers: refactors are checked against them.

Record or refresh the goldens (only when a change of numbers is intended and
explained in the commit):

    python -m pytest CuSuperHelium/Python/tests/test_golden.py --update-goldens

Error measure per array: max |new - golden| / max |golden| (relative to the
array's scale, so arrays with zeros are handled).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

# Physical parameters (SI) shared by the Cartesian cases: a 1 um period, 15 nm helium film.
L_SI = 1.0e-6
DEPTH_SI = 15.0e-9
RHO = 150.0
KAPPA = 0.0

# Material the goldens were recorded with (helium film), passed explicitly to every call.
ALPHA_HAMAKER = 3.5e-24
FILM_DENSITY = 150.0
MATERIAL = dict(hamaker_constant=ALPHA_HAMAKER, film_density=FILM_DENSITY)

# Base units, mirroring Physics/NonDimensionalization.h.
BASE_LENGTH = L_SI / (2.0 * np.pi)
BASE_TIME = np.sqrt(BASE_LENGTH / (3.0 * ALPHA_HAMAKER / DEPTH_SI**4))


def periodic_state(n: int, amplitude: float = 0.01) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nondimensional surface x, y and potential phi on a 2*pi period with two Fourier modes."""
    x = 2.0 * np.pi * np.arange(n, dtype=np.float64) / n
    y = amplitude * (np.cos(x) + 0.5 * np.sin(2.0 * x))
    phi = amplitude * (np.sin(x) - 0.3 * np.cos(3.0 * x))
    return x, y, phi


def sim_props(rhs):
    return rhs.CSimulationProperties(L=L_SI, rho=RHO, kappa=KAPPA, depth=DEPTH_SI,
                                     use_expansions=False, expansion_order=2, infinite_depth=False, **MATERIAL)


def optomechanical_props(rhs, drive_type: int):
    return rhs.COptomechanicalProperties(
        detuning=0.2 / BASE_TIME,
        gamma=0.5 / BASE_TIME,
        G=-0.02 / (BASE_TIME * BASE_LENGTH),
        tau=1.0 * BASE_TIME,
        max_intensity=0.1,
        initial_time=0.0,
        location_x0_mode=np.pi * BASE_LENGTH,
        sigma_optical_mode=0.2 * BASE_LENGTH,
        sigma_thermal_mode=0.5 * BASE_LENGTH,
        beta=1.0,
        damping_strength=0.0,
        drive_type=drive_type,
        ramp_rate=0.25 / BASE_TIME,
        omega_drive=0.0,
    )


def rk4_options(rhs, steps: int = 20, step: float = 0.01):
    return rhs.CRK4Options(timeStep=step * BASE_TIME, t0=0.0, t1=steps * step * BASE_TIME, returnTrajectory=True)


# ---------------------------------------------------------------- cases

Outputs = dict[str, np.ndarray]


def case_rhs_from_vectors(n: int) -> Callable[..., Outputs]:
    def run(sim, rhs) -> Outputs:
        x, y, phi = periodic_state(n)
        res, vx, vy, dphi = sim.calculate_rhs_from_vectors(x, y, phi, L_SI, RHO, KAPPA, DEPTH_SI, **MATERIAL)
        assert res == 0
        return {"vx": vx, "vy": vy, "dphi": dphi}
    return run


def case_rhs256_fixed(sim, rhs) -> Outputs:
    x, y, phi = periodic_state(256)
    res, vx, vy, dphi = sim.calculate_rhs_from_vectors(x, y, phi, L_SI, RHO, KAPPA, DEPTH_SI, **MATERIAL)
    assert res == 0
    return {"vx": vx, "vy": vy, "dphi": dphi}


def case_rhs256_batched(sim, rhs) -> Outputs:
    batch = 4
    states = [periodic_state(256, amplitude=0.005 * (b + 1)) for b in range(batch)]
    x, y, phi = (np.concatenate([s[k] for s in states]) for k in range(3))
    res, vx, vy, dphi = sim.calculate_rhs_from_vectors_batched(x, y, phi, L_SI, RHO, KAPPA, DEPTH_SI, batch, **MATERIAL)
    assert res == 0
    return {"vx": vx, "vy": vy, "dphi": dphi}


def case_vorticities256(sim, rhs) -> Outputs:
    x, y, phi = periodic_state(256)
    out = sim.calculate_vorticities_from_vectors(x + 1j * y, phi.astype(np.complex128), L_SI, RHO, KAPPA, DEPTH_SI, **MATERIAL)
    res, a, zp, zpp = out
    assert res == 0
    return {"a": a, "Zp": zp, "Zpp": zpp}


def case_derivative_fft256(sim, rhs) -> Outputs:
    x = 2.0 * np.pi * np.arange(256) / 256
    f = (np.sin(3.0 * x) + 0.2 * np.cos(7.0 * x)) + 1j * np.cos(2.0 * x)
    res, df = sim.calculate_derivative_fft(f.astype(np.complex128))
    assert res == 0
    return {"df": df}


def case_jacobian(n: int) -> Callable[..., Outputs]:
    def run(sim, rhs) -> Outputs:
        x, y, phi = periodic_state(n)
        res, jac = sim.calculate_jacobian(x, y, phi, L_SI, RHO, KAPPA, DEPTH_SI, **MATERIAL)
        assert res == 0
        return {"J": jac}
    return run


def case_gauss_legendre(n: int) -> Callable[..., Outputs]:
    def run(sim, rhs) -> Outputs:
        x, y, phi = periodic_state(n)
        gl = rhs.CGaussLegendreProperties(t0=0.0, t1=0.2, stepSize=0.05, newtonTolerance=1e-10,
                                          maxNewtonIterations=12, allowSimplifiedFallback=True,
                                          returnTrajectory=True, armijo_c=1e-4, backtrack=0.5,
                                          minAlpha=1e-6, maxStepsHalves=6)
        res, t, states = sim.integrate(np.concatenate((x, y, phi)), sim_props(rhs), gl)
        assert res == 0
        return {"t": t, "states": states}
    return run


def case_optomechanical_rk4(n: int) -> Callable[..., Outputs]:
    def run(sim, rhs) -> Outputs:
        x, y, phi = periodic_state(n)
        res, t, states = sim.integrate_optomechanical_problem(
            np.concatenate((x, y, phi)), sim_props(rhs), optomechanical_props(rhs, rhs.DriveType.Constant), rk4_options(rhs))
        assert res == 0
        return {"t": t, "states": states}
    return run


def augmented_state(n: int, ramped: bool) -> np.ndarray:
    x, y, phi = periodic_state(n)
    delayed = 0.01 * np.exp(-((x - np.pi) ** 2) / 0.5)
    parts = [x, y, phi, delayed] + ([np.array([0.05])] if ramped else [])
    return np.ascontiguousarray(np.concatenate(parts))


def case_augmented_rk4(n: int, ramped: bool) -> Callable[..., Outputs]:
    def run(sim, rhs) -> Outputs:
        drive = rhs.DriveType.Ramped if ramped else rhs.DriveType.Constant
        res, t, states = sim.integrate_augmented_optomechanical_problem(
            augmented_state(n, ramped), sim_props(rhs), optomechanical_props(rhs, drive), rk4_options(rhs))
        assert res == 0
        return {"t": t, "states": states}
    return run


def case_augmented_rhs(n: int) -> Callable[..., Outputs]:
    def run(sim, rhs) -> Outputs:
        out = sim.calculate_augmented_rhs(augmented_state(n, False), sim_props(rhs),
                                          optomechanical_props(rhs, rhs.DriveType.Constant))
        return {"rhs": out}
    return run


def case_radial(n: int) -> Callable[..., Outputs]:
    """Axisymmetric Bessel solver: flat and curved surfaces, deep and shallow film, Bessel-mode potential."""
    def run(sim, rhs) -> Outputs:
        from scipy.special import j0, jn_zeros

        r = (np.arange(n, dtype=np.float64) + 0.5) / n
        phi = j0(float(jn_zeros(0, 2)[-1]) * r)
        surfaces = {"flat": np.zeros_like(r), "curved": 0.02 * j0(float(jn_zeros(0, 3)[-1]) * r)}
        out: Outputs = {}
        for name, z in surfaces.items():
            for depth in (0.1, 1.0):
                props = rhs.CSimulationProperties(1.0, 145.0, 0.0, depth, False, 2, False, **MATERIAL)
                res, vr, vz = sim.calculate_velocities_radial_symmetry(r, z, phi, props)
                assert res == 0
                out[f"{name}_d{depth}_vr"] = vr
                out[f"{name}_d{depth}_vz"] = vz
        return out
    return run


@dataclass(frozen=True)
class Case:
    name: str
    run: Callable[..., Outputs]
    rtol: float = 1e-10


CASES = [
    Case("rhs_N64", case_rhs_from_vectors(64)),
    Case("rhs_N256", case_rhs_from_vectors(256)),
    Case("rhs256_fixed", case_rhs256_fixed),
    Case("rhs256_batched4", case_rhs256_batched),
    Case("vorticities256", case_vorticities256),
    Case("derivative_fft256", case_derivative_fft256),
    Case("jacobian_N32", case_jacobian(32)),
    Case("jacobian_N64", case_jacobian(64)),
    Case("gauss_legendre_N32", case_gauss_legendre(32), rtol=1e-9),
    Case("gauss_legendre_N64", case_gauss_legendre(64), rtol=1e-9),
    Case("optomechanical_rk4_N32", case_optomechanical_rk4(32), rtol=1e-9),
    Case("optomechanical_rk4_N64", case_optomechanical_rk4(64), rtol=1e-9),
    Case("augmented_rk4_N32", case_augmented_rk4(32, ramped=False), rtol=1e-9),
    Case("augmented_rk4_ramped_N64", case_augmented_rk4(64, ramped=True), rtol=1e-9),
    Case("augmented_rhs_N64", case_augmented_rhs(64)),
    Case("radial_N256", case_radial(256)),
    Case("radial_N1024", case_radial(1024)),
]


# ---------------------------------------------------------------- harness

def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=GOLDEN_DIR.parent, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def relative_error(new: np.ndarray, ref: np.ndarray) -> float:
    scale = float(np.max(np.abs(ref))) if ref.size else 0.0
    diff = float(np.max(np.abs(new - ref))) if ref.size else 0.0
    return diff / scale if scale > 0.0 else diff


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_golden(case: Case, sim, update_goldens: bool) -> None:
    import cusuperhelium as rhs

    outputs = {k: np.asarray(v) for k, v in case.run(sim, rhs).items()}
    for key, value in outputs.items():
        assert np.all(np.isfinite(value)), f"{case.name}: non-finite values in {key}"

    path = GOLDEN_DIR / f"{case.name}.npz"
    if update_goldens:
        GOLDEN_DIR.mkdir(exist_ok=True)
        np.savez_compressed(path, __recorded_at_commit=np.array(_git_commit()), **outputs)
        pytest.skip(f"recorded {path.name}")

    assert path.is_file(), f"missing golden {path.name}; record it with --update-goldens"
    golden = np.load(path)
    expected_keys = {k for k in golden.files if not k.startswith("__")}
    assert set(outputs) == expected_keys, f"{case.name}: outputs {sorted(outputs)} != golden {sorted(expected_keys)}"

    failures = []
    for key in sorted(expected_keys):
        ref = golden[key]
        new = outputs[key]
        if new.shape != ref.shape:
            failures.append(f"{key}: shape {new.shape} != golden {ref.shape}")
            continue
        err = relative_error(new, ref)
        if err > case.rtol:
            failures.append(f"{key}: max relative error {err:.3e} > {case.rtol:.0e}")
    assert not failures, f"{case.name} (golden from {golden['__recorded_at_commit']}):\n  " + "\n  ".join(failures)
