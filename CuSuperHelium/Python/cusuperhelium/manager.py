"""Convenience API over CuSuperHelium.dll: numpy arrays in, numpy arrays out.

The signatures are set once by bindings.bind when the DLL is loaded. Methods return the DLL's status code
(0 = success) first, followed by the outputs, as they always have.
"""
from __future__ import annotations

import os
from ctypes import POINTER, byref, c_double, c_size_t

import numpy as np
from numpy.typing import NDArray

from .library import load_library
from .types import (HELIUM_FILM, CGaussLegendreProperties, COptomechanicalProperties, CRK4Options,
                    CSimulationProperties, DriveType, sim_props)

_ALPHA = HELIUM_FILM.hamaker_constant
_DENSITY = HELIUM_FILM.density


def _same_length(*arrays: NDArray) -> int:
    n = arrays[0].shape[0]
    if any(a.shape[0] != n for a in arrays[1:]):
        raise ValueError("Input arrays must have the same length")
    return n


def _as_f64(a: NDArray) -> NDArray:
    return np.ascontiguousarray(a, dtype=np.float64)


class SimulationManager:
    """Owns the loaded DLL. `dll_path` overrides the lookup described in cusuperhelium.library."""

    def __init__(self, dll_path: str | os.PathLike | None = None):
        self.lib = load_library(dll_path)

    # ------------------------------------------------------------ periodic 2D right-hand sides

    def calculate_rhs_from_file(self, input_file: str, output_file: str, n: int, L: float, rho: float, kappa: float,
                                depth: float, *, hamaker_constant: float = _ALPHA, film_density: float = _DENSITY) -> int:
        """RHS of the n-point state in an HDF5 file, written to another HDF5 file. Returns the status."""
        return self.lib.calculateRHSFromFile(input_file.encode(), output_file.encode(),
                                             byref(sim_props(L, rho, kappa, depth, hamaker_constant, film_density)), n)

    def _rhs(self, function, x, y, phi, props, *extra):
        _same_length(x, y, phi)
        x, y, phi = _as_f64(x), _as_f64(y), _as_f64(phi)
        vx, vy, dphi = np.empty_like(x), np.empty_like(x), np.empty_like(x)
        return function(x, y, phi, vx, vy, dphi, byref(props), *extra), vx, vy, dphi

    def calculate_rhs_from_vectors(self, x: NDArray, y: NDArray, phi: NDArray, L: float, rho: float, kappa: float,
                                   depth: float, *, hamaker_constant: float = _ALPHA, film_density: float = _DENSITY):
        """RHS (vx, vy, dphi/dt) of one periodic interface of N = len(x) points (any N >= 2)."""
        return self._rhs(self.lib.calculateRHSFromVectors, x, y, phi,
                         sim_props(L, rho, kappa, depth, hamaker_constant, film_density), x.shape[0])

    def calculate_rhs_from_vectors_batched(self, x: NDArray, y: NDArray, phi: NDArray, L: float, rho: float,
                                           kappa: float, depth: float, batch_size: int, *,
                                           hamaker_constant: float = _ALPHA, film_density: float = _DENSITY):
        """RHS of batch_size independent interfaces stored back to back; each has N = len(x) / batch_size points."""
        if batch_size < 1 or x.shape[0] % batch_size != 0:
            raise ValueError("The array length must be a positive multiple of batch_size")
        n = x.shape[0] // batch_size
        _same_length(x, y, phi)
        return self._rhs(self.lib.calculateRHSFromVectorsBatched, x, y, phi,
                         sim_props(L, rho, kappa, depth, hamaker_constant, film_density), n, batch_size)

    def calculate_vorticities_from_vectors(self, Z: NDArray, phi: NDArray, L: float, rho: float, kappa: float,
                                           depth: float, *, hamaker_constant: float = _ALPHA,
                                           film_density: float = _DENSITY):
        """Vorticities a and the derivatives Zp, Zpp of an N = len(Z) point interface. Returns (status, a, Zp, Zpp)."""
        n = _same_length(Z, phi)
        vorticity = np.empty(n, dtype=np.float64)
        Zp = np.empty(n, dtype=np.complex128)
        Zpp = np.empty(n, dtype=np.complex128)
        status = self.lib.calculateVorticitiesFromVectors(
            np.ascontiguousarray(Z, dtype=np.complex128), np.ascontiguousarray(phi, dtype=np.complex128),
            vorticity, Zp, Zpp, byref(sim_props(L, rho, kappa, depth, hamaker_constant, film_density)), n)
        return status, vorticity, Zp, Zpp

    def calculate_derivative_fft(self, input: NDArray) -> tuple[int, NDArray]:
        """Spectral derivative of N = len(input) periodic complex samples (no 2 pi / N scaling)."""
        output = np.empty(input.shape[0], dtype=np.complex128)
        return self.lib.calculateDerivativeFFT(np.ascontiguousarray(input, dtype=np.complex128), output, input.shape[0]), output

    def calculate_perturbed_states(self, x: NDArray, y: NDArray, phi: NDArray, L: float, rho: float, kappa: float,
                                      depth: float, epsilon: float = 1e-6, *, hamaker_constant: float = _ALPHA,
                                      film_density: float = _DENSITY):
        n = _same_length(x, y, phi)
        perturbed = np.empty((6 * n * n,), dtype=np.complex128)
        status = self.lib.calculatePerturbedStates(_as_f64(x), _as_f64(y), _as_f64(phi), perturbed,
                                                   byref(sim_props(L, rho, kappa, depth, hamaker_constant, film_density)),
                                                   epsilon, n)
        return status, perturbed

    def calculate_jacobian(self, x: NDArray, y: NDArray, phi: NDArray, L: float, rho: float, kappa: float, depth: float,
                           epsilon: float = 1e-6, *, hamaker_constant: float = _ALPHA, film_density: float = _DENSITY):
        n = _same_length(x, y, phi)
        J = np.ones((3 * n, 3 * n), dtype=np.float64, order='F') * 100
        status = self.lib.calculateJacobian(_as_f64(np.concatenate((x, y, phi))), J,
                                            byref(sim_props(L, rho, kappa, depth, hamaker_constant, film_density)),
                                            epsilon, n)
        return status, J

    # ------------------------------------------------------------ axisymmetric

    def calculate_velocities_radial_symmetry(self, r: NDArray, z: NDArray, phi: NDArray, sim_props: CSimulationProperties,
                                             n_modes: int | None = None):
        """Surface velocities (vr, vz) of an axisymmetric flow. n_modes sets the number of Bessel modes
        (None: the DLL default of 1024). Returns (status, vr, vz)."""
        if not isinstance(sim_props, CSimulationProperties):
            raise TypeError("sim_props must be a CSimulationProperties instance")
        r, z, phi = _as_f64(r), _as_f64(z), _as_f64(phi)
        if r.ndim != 1 or z.ndim != 1 or phi.ndim != 1:
            raise ValueError("Input arrays must be one-dimensional")
        n = _same_length(r, z, phi)
        vr, vz = np.empty_like(r), np.empty_like(r)
        if n_modes is None:
            return self.lib.calculateVelocitiesRadialSymmetry(r, z, phi, vr, vz, byref(sim_props), n), vr, vz
        if n_modes < 1:
            raise ValueError("n_modes must be at least 1")
        return self.lib.calculateVelocitiesRadialSymmetryModes(r, z, phi, vr, vz, byref(sim_props), n, n_modes), vr, vz

    # ------------------------------------------------------------ time integration

    def _trajectory(self, integrate, free, y0: NDArray, *args):
        """Runs an export that returns malloc'ed states and times, copies them, and frees the C memory."""
        y0 = _as_f64(y0)
        states_ptr, states_len = POINTER(c_double)(), c_size_t()
        times_ptr, times_len = POINTER(c_double)(), c_size_t()
        status = integrate(y0, byref(states_ptr), byref(states_len), byref(times_ptr), byref(times_len), *args)
        try:
            if status != 0 or not states_ptr:
                return status, None, None
            states = np.array(np.ctypeslib.as_array(states_ptr, shape=(states_len.value, y0.shape[0])), copy=True)
            times = (np.array(np.ctypeslib.as_array(times_ptr, shape=(times_len.value,)), copy=True)
                     if times_ptr else np.empty(0))
            return status, times, states
        finally:
            if states_ptr or times_ptr:
                free(states_ptr, times_ptr)

    def integrate(self, y0: NDArray, sim_props: CSimulationProperties, gl_props: CGaussLegendreProperties):
        """Gauss-Legendre (implicit, order 4) integration of the helium film. Returns (status, times, states)."""
        return self._trajectory(self.lib.integrateSimulationGL2, self.lib.integrateSimulationGL2_freeMemory,
                                y0, byref(sim_props), byref(gl_props), y0.shape[0] // 3)

    def integrate_optomechanical_problem(self, y0: NDArray, sim_props: CSimulationProperties,
                                         opt_props: COptomechanicalProperties, rk4_props: CRK4Options):
        """RK4 integration with the time-dependent optomechanical drive. Returns (status, times, states)."""
        return self._trajectory(self.lib.integrateOptomechanicalSimulationRK4,
                                self.lib.integrateOptomechanicalSimulationRK4_freeMemory,
                                y0, byref(sim_props), byref(rk4_props), byref(opt_props), y0.shape[0] // 3)

    def integrate_augmented_optomechanical_problem(self, y0: NDArray, sim_props: CSimulationProperties,
                                                   opt_props: COptomechanicalProperties, rk4_props: CRK4Options):
        """RK4 integration of the augmented state (x, y, phi, delayed intensity[, drive variable]).
        Returns (status, times, states)."""
        if opt_props.drive_type in (DriveType.Sine, DriveType.Ramped) and y0.shape[0] % 4 != 1:
            # the ramped and sine drives carry one extra scalar variable
            raise ValueError("Initial state vector length must be of the form 4n + 1 when ramping or sine driving is enabled")
        # the augmented state has 4 components per collocation point: x, y, phi and the delayed intensity
        return self._trajectory(self.lib.integrateAugmentedOptomechanicalSimulationRK4,
                                self.lib.integrateAugmentedOptomechanicalSimulationRK4_freeMemory,
                                y0, byref(sim_props), byref(rk4_props), byref(opt_props), y0.shape[0] // 4)

    def calculate_augmented_rhs(self, y0: NDArray, sim_props: CSimulationProperties, opt_props: COptomechanicalProperties):
        """Right-hand side of the augmented optomechanical state. Raises RuntimeError on failure."""
        y0 = _as_f64(y0)
        rhs = np.empty_like(y0)
        status = self.lib.calculateRhsAugmentedOptomechanical(y0, rhs, byref(sim_props), byref(opt_props), y0.shape[0] // 4)
        if status != 0:
            raise RuntimeError("Error in calculating augmented RHS")
        return rhs
