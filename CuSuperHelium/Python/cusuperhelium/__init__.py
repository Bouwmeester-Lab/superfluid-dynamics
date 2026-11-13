"""Python bindings and tools for CuSuperHelium.dll, the CUDA boundary-integral solver for superfluid helium films.

    import cusuperhelium as cush
    sim = cush.SimulationManager()          # finds the DLL: CUSUPERHELIUM_DLL, else this checkout's build output
    status, vx, vy, dphi = sim.calculate_rhs_from_vectors(x, y, phi, L, rho, kappa, depth)

Layers:
- types: the C ABI structs (CSimulationProperties, ...) and the film material defaults (HELIUM_FILM).
- bindings: the ctypes signature of every export, set once when the DLL is loaded.
- library: finding and loading the DLL.
- manager: SimulationManager, the numpy-facing API.
Tools: fourier_series, gauss_legendre (host-side integrator), finite_differences, thin_film (analytic reference),
plots.
"""
from .finite_differences import jacobian_fd, normalize, richardson_extrapolation, richardson_jacobian
from .library import find_dll, load_library
from .manager import SimulationManager
from .types import (HELIUM_FILM, CGaussLegendreProperties, COptomechanicalProperties, CRK4Options,
                    CSimulationProperties, DriveType, FilmMaterial)

__all__ = [
    "SimulationManager", "find_dll", "load_library",
    "FilmMaterial", "HELIUM_FILM", "CSimulationProperties", "CGaussLegendreProperties", "CRK4Options",
    "COptomechanicalProperties", "DriveType",
    "jacobian_fd", "normalize", "richardson_extrapolation", "richardson_jacobian",
]
