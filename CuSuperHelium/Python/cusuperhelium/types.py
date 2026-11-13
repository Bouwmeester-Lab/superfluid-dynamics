"""C ABI types of CuSuperHelium.dll (mirrors Export/ExportTypes.cuh) and the film material defaults."""
from ctypes import Structure, c_bool, c_double, c_int, c_size_t
from dataclasses import dataclass
from enum import IntEnum


@dataclass(frozen=True)
class FilmMaterial:
    """Material of the film. The C++ library has no default material: every call states it. These Python
    defaults only save callers from repeating the same values; pass another FilmMaterial, or the fields
    directly, to simulate a different material."""
    hamaker_constant: float  # van der Waals coefficient alpha [m^5/s^2]; the film feels 3 alpha / h^4
    density: float           # [kg/m^3]


# Helium-4 film. alpha: https://arxiv.org/html/2504.13001v1#S5 (section 6.3); the value the library hard-coded
# before the material became an input.
HELIUM_FILM = FilmMaterial(hamaker_constant=3.5e-24, density=150.0)


class CSimulationProperties(Structure):
    _fields_ = [("L", c_double),
                ("rho", c_double),
                ("kappa", c_double),
                ("depth", c_double),
                ("use_expansions", c_bool),
                ("expansion_order", c_int),
                ("infinite_depth", c_bool),
                ("hamaker_constant", c_double),
                ("film_density", c_double),
                ]

    def __init__(self, *args, material: FilmMaterial = HELIUM_FILM, **kwargs):
        # ctypes would leave unset fields at 0, which the library rejects; fill the material from `material`
        # unless the caller gives the fields explicitly.
        names = [name for name, _ in self._fields_]
        for name, value in (("hamaker_constant", material.hamaker_constant), ("film_density", material.density)):
            if names.index(name) >= len(args):
                kwargs.setdefault(name, value)
        super().__init__(*args, **kwargs)


def sim_props(L: float, rho: float, kappa: float, depth: float, hamaker_constant: float, film_density: float) -> CSimulationProperties:
    """CSimulationProperties for the scalar-argument methods: finite depth, no expansions."""
    return CSimulationProperties(L=L, rho=rho, kappa=kappa, depth=depth, use_expansions=False, expansion_order=1,
                                 infinite_depth=False, hamaker_constant=hamaker_constant, film_density=film_density)


class CGaussLegendreProperties(Structure):
    _fields_ = [("t0", c_double),
                ("t1", c_double),
                ("stepSize", c_double),
                ("newtonTolerance", c_double),
                ("maxNewtonIterations", c_size_t),
                ("allowSimplifiedFallback", c_bool),
                ("returnTrajectory", c_bool),
                ("armijo_c", c_double),
                ("backtrack", c_double),
                ("minAlpha", c_double),
                ("maxStepsHalves", c_size_t)
                ]


class CRK4Options(Structure):
    _fields_ = [("timeStep", c_double),
                ("t0", c_double),
                ("t1", c_double),
                ("returnTrajectory", c_bool)
                ]


class DriveType(IntEnum):
    Constant = 0
    Ramped = 1
    Sine = 2


class COptomechanicalProperties(Structure):
    _fields_ = [("detuning", c_double),
                ("gamma", c_double),
                ("G", c_double),
                ("tau", c_double),
                ("max_intensity", c_double),
                ("initial_time", c_double),
                ("location_x0_mode", c_double),
                ("sigma_optical_mode", c_double),
                ("sigma_thermal_mode", c_double),
                ("beta", c_double),
                ("damping_strength", c_double),
                ("drive_type", c_int),
                ("ramp_rate", c_double),
                ("omega_drive", c_double),
                ]
