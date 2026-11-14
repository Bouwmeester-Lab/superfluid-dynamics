"""ctypes signatures of every export of CuSuperHelium.dll (mirrors Export/ExportApi.cuh).

`bind(lib)` sets argtypes/restype once, when the DLL is loaded. The convenience API in manager.py only calls
the functions. Every parameter struct is passed by pointer, as the C signatures declare.
"""
from ctypes import CDLL, POINTER, c_char_p, c_double, c_int, c_size_t

import numpy as np
from numpy.ctypeslib import ndpointer

from .types import CGaussLegendreProperties, COptomechanicalProperties, CRK4Options, CSimulationProperties

# Array arguments: inputs are read-only, outputs must be writeable.
_IN = ndpointer(c_double, flags=("C_CONTIGUOUS", "ALIGNED"))
_OUT = ndpointer(c_double, flags=("C_CONTIGUOUS", "ALIGNED", "WRITEABLE"))
_IN_C = ndpointer(np.complex128, flags=("C_CONTIGUOUS", "ALIGNED"))
_OUT_C = ndpointer(np.complex128, flags=("C_CONTIGUOUS", "ALIGNED", "WRITEABLE"))
_OUT_F = ndpointer(c_double, flags=("F_CONTIGUOUS", "ALIGNED", "WRITEABLE"))

_SIM = POINTER(CSimulationProperties)
_GL = POINTER(CGaussLegendreProperties)
_RK4 = POINTER(CRK4Options)
_OPTO = POINTER(COptomechanicalProperties)
_DOUBLE_ARRAY_OUT = POINTER(POINTER(c_double))
_COUNT_OUT = POINTER(c_size_t)

# name: (argtypes, restype)
SIGNATURES = {
    "calculateRHSFromFile": ((c_char_p, c_char_p, _SIM, c_size_t), c_int),
    "calculateRHSFromVectors": ((_IN, _IN, _IN, _OUT, _OUT, _OUT, _SIM, c_size_t), c_int),
    "calculateRHSFromVectorsBatched": ((_IN, _IN, _IN, _OUT, _OUT, _OUT, _SIM, c_size_t, c_size_t), c_int),
    "calculateVorticitiesFromVectors": ((_IN_C, _IN_C, _OUT, _OUT_C, _OUT_C, _SIM, c_size_t), c_int),
    "calculateVelocitiesRadialSymmetry": ((_IN, _IN, _IN, _OUT, _OUT, _SIM, c_size_t), c_int),
    "calculateVelocitiesRadialSymmetryModes": ((_IN, _IN, _IN, _OUT, _OUT, _SIM, c_size_t, c_size_t), c_int),
    "calculateDerivativeFFT": ((_IN_C, _OUT_C, c_size_t), c_int),
    "calculateJacobian": ((_IN, _OUT_F, _SIM, c_double, c_size_t), c_int),
    "calculatePerturbedStates": ((_IN, _IN, _IN, _OUT_C, _SIM, c_double, c_size_t), c_int),
    "integrateSimulationGL2": ((_OUT, _DOUBLE_ARRAY_OUT, _COUNT_OUT, _DOUBLE_ARRAY_OUT, _COUNT_OUT, _SIM, _GL, c_size_t), c_int),
    "integrateSimulationGL2_freeMemory": ((POINTER(c_double), POINTER(c_double)), c_int),
    "integrateAugmentedOptomechanicalSimulationRK4": ((_OUT, _DOUBLE_ARRAY_OUT, _COUNT_OUT, _DOUBLE_ARRAY_OUT, _COUNT_OUT, _SIM, _RK4, _OPTO, c_size_t), c_int),
    "integrateAugmentedOptomechanicalSimulationRK4_freeMemory": ((POINTER(c_double), POINTER(c_double)), c_int),
    "calculateRhsAugmentedOptomechanical": ((_OUT, _OUT, _SIM, _OPTO, c_size_t), c_int),
}


def bind(lib: CDLL) -> CDLL:
    """Sets the signature of every export. Raises AttributeError if the DLL lacks one (an older build)."""
    for name, (argtypes, restype) in SIGNATURES.items():
        function = getattr(lib, name)
        function.argtypes = argtypes
        function.restype = restype
    return lib
