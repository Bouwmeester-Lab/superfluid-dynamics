import ctypes, os, sys
from ctypes import c_double, c_size_t, c_char_p, c_char, POINTER, create_string_buffer, c_int, Structure
import scipy.special as sp
import numpy as np
from numpy.typing import NDArray
from numpy.ctypeslib import ndpointer
import matplotlib.pyplot as plt
import h5py
import os
if os.name != "nt":
    raise Exception("Invalid OS this code working only on NT(Windows)")
from ctypes import CDLL # this is normal module
from _ctypes import LoadLibrary # we are import back-end ctypes module
paths = os.environ[ 'path' ].split( ';' )
paths.reverse()
print(paths)

for p in paths:
    if os.path.isdir( p ) and p != ".":
        os.add_dll_directory(p)
os.add_dll_directory( r"D:\repos\superfluid-dynamics\CuSuperHelium\x64\Release" )
os.add_dll_directory(os.getcwd())

def load_dll(path: str) -> CDLL:
    if path == "" or path is None:
        raise TypeError("Please fill valid path")
    # handle = LoadLibrary(path)
    # return ctypes.CDLL(None, handle=handle)
    return ctypes.CDLL(path)


def calculate_rhs256(input_file : str, output_file : str, L : float, rho : float, kappa : float, depth : float) -> int:
    lib = load_dll("D:\\repos\\superfluid-dynamics\\CuSuperHelium\\x64\\Release\\CuSuperHelium.dll")
    lib.calculateRHS256FromFile.argtypes = (c_char_p, c_char_p, c_double, c_double, c_double, c_double)
    lib.calculateRHS256FromFile.restype = c_int

    input_file_b = input_file.encode()
    output_file_b = output_file.encode()

    return lib.calculateRHS256FromFile(input_file_b, output_file_b, L, rho, kappa, depth)

def calculate_rhs256_from_vectors(x : NDArray, y: NDArray, phi: NDArray, L : float, rho : float, kappa : float, depth : float):
    lib = load_dll("D:\\repos\\superfluid-dynamics\\CuSuperHelium\\x64\\Release\\CuSuperHelium.dll")
    lib.calculateRHS256FromVectors.argtypes = (ndpointer(c_double, flags=("C_CONTIGUOUS","ALIGNED","WRITEABLE")), ndpointer(c_double, flags=("C_CONTIGUOUS","ALIGNED","WRITEABLE")), ndpointer(c_double, flags=("C_CONTIGUOUS","ALIGNED","WRITEABLE")), ndpointer(c_double, flags=("C_CONTIGUOUS","ALIGNED","WRITEABLE")), ndpointer(c_double, flags=("C_CONTIGUOUS","ALIGNED","WRITEABLE")), ndpointer(c_double, flags=("C_CONTIGUOUS","ALIGNED","WRITEABLE")), c_double, c_double, c_double, c_double)
    lib.calculateRHS256FromVectors.restype = c_int

    n = x.shape[0]
    if y.shape[0] != n or phi.shape[0] != n:
        raise ValueError("Input arrays must have the same length")

    vx = np.empty_like(x)
    vy = np.empty_like(x)
    dphi = np.empty_like(x)

    return lib.calculateRHS256FromVectors(x, y, phi, vx, vy, dphi, L, rho, kappa, depth), vx, vy, dphi

class CDouble(Structure):
    _fields_ = [("re", c_double), ("im", c_double)]

def calculate_vorticities256_from_vectors(Z : NDArray, phi: NDArray, L : float, rho : float, kappa : float, depth : float):
    lib = load_dll("D:\\repos\\superfluid-dynamics\\CuSuperHelium\\x64\\Release\\CuSuperHelium.dll")
    lib.calculateVorticities256FromVectors.argtypes = (ndpointer(np.complex128, flags=("C_CONTIGUOUS","ALIGNED")), ndpointer(np.complex128, flags=("C_CONTIGUOUS","ALIGNED")), ndpointer(c_double, flags=("C_CONTIGUOUS","ALIGNED","WRITEABLE")), c_double, c_double, c_double, c_double)
    lib.calculateVorticities256FromVectors.restype = c_int

    n = Z.shape[0]
    if phi.shape[0] != n:
        raise ValueError("Input arrays must have the same length")

    vorticity = np.empty(n, dtype=np.float64)

    return lib.calculateVorticities256FromVectors(np.ascontiguousarray(Z, dtype=np.complex128), np.ascontiguousarray(phi, dtype=np.complex128), vorticity, L, rho, kappa, depth), vorticity

alpha_hammaker = 2.6e-24
def mode(t, r, r0, R, omega, zeta, A, phase_space=0, phase_time=0):
    
    return A * sp.jv(0, zeta * (r - r0) / R + phase_space) * np.cos(omega * t + phase_time)
def c3(d):
    return np.sqrt(3 * alpha_hammaker / d**3)
def angular_freq(zeta, _c3, R):
    return zeta * _c3 / R

def zeta(n):
    return sp.jn_zeros(1, n)


H0 = 15e-9 # 15 nm
L = 1e-6 # 1 micron
modes = 20

zetas = zeta(modes)
R = 2*np.pi
speed = c3(H0)

omegas = angular_freq(zetas, speed, R)
depth = 15e-9
initial_amplitude = 0.01*depth
phase_spaces = np.random.uniform(0, 0.8*np.pi, modes)

r = np.array([2.0*np.pi/256*x for x in range(256)])

amplitude = []
for i in range(modes):
    amplitude.append(mode(0, r, np.pi, 2.0*np.pi, omegas[i], zetas[i], 1, 0, phase_spaces[i]))

sum = np.sum(amplitude, axis=0)
ampl = sum / np.max(np.abs(sum)) * initial_amplitude * L / (2.0*np.pi)
x = np.array([2.0*np.pi/256*x for x in range(256)])

path = r"./Python/integration/files/"
filename = "rhs_test.h5"

Z = x + 1j * ampl
print(Z.shape)
sum = np.zeros_like(x)
for i in range(1, 5):
    sum += np.sin(i * (x- np.pi))
pot = (sum) * initial_amplitude

pot = pot.astype(np.complex128)
print(pot[0])

# with h5py.File(os.path.join(path, filename), 'w') as f:
#     dset = f.create_dataset("interface", data=Z, shape=(256, 2), dtype=np.float64)
#     potential = f.create_dataset("potential", data=pot, shape=(256,), dtype=np.float64)

def normalize(v):
    return v / np.max(np.abs(v))

res, a = calculate_vorticities256_from_vectors(Z, pot, L, 145, 0, depth)
res2, a2 = calculate_vorticities256_from_vectors(np.real(Z) + 1j * (np.imag(Z) + 1e-2), pot, L, 145, 0, depth)

if res != 0 and res2 != 0:
    raise Exception("Error in calculation")

# with h5py.File(os.path.join(path, "data.h5"), 'r') as f:
#     velocities = f["interface"][:]
#     potential = f["potential"][:]
print(a.shape)
plt.figure()
# plt.plot(x, normalize(pot), label="initial potential - normalized")
plt.plot(x, a2-a, label="difference in vorticity - normalized")

# plt.legend()
# plt.figure()
# plt.plot(x, normalize(ampl), label="initial interface - normalized")
# plt.plot(x, normalize(dphi), label="change in potential (dPhi/dt = rhs Phi) - normalized")

# plt.legend()

plt.show()