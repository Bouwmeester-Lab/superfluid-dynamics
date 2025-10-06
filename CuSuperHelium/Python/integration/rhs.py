import ctypes, os, sys
from ctypes import c_double, c_size_t, c_char_p, c_char, POINTER, create_string_buffer, c_int
import scipy.special as sp
import numpy as np
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
    lib.calculateRHS256.argtypes = (c_char_p, c_char_p, c_double, c_double, c_double, c_double)
    lib.calculateRHS256.restype = c_int

    input_file_b = input_file.encode()
    output_file_b = output_file.encode()

    return lib.calculateRHS256(input_file_b, output_file_b, L, rho, kappa, depth)


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
ampl = sum / np.max(np.abs(sum)) * initial_amplitude
x = np.array([2.0*np.pi/256*x for x in range(256)])

path = r"./Python/integration/files/"
filename = "rhs_test.h5"

Z = np.vstack((x, ampl)).T
print(Z.shape)
sum = np.zeros_like(x)
for i in range(1, 5000):
    sum += np.sin(i * (x- np.pi))/np.sin(x - np.pi)
pot = (sum) * initial_amplitude
with h5py.File(os.path.join(path, filename), 'w') as f:
    dset = f.create_dataset("interface", data=Z, shape=(256, 2), dtype=np.float64)
    potential = f.create_dataset("potential", data=pot, shape=(256,), dtype=np.float64)

def normalize(v):
    return v / np.max(np.abs(v))

if calculate_rhs256(os.path.join(path, filename), os.path.join(path, "data.h5"), 1e-6, 145, 0, depth) != 0:
    raise Exception("Error in calculation")

with h5py.File(os.path.join(path, "data.h5"), 'r') as f:
    velocities = f["interface"][:]
    potential = f["potential"][:]
print(velocities.shape)
plt.figure()
plt.plot(x, normalize(pot), label="initial potential - normalized")
plt.plot(x, normalize(velocities[:, 0]), label="vx - normalized")
plt.plot(x, normalize(velocities[:, 1]), label="vy - normalized")

plt.legend()
plt.figure()
plt.plot(x, normalize(ampl), label="initial interface - normalized")
plt.plot(x, normalize(potential), label="change in potential (dPhi/dt = rhs Phi) - normalized")

plt.legend()

plt.show()