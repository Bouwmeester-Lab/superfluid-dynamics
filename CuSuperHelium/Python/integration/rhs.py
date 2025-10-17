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

from scipy.integrate import solve_ivp
from scipy.sparse.linalg import eigs, LinearOperator
import tqdm

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

    return lib.calculateRHS256FromVectors(np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(phi), vx, vy, dphi, L, rho, kappa, depth), vx, vy, dphi

def calculate_rhs2048_from_vectors(x : NDArray, y: NDArray, phi: NDArray, L : float, rho : float, kappa : float, depth : float):
    lib = load_dll("D:\\repos\\superfluid-dynamics\\CuSuperHelium\\x64\\Release\\CuSuperHelium.dll")
    lib.calculateRHS2048FromVectors.argtypes = (ndpointer(c_double, flags=("C_CONTIGUOUS","ALIGNED","WRITEABLE")), ndpointer(c_double, flags=("C_CONTIGUOUS","ALIGNED","WRITEABLE")), ndpointer(c_double, flags=("C_CONTIGUOUS","ALIGNED","WRITEABLE")), ndpointer(c_double, flags=("C_CONTIGUOUS","ALIGNED","WRITEABLE")), ndpointer(c_double, flags=("C_CONTIGUOUS","ALIGNED","WRITEABLE")), ndpointer(c_double, flags=("C_CONTIGUOUS","ALIGNED","WRITEABLE")), c_double, c_double, c_double, c_double)
    lib.calculateRHS2048FromVectors.restype = c_int

    n = x.shape[0]
    if y.shape[0] != n or phi.shape[0] != n:
        raise ValueError("Input arrays must have the same length")

    vx = np.empty_like(x)
    vy = np.empty_like(x)
    dphi = np.empty_like(x)

    return lib.calculateRHS2048FromVectors(np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(phi), vx, vy, dphi, L, rho, kappa, depth), vx, vy, dphi

class CDouble(Structure):
    _fields_ = [("re", c_double), ("im", c_double)]

def calculate_vorticities256_from_vectors(Z : NDArray, phi: NDArray, L : float, rho : float, kappa : float, depth : float):
    lib = load_dll("D:\\repos\\superfluid-dynamics\\CuSuperHelium\\x64\\Release\\CuSuperHelium.dll")
    lib.calculateVorticities256FromVectors.argtypes = (ndpointer(np.complex128, flags=("C_CONTIGUOUS","ALIGNED")), ndpointer(np.complex128, flags=("C_CONTIGUOUS","ALIGNED")), ndpointer(c_double, flags=("C_CONTIGUOUS","ALIGNED","WRITEABLE")), ndpointer(np.complex128, flags=("C_CONTIGUOUS","ALIGNED", "WRITEABLE")), ndpointer(np.complex128, flags=("C_CONTIGUOUS","ALIGNED", "WRITEABLE")), c_double, c_double, c_double, c_double)
    lib.calculateVorticities256FromVectors.restype = c_int

    n = Z.shape[0]
    if phi.shape[0] != n:
        raise ValueError("Input arrays must have the same length")

    vorticity = np.empty(n, dtype=np.float64)
    Zp = np.empty(n, dtype=np.complex128)
    Zpp = np.empty(n, dtype=np.complex128)
    return lib.calculateVorticities256FromVectors(np.ascontiguousarray(Z, dtype=np.complex128), np.ascontiguousarray(phi, dtype=np.complex128), vorticity, Zp, Zpp, L, rho, kappa, depth), vorticity, Zp, Zpp


def calculate_derivativeFFT256(input : NDArray) -> tuple[int, NDArray]:
    lib = load_dll("D:\\repos\\superfluid-dynamics\\CuSuperHelium\\x64\\Release\\CuSuperHelium.dll")
    lib.calculateDerivativeFFT256.argtypes = (ndpointer(np.complex128, flags=("C_CONTIGUOUS","ALIGNED")), ndpointer(np.complex128, flags=("C_CONTIGUOUS","ALIGNED","WRITEABLE")))
    lib.calculateDerivativeFFT256.restype = c_int

    n = input.shape[0]
    output = np.empty(n, dtype=np.complex128)
    if output.shape[0] != n:
        raise ValueError("Input and output arrays must have the same length")

    return lib.calculateDerivativeFFT256(np.ascontiguousarray(input), output), output

def calculate_rhs256_from_vectors_batched(x : NDArray, y: NDArray, phi: NDArray, L : float, rho : float, kappa : float, depth : float, batch_size: int):
    lib = load_dll("D:\\repos\\superfluid-dynamics\\CuSuperHelium\\x64\\Release\\CuSuperHelium.dll")
    lib.calculateRHS256FromVectorsBatched.argtypes = (ndpointer(c_double, flags=("C_CONTIGUOUS","ALIGNED","WRITEABLE")), ndpointer(c_double, flags=("C_CONTIGUOUS","ALIGNED","WRITEABLE")), ndpointer(c_double, flags=("C_CONTIGUOUS","ALIGNED","WRITEABLE")), ndpointer(c_double, flags=("C_CONTIGUOUS","ALIGNED","WRITEABLE")), ndpointer(c_double, flags=("C_CONTIGUOUS","ALIGNED","WRITEABLE")), ndpointer(c_double, flags=("C_CONTIGUOUS","ALIGNED","WRITEABLE")), c_double, c_double, c_double, c_double, c_int)
    lib.calculateRHS256FromVectorsBatched.restype = c_int

    n = x.shape[0] // batch_size
    if y.shape[0] != n * batch_size or phi.shape[0] != 256 * batch_size:
        raise ValueError("Input arrays must have the same length equal to 256 * batch_size")

    vx = np.empty_like(x)
    vy = np.empty_like(x)
    dphi = np.empty_like(x)

    return lib.calculateRHS256FromVectorsBatched(np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(phi), vx, vy, dphi, L, rho, kappa, depth, batch_size), vx, vy, dphi


# with h5py.File(os.path.join(path, filename), 'w') as f:
#     dset = f.create_dataset("interface", data=Z, shape=(256, 2), dtype=np.float64)
#     potential = f.create_dataset("potential", data=pot, shape=(256,), dtype=np.float64)

def normalize(v):
    return v / np.max(np.abs(v))


eps = 1e-6


def jacobian_fd(f, y0, eps=None, scales=None):
    """
    Dense Jacobian via central differences using scalar f: (n,) -> (n,)
    """
    f0 = f(y0)
    n = f0.shape[0] # output dimension
    m = y0.shape[0] # input dimension

    if eps is None:
        h0 = np.sqrt(np.finfo(float).eps) * (1.0 + np.abs(y0))
    else:
        h0 = np.full(m, float(eps))
    if scales is not None:
        s = np.asarray(scales, dtype=float)
        h0 = h0 * np.maximum(1.0, np.abs(s))

    J = np.empty((n, m), dtype=float) # matrix should be output dimension x input dimension
    # f0 = f(y0)  # not strictly needed, but useful for sanity checks

    for j in tqdm.tqdm(range(m), desc="Calculating Jacobian"):
        ej = np.zeros(m); ej[j] = 1.0
        hj = h0[j]
        Fp = f(y0 + hj*ej)
        Fm = f(y0 - hj*ej)
        J[:, j] = (Fp - Fm) / (2.0*hj)
    return J
def richardson_extrapolation(J1, J2, factor=4):
    return (factor * J2 - J1) / (factor - 1)

def richardson_jacobian(f, y0, h):
    J1 = jacobian_fd(f, y0, h)
    J2 = jacobian_fd(f, y0, h * 0.5)
    return richardson_extrapolation(J1, J2)

if __name__ == "__main__":
    alpha_hammaker = 2.6e-24
    def mode(t, r, r0, R, omega, zeta, A, phase_space=0, phase_time=0):
        
        return A * sp.jv(0, zeta * (r - r0) / R + phase_space) * np.cos(omega * t + phase_time)
    def c3(d):
        return np.sqrt(3 * alpha_hammaker / d**3)
    def angular_freq(zeta, _c3, R):
        return zeta * _c3 / R

    def zeta(n):
        return sp.jn_zeros(1, n)
    def rhs_func(t, y, L = 1e-6, depth = 15e-9):
        N = 256
        local_x = y[:N]
        local_ampl = y[N:2*N]
        local_pot = y[2*N:3*N]
        res, vx, vy, dphi = calculate_rhs256_from_vectors(local_x, local_ampl, local_pot, L, 145, 0, depth)
        if res != 0:
            raise Exception("Error in calculation")
        return np.concatenate((vx, vy, dphi))
    def f(y):
        return rhs_func(0, y, L, depth)
    H0 = 15e-9 # 15 nm
    L = 1e-6 # 1 micron
    L0 = L / (2.0 * np.pi)
    g = 3*2.6e-24 / H0**4
    modes = 20

    zetas = zeta(modes)
    R = 2*np.pi
    speed = c3(H0)

    omegas = angular_freq(zetas, speed, R)
    depth = 15e-9
    initial_amplitude = 0.01*depth
    phase_spaces = np.random.uniform(0, 0.8*np.pi, modes)
    N = 256
    r = np.array([2.0*np.pi/N*x for x in range(N)]*2)
    x = np.array([2.0*np.pi/N*x for x in range(N)]*2)
    amplitude = []
    for i in range(modes):
        amplitude.append(mode(0, r, np.pi, 2.0*np.pi, omegas[i], zetas[i], 1, 0, phase_spaces[i]))

    sum = np.sum(amplitude, axis=0)
    
    ampl = np.cos(x) * initial_amplitude / L0 # sum / np.max(np.abs(sum)) * (initial_amplitude / (L / (2.0*np.pi)))

    

    path = r"./Python/integration/files/"
    filename = "rhs_test.h5"

    Z = x + 1j * ampl
    print(Z.shape)
    sum = np.zeros_like(x)
    for i in range(1, 5):
        sum += np.sin(i * (x- np.pi))
    # pot = (sum) * initial_amplitude
    pot = initial_amplitude/L0 * np.sin(x)

    res, vx, vy, dphi = calculate_rhs256_from_vectors_batched(x, ampl, pot, L, 145, 0, depth, 2)

    if res != 0:
        raise Exception("Error in calculation")
    print("Calculation successful")
    plt.figure()
    plt.plot(x[:256], vx[:256], label="vx batch 1")
    plt.plot(x[256:], vx[256:], label="vx batch 2")
    plt.legend()
    plt.figure()
    plt.plot(x[:256], vy[:256], label="vy batch 1")
    plt.plot(x[256:], vy[256:], label="vy batch 2")
    plt.legend()
    plt.figure()
    plt.plot(x[:256], dphi[:256], label="dphi/dt batch 1")  
    plt.plot(x[256:], dphi[256:], label="dphi/dt batch 2")
    plt.legend()
    # plt.show()

    plt.figure()
    plt.plot(x[256:], ampl[256:], label="initial interface")
    # plt.axhline(-depth*2.0*np.pi/L, color='r', linestyle='--', label="equilibrium height")
    plt.plot(x[256:], pot[256:], label="initial potential")

    plt.legend()
    plt.show()

    t0 = np.sqrt(L0 /g)
    print("Characteristic time: ", t0)
    tend = 1e-3 #ms
    print("End time: ", tend)
    print("Number of characteristic times: ", tend/t0)

    # pot = pot.astype(np.complex128)
    print(pot[0])

    y0 = np.concatenate((x, ampl, pot.astype(np.float64)))
    hs = np.linspace(1e-5, 1e-15, 20)
    rows = []
    for h in hs:
        J = jacobian_fd(f, y0, eps = h)
        eigenvals = np.linalg.eigvals(J)
        r = np.max(np.real(eigenvals))
        S = 0.5 * (J + J.T)
        s = np.linalg.norm(S, "fro")
        j = np.linalg.norm(J, "fro")

        rows.append((h, r, s, j))

    plt.figure()
    plt.title("Largest real valued eigenvalue vs epsilon")
    plt.plot(hs, [row[1] for row in rows], label="max real part")
    plt.figure()
    plt.plot(hs, [row[2] / row[3] for row in rows], label="symmetric norm")
    plt.xlabel("Epsilon")
    plt.title("Symmetric norm vs epsilon")
    plt.show()
    # print("Calculating Jacobian...")
    Jac = jacobian_fd(f, y0, eps = 1e-6)
    # Jac2 = jacobian_fd(f, y0, h = eps *1e-2)
    print("Finished calculating Jacobian")
    # print(Jac.shape)
    print("Calculating eigenvalues...")
    vals, vecs = np.linalg.eig(Jac)
    # vals2, vecs2 = np.linalg.eig(Jac2)
    # print("Eigenvalues:")
    # for v in vals:
    #     print(v)
    ### get all the eigenvalues with zero real part and non zero imaginary part
    imaginary_only_eigenvalues = np.all(np.isclose(np.real(vals), 0, atol=1e-5)) & ~np.isclose(np.imag(vals), 0, atol=1e-5)
    ### get all the eigenvalues with zero real part and zero imaginary part
    zero_eigenvalue = np.isclose(np.abs(vals), 0, atol=1e-5)
    print(f"There's {len(vals[zero_eigenvalue])} zero eigenvalues.")
    print(f"{len(vals[imaginary_only_eigenvalues])} eigenvalues are purely imaginary.")
    print(f" {len(vals) - len(vals[imaginary_only_eigenvalues]) - len(vals[zero_eigenvalue])} eigenvalues have a real part.")

    def max_allowed_eigenvalue_for_stability_rk4(timestep: float):
        return 2.83 / timestep

    # ## plot the eigen values in the complex plane
    plt.figure()
    plt.scatter(np.real(vals[imaginary_only_eigenvalues]), np.imag(vals[imaginary_only_eigenvalues]), color='g', label="Purely imaginary eigenvalues")
    plt.axhline(max_allowed_eigenvalue_for_stability_rk4(0.1), color='g', linestyle='--', label="Max allowed eigenvalue for stability (dt=0.1)")
    plt.axhline(max_allowed_eigenvalue_for_stability_rk4(1), color='orange', linestyle='--', label="Max allowed eigenvalue for stability (dt=1)")
    plt.axhline(max_allowed_eigenvalue_for_stability_rk4(10), color='red', linestyle='--', label="Max allowed eigenvalue for stability (dt=10)")
    plt.legend()
    # plt.scatter(np.real(vals2), np.imag(vals2), color='r')
    plt.xlabel("Real part")
    plt.ylabel("Imaginary part")
    plt.title("Imaginary Eigenvalues of the Jacobian at the sweet epsilon point compared to RK4 stability limits")
    plt.show()

# res = solve_ivp(rhs, (0, 5000), np.concatenate((x, ampl, pot.astype(np.float64))), method='DOP853', atol=1e-5, rtol=1e-3)

# print(res)
# plt.figure()
# plt.plot(x, ampl, label="initial interface")
# plt.plot(res.y[:N, -1], res.y[N:2*N, -1], label="final interface")
# plt.show()
# res, a = calculate_vorticities256_from_vectors(Z, pot, L, 145, 0, depth)
# res2, a2 = calculate_vorticities256_from_vectors(np.real(Z) + 1j * (np.imag(Z) + 1e-2), pot, L, 145, 0, depth)

# if res != 0 and res2 != 0:
#     raise Exception("Error in calculation")

# # with h5py.File(os.path.join(path, "data.h5"), 'r') as f:
# #     velocities = f["interface"][:]
# #     potential = f["potential"][:]
# print(a.shape)
# plt.figure()
# # plt.plot(x, normalize(pot), label="initial potential - normalized")
# plt.plot(x, a2-a, label="difference in vorticity - normalized")

# # plt.legend()
# # plt.figure()
# # plt.plot(x, normalize(ampl), label="initial interface - normalized")
# # plt.plot(x, normalize(dphi), label="change in potential (dPhi/dt = rhs Phi) - normalized")

# # plt.legend()

# plt.show()