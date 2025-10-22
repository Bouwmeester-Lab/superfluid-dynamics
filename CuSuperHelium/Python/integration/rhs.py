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
from time import process_time, sleep

paths = os.environ[ 'path' ].split( ';' )
paths.reverse()
# print(paths)

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

def calculate_perturbed_states256(x : NDArray, y: NDArray, phi: NDArray, L : float, rho : float, kappa : float, depth : float, epsilon: float = 1e-6):
    lib = load_dll("D:\\repos\\superfluid-dynamics\\CuSuperHelium\\x64\\Release\\CuSuperHelium.dll")
    lib.calculatePerturbedStates256.argtypes = (ndpointer(c_double, flags=("C_CONTIGUOUS","ALIGNED","WRITEABLE")), ndpointer(c_double, flags=("C_CONTIGUOUS","ALIGNED","WRITEABLE")), ndpointer(c_double, flags=("C_CONTIGUOUS","ALIGNED","WRITEABLE")), ndpointer(np.complex128, flags=("C_CONTIGUOUS","ALIGNED","WRITEABLE")), c_double, c_double, c_double, c_double, c_double)
    lib.calculatePerturbedStates256.restype = c_int

    n = x.shape[0]
    if y.shape[0] != n or phi.shape[0] != n:
        raise ValueError("Input arrays must have the same length")

    Zperturbed = np.ascontiguousarray(np.empty((6 * n * n, ), dtype=np.complex128))
    # print(Zperturbed.flags)
    # print(hex(Zperturbed.ctypes.data))
    # print(Zperturbed.shape)

    return lib.calculatePerturbedStates256(np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(phi), Zperturbed, L, rho, kappa, depth, epsilon), Zperturbed

def calculate_jacobian(x : NDArray, y: NDArray, phi: NDArray, L : float, rho : float, kappa : float, depth : float, epsilon : float = 1e-6):
    lib = load_dll("D:\\repos\\superfluid-dynamics\\CuSuperHelium\\x64\\Release\\CuSuperHelium.dll")
    lib.calculateJacobian.argtypes = (ndpointer(c_double, flags=("C_CONTIGUOUS","ALIGNED")), ndpointer(c_double, flags=("F_CONTIGUOUS","ALIGNED","WRITEABLE")), c_double, c_double, c_double, c_double, c_double, c_size_t)
    lib.calculateJacobian.restype = c_int

    n = x.shape[0]
    if y.shape[0] != n or phi.shape[0] != n:
        raise ValueError("Input arrays must have the same length")
    J = np.empty((3*n, 3*n), dtype=np.float64, order='F')
    return lib.calculateJacobian(np.ascontiguousarray(np.concatenate((x, y, phi))), J, L, rho, kappa, depth, epsilon, n), J

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

    for j in range(m): #, desc="Calculating Jacobian"):
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
    batchSize = 256*3
    r = np.array([2.0*np.pi/N*x for x in range(N)])
    x = np.array([2.0*np.pi/N*x for x in range(N)]*batchSize)
    amplitude = []
    # for i in range(modes):
    #     amplitude.append(mode(0, r, np.pi, 2.0*np.pi, omegas[i], zetas[i], 1, 0, phase_spaces[i]))

    # sum = np.sum(amplitude, axis=0)
    
    ampl_batched = np.cos(x) * initial_amplitude / L0 # sum / np.max(np.abs(sum)) * (initial_amplitude / (L / (2.0*np.pi)))
    ampl = np.cos(r) * initial_amplitude / L0 # sum / np.max(np.abs(sum)) * (initial_amplitude / (L / (2.0*np.pi)))
    

    path = r"./Python/integration/files/"
    filename = "rhs_test.h5"

    Z = r + 1j * ampl
    print(Z.shape)
    # sum = np.zeros_like(x)
    # for i in range(1, 5):
    #     sum += np.sin(i * (x- np.pi))
    # pot = (sum) * initial_amplitude
    pot = initial_amplitude/L0 * np.sin(r)
    pot_batched = initial_amplitude/L0 * np.sin(x)

    for i in range(100):
        res, jacobian = calculate_jacobian(r, ampl, pot, L, 145, 0, depth, epsilon=1e-6)
        # sleep(1)
    
    y0 = np.concatenate((r, ampl, pot.astype(np.float64)))
    from time import process_time_ns

    t0 = process_time_ns()
    Jac = jacobian_fd(f, y0, eps = 1e-6)
    t1 = process_time_ns()

    print("Time taken for FD Jacobian - Python: ", (t1 - t0) * 1e-9, " seconds")
    plt.figure()
    plt.imshow(np.abs(jacobian))
    plt.colorbar()
    plt.title("Jacobian matrix magnitude (C++)")
    plt.figure()
    plt.imshow(np.abs(Jac))
    plt.colorbar()
    plt.title("Jacobian matrix magnitude (FD - Python)")
    # plt.show()

    plt.figure()
    plt.imshow(np.abs(jacobian - Jac))
    plt.colorbar()
    plt.title("Jacobian matrix difference magnitude (C++ - FD - Python)")
    plt.show()
    res, perturbed = calculate_perturbed_states256(r, ampl, pot, L, 145, 0, depth, eps)
    res, perturbedNeg = calculate_perturbed_states256(r, ampl, pot, L, 145, 0, depth, -eps)

    batchSize = perturbed.shape[0] // (2*N)

    perturbed_real = np.real(perturbed[0:256])-x[0:256]
    perturbed_ampl = np.imag(perturbed[0:256])
    potential_perturbed = np.real(perturbed[256:512]) - x[0:256]
    perturbed_x3 = np.real(perturbed[512:768]) - x[0:256]
    perturbed_x4 = np.real(perturbed[768:1024]) - x[0:256]
    perturbed_x5 = np.real(perturbed[1024:1280]) - x[0:256]

    figure1, ax1 = plt.subplots()
    figure2, ax2 = plt.subplots()
    figure3, ax3 = plt.subplots()
    figure4, ax4 = plt.subplots()

    figure5, ax5 = plt.subplots()
    figure6, ax6 = plt.subplots()

    figure7, ax7 = plt.subplots()
    figure8, ax8 = plt.subplots()
    figure9, ax9 = plt.subplots()
    figure10, ax10 = plt.subplots()
    figure11, ax11 = plt.subplots()
    figure12, ax12 = plt.subplots()


    figure1.suptitle("Position X perturbations - comb like")
    figure2.suptitle("Amplitude Y for X perturbations - so zero")

    figure3.suptitle("Position X for Y perturbations - should be zero")
    figure4.suptitle("Amplitude Y perturbations - comb like")

    figure5.suptitle("Position X for potential perturbations - should be zero")
    figure6.suptitle("Amplitude Y for potential perturbations - should be zero")

    figure7.suptitle("Potential Real Part for perturbation on x - should be zero")
    figure8.suptitle("Potential Imaginary Part for perturbation on x - should be zero")

    figure9.suptitle("Potential Real Part for perturbation on y - should be zero")
    figure10.suptitle("Potential Imaginary Part for perturbation on y - should be zero")

    figure11.suptitle("Potential Real Part for perturbation on potential - comb like")
    figure12.suptitle("Potential Imaginary Part for perturbation on potential - should be zero")

    for i in range(3*256):
        perturbed_x = np.real(perturbed[i * N:(i + 1) * N]) - x[0:256]
        perturbed_y = np.imag(perturbed[i * N:(i + 1) * N]) - ampl[0:256]
        
        if i < 256:
            ax1.plot(perturbed_x, label=f"perturbed interface batch {i+1}")
            ax2.plot(perturbed_y, label=f"perturbed interface batch {i+1}")
        elif i < 512:
            ax3.plot(perturbed_x, label=f"perturbed x batch {i+1}")
            ax4.plot(perturbed_y, label=f"perturbed y batch {i+1}")
        elif i < 768:
            ax5.plot(perturbed_x, label=f"perturbed x batch {i+1}")
            ax6.plot(perturbed_y, label=f"perturbed y batch {i+1}")
    for i in range(3*256, 6*256):
        perturbed_x = np.real(perturbed[i * N:(i + 1) * N])- pot[0:256]
        perturbed_y = np.imag(perturbed[i * N:(i + 1) * N])
        if i < 1024:
            ax7.plot(perturbed_x, label=f"perturbed interface batch {i+1}")
            ax8.plot(perturbed_y, label=f"perturbed potential batch {i+1}")
        elif i < 1280:
            ax9.plot(perturbed_x, label=f"perturbed x batch {i+1}")
            ax10.plot(perturbed_y, label=f"perturbed y batch {i+1}")
        elif i < 1536:
            ax11.plot(perturbed_x, label=f"perturbed x batch {i+1}")
            ax12.plot(perturbed_y, label=f"perturbed y batch {i+1}")


            




    # plt.plot(perturbed_real, label="perturbed interface")
    # plt.plot(potential_perturbed, label="perturbed potential")
    # plt.plot(perturbed_x3, label="perturbed x3")
    # plt.plot(perturbed_x4, label="perturbed x4")
    # plt.plot(perturbed_x5, label="perturbed x5")
    # plt.show()

    res, vx, vy, dphi = calculate_rhs256_from_vectors_batched(np.real(perturbed[:3*N*N]), np.imag(perturbed[:3*N*N]), np.real(perturbed[3*N*N:]), L, 145, 0, depth, batchSize)  
    if res != 0:
        raise Exception("Error in calculation")
    res, vx_neg, vy_neg, dphi_neg = calculate_rhs256_from_vectors_batched(np.real(perturbedNeg[:3*N*N]), np.imag(perturbedNeg[:3*N*N]), np.real(perturbedNeg[3*N*N:]), L, 145, 0, depth, batchSize)  
    if res != 0:
        raise Exception("Error in calculation")
    
    # calculate the difference for the jacobian
    approx_vx = (vx - vx_neg) / (2.0 * eps)
    approx_vy = (vy - vy_neg) / (2.0 * eps)
    approx_dphi = (dphi - dphi_neg) / (2.0 * eps)

    plt.figure()
    # we want to transform approx_vx, approx_vy, approx_dphi into a matrix of shape (3N, 3N) representing the jacobian knowing that vx, vy, dphi are (N * batchSize) with then the first N values corresponding to the first perturbation in x1, the second N values to the second perturbation x2, etc.
    plt.title("Jacobian approximation from perturbed states")
    Jacobian = np.zeros((3*N, 3*N))
    Jacobian = np.zeros((3*N, 3*N))

    # Reshape the input arrays as (N, batchSize)
    vx = approx_vx.reshape(N, batchSize, order='F')
    vy = approx_vy.reshape(N, batchSize, order='F')
    dphi = approx_dphi.reshape(N, batchSize, order='F')

    Jacobian[0:N,        :batchSize] = vx
    Jacobian[N:2*N,      :batchSize] = vy
    Jacobian[2*N:3*N,    :batchSize] = dphi
    # for i in range(N*batchSize):
    #     col = i // N
    #     row = i % N
    #     Jacobian[row, col] = approx_vx[i]
    #     Jacobian[row + N, col] = approx_vy[i]
    #     Jacobian[row + 2*N, col] = approx_dphi[i]
    plt.imshow(np.abs(Jacobian))
    plt.colorbar()
    plt.show()

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
    plt.plot(x[256:], ampl_batched[256:], label="initial interface")
    # plt.axhline(-depth*2.0*np.pi/L, color='r', linestyle='--', label="equilibrium height")
    plt.plot(x[256:], pot_batched[256:], label="initial potential")

    plt.legend()
    plt.show()

    t0 = np.sqrt(L0 /g)
    print("Characteristic time: ", t0)
    tend = 1e-3 #ms
    print("End time: ", tend)
    print("Number of characteristic times: ", tend/t0)

    # pot = pot.astype(np.complex128)
    print(pot[0])

    y0 = np.concatenate((r, ampl, pot.astype(np.float64)))
    # hs = np.linspace(1e-5, 1e-15, 20)
    # rows = []
    # for h in hs:
    #     J = jacobian_fd(f, y0, eps = h)
    #     eigenvals = np.linalg.eigvals(J)
    #     r = np.max(np.real(eigenvals))
    #     S = 0.5 * (J + J.T)
    #     s = np.linalg.norm(S, "fro")
    #     j = np.linalg.norm(J, "fro")

    #     rows.append((h, r, s, j))

    # plt.figure()
    # plt.title("Largest real valued eigenvalue vs epsilon")
    # plt.plot(hs, [row[1] for row in rows], label="max real part")
    # plt.figure()
    # plt.plot(hs, [row[2] / row[3] for row in rows], label="symmetric norm")
    # plt.xlabel("Epsilon")
    # plt.title("Symmetric norm vs epsilon")
    # plt.show()
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
    zero_eigenvalue = np.isclose(np.abs(vals), 0, atol=1e-3)
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