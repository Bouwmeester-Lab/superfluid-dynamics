"""Study of Bessel-mode initial conditions with the periodic-2D right-hand side: finite-difference Jacobians,
eigenvalues and RK4 stability. Moved from the bottom of the old integration/rhs.py (Phase 6)."""
import os
from time import sleep

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
import tqdm
from scipy.integrate import solve_ivp

from cusuperhelium import SimulationManager, jacobian_fd, normalize

eps = 1e-6

if __name__ == "__main__":
    alpha_hammaker = 2.6e-24
    simManager = SimulationManager()
    def mode(t, r, r0, R, omega, zeta, A, phase_space=0, phase_time=0):
        
        return A * sp.jv(0, zeta * (r - r0) / R + phase_space) * np.cos(omega * t + phase_time)
    def c3(d):
        return np.sqrt(3 * alpha_hammaker / d**3)
    def angular_freq(zeta, _c3, R):
        return zeta * _c3 / R

    def zeta(n):
        return sp.jn_zeros(1, n)
    def rhs_func(t, y, L = 1e-6, depth = 15e-9, N = 256):
        local_x = y[:N]
        local_ampl = y[N:2*N]
        local_pot = y[2*N:3*N]
        res, vx, vy, dphi = simManager.calculate_rhs_from_vectors(local_x, local_ampl, local_pot, L, 145, 0, depth, N)
        if res != 0:
            raise Exception("Error in calculation")
        return np.concatenate((vx, vy, dphi))
    
    N = 128
    def f(y):
        return rhs_func(0, y, L, depth, N)
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
    initial_amplitude = 0.1*depth
    phase_spaces = np.random.uniform(0, 0.8*np.pi, modes)
    
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


    ## test for best epsilon
    # hs = np.linspace(1e-5, 1e-15, 20)
    # rows = []
    # for h in tqdm.tqdm(hs):
    #     res, J = simManager.calculate_jacobian(r, ampl, pot, L, 145, 0, depth, epsilon=h)
    #     eigenvals = np.linalg.eigvals(J)
    #     r_val = np.max(np.real(eigenvals))
    #     S = 0.5 * (J + J.T)
    #     s = np.linalg.norm(S, "fro")
    #     j = np.linalg.norm(J, "fro")
    #     rows.append((h, r_val, s, j))

    # plt.figure()
    # plt.title("Largest real valued eigenvalue vs epsilon")
    # plt.plot(hs, [row[1] for row in rows], label="max real part")
    # plt.figure()
    # plt.plot(hs, [row[2] / row[3] for row in rows], label="symmetric norm")
    # plt.xlabel("Epsilon")
    # plt.title("Symmetric norm vs epsilon")
    # plt.show()




    res, vx_original, vy_original, dphi_original = simManager.calculate_rhs_from_vectors(r, ampl, pot, L, 145, 0, depth)
    # res, vx, vy, dphi = simManager.calculate_rhs256_from_vectors_batched(x, ampl_batched, pot_batched, L, 145, 0, depth, batchSize)

    if res != 0:
        raise Exception("Error in calculation")
    
    plt.figure()
    plt.plot(r, vx_original, label="vx original")
    plt.plot(r, vy_original, label="vy original")
    plt.plot(r, dphi_original, label="dphi/dt original")
    plt.legend()
    plt.show()
    # plt.figure()
    # for i in range(batchSize):
    #     plt.plot(x[i*N:(i+1)*N], vx[i*N:(i+1)*N] - vx_original, label=f"vx batch {i+1}")
    # plt.figure()
    # for i in range(batchSize):
    #     plt.plot(x[i*N:(i+1)*N], dphi[i*N:(i+1)*N] - dphi_original, label=f"dphi/dt batch {i+1}")
    # plt.figure()
    # for i in range(batchSize):
    #     plt.plot(x[i*N:(i+1)*N], vy[i*N:(i+1)*N] - vy_original, label=f"vy batch {i+1}")
    # # plt.plot(x[:256], vx[:256], label="vx batch 1")
    # # plt.plot(x[256:], vx[256:], label="vx batch 2")


    # plt.legend()
    # plt.show()    

    # for i in range(100):
    res, jacobian = simManager.calculate_jacobian(r, ampl, pot, L, 145, 0, depth, epsilon=1e-8)
        # sleep(1)
    eigvals = np.linalg.eigvals(jacobian)
    print("Eigenvalues of the Jacobian (C++):")
    plt.figure()
    plt.scatter(np.real(eigvals), np.imag(eigvals))
    plt.title("Eigenvalues of the Jacobian (C++)")
    plt.xlabel("Real part")
    plt.ylabel("Imaginary part")
    # plt.show()
    y0 = np.concatenate((r, ampl, pot.astype(np.float64)))
    from time import process_time_ns

    t0 = process_time_ns()
    Jac = jacobian_fd(f, y0, eps = 1e-6)
    t1 = process_time_ns()

    print("Time taken for FD Jacobian - Python: ", (t1 - t0) * 1e-9, " seconds")
    plt.figure()
    plt.imshow(np.abs(jacobian)[:2*N, :2*N])
    plt.colorbar()
    plt.title("Jacobian matrix magnitude (C++)")
    plt.figure()
    plt.imshow(np.abs(Jac)[:2*N, :2*N])
    plt.colorbar()
    plt.title("Jacobian matrix magnitude (FD - Python)")
    # plt.show()

    plt.figure()
    plt.imshow(np.abs(jacobian - Jac)[:2*N, :2*N])
    plt.colorbar()
    plt.title("Jacobian matrix difference magnitude (C++ - FD - Python)")

    plt.figure()
    plt.plot(jacobian[0, :2*N], label="C++ jac at row 0")
    plt.plot(Jac[0, :2*N], label="FD - Python jac at row 0")
    plt.legend()
    plt.figure()
    plt.plot(jacobian[1, :2*N], label="C++ jac at row 1")
    plt.plot(Jac[1, :2*N], label="FD - Python jac at row 1")
    plt.legend()
    plt.figure()
    plt.plot(jacobian[2, :2*N], label="C++ jac at row 2")
    plt.plot(Jac[2, :2*N], label="FD - Python jac at row 2")
    plt.legend()

    plt.figure()
    plt.plot(jacobian[255, :2*N], label="C++ jac at row 255")
    plt.plot(Jac[255, :2*N], label="FD - Python jac at row 255")

    # plt.plot(np.abs(jacobian[255, :] - Jac[255, :]), label="Difference magnitude at row 255")
    plt.legend()
    
    eigs_vals_fd = np.linalg.eigvals(Jac)
    print("Eigenvalues of the Jacobian (FD - Python):")
    plt.figure()
    plt.scatter(np.real(eigs_vals_fd), np.imag(eigs_vals_fd))
    plt.title("Eigenvalues of the Jacobian (FD - Python)")
    plt.xlabel("Real part")
    plt.ylabel("Imaginary part")



    plt.show()
    res, perturbed = simManager.calculate_perturbed_states256(r, ampl, pot, L, 145, 0, depth, eps)
    res, perturbedNeg = simManager.calculate_perturbed_states256(r, ampl, pot, L, 145, 0, depth, -eps)

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
    ej = np.zeros(3*N)
    ej[2] = 1.0
    y1 = y0 + 1e-6 * ej
    y2 = y0 # - 1e-6 * ej

    res, vx1, vy1, dphi1 = simManager.calculate_rhs256_from_vectors(y1[:256], y1[256:512], y1[512:768], L, 145, 0, depth)

    res, vx2, vy2, dphi2 = simManager.calculate_rhs256_from_vectors(y2[:256], y2[256:512], y2[512:768], L, 145, 0, depth)

    approx_vx_fd = (vx1 - vx2) / (eps)
    approx_vy_fd = (vy1 - vy2) / (2.0 * eps)
    approx_dphi_fd = (dphi1 - dphi2) / (2.0 * eps)

    # compare to the batched implementation of rhs256
    y_batched = np.concatenate((y1[:256], y2[:256], y1[256:512], y2[256:512], y1[512:768], y2[512:768]))
    res, vx_batched, vy_batched, dphi_batched = simManager.calculate_rhs256_from_vectors_batched(y_batched[:512], y_batched[512:1024], y_batched[1024:1536], L, 145, 0, depth, 2)

    approx_vx_fd_batched = (vx_batched[:256] - vx_batched[256:]) / ( eps)

    plt.figure()
    plt.plot(y_batched[:256] - y_batched[256:512], label="Batched input amplitude")
    plt.legend()

    plt.figure()
    plt.plot(vx_batched, label="Batched vx output")
    plt.plot(vx_batched[:256])
    plt.plot(vx_batched[256:])
    plt.figure()
    plt.plot(approx_vx_fd, label="Approx vx FD")
    # plt.plot(Jac[:, 2][:256], label="Jacobian vx col 2")
    # plt.plot(jacobian[:, 2][:256], label="C++ Jacobian vx col 2")
    plt.plot(approx_vx_fd_batched, label="Approx vx FD batched")
    plt.legend()
    plt.show()

    res, vx, vy, dphi = simManager.calculate_rhs256_from_vectors_batched(np.real(perturbed[:3*N*N]), np.imag(perturbed[:3*N*N]), np.real(perturbed[3*N*N:]), L, 145, 0, depth, batchSize)  
    if res != 0:
        raise Exception("Error in calculation")
    res, vx_neg, vy_neg, dphi_neg = simManager.calculate_rhs256_from_vectors_batched(np.real(perturbedNeg[:3*N*N]), np.imag(perturbedNeg[:3*N*N]), np.real(perturbedNeg[3*N*N:]), L, 145, 0, depth, batchSize)  
    if res != 0:
        raise Exception("Error in calculation")
    
    # calculate the difference for the jacobian
    approx_vx = (vx - vx_neg) / (2.0 * eps)
    approx_vy = (vy - vy_neg) / (2.0 * eps)
    approx_dphi = (dphi - dphi_neg) / (2.0 * eps)

    plt.figure()
    # we want to transform approx_vx, approx_vy, approx_dphi into a matrix of shape (3N, 3N) representing the jacobian knowing that vx, vy, dphi are (N * batchSize) with then the first N values corresponding to the first perturbation in x1, the second N values to the second perturbation x2, etc.
    plt.title("Jacobian approximation from perturbed states")
    mixed_jacobian = np.zeros((3*N, 3*N))
    mixed_jacobian = np.zeros((3*N, 3*N))

    # Reshape the input arrays as (N, batchSize)
    vx = approx_vx.reshape(N, batchSize, order='F')
    vy = approx_vy.reshape(N, batchSize, order='F')
    dphi = approx_dphi.reshape(N, batchSize, order='F')

    mixed_jacobian[0:N,        :batchSize] = vx
    mixed_jacobian[N:2*N,      :batchSize] = vy
    mixed_jacobian[2*N:3*N,    :batchSize] = dphi
    # for i in range(N*batchSize):
    #     col = i // N
    #     row = i % N
    #     Jacobian[row, col] = approx_vx[i]
    #     Jacobian[row + N, col] = approx_vy[i]
    #     Jacobian[row + 2*N, col] = approx_dphi[i]
    plt.imshow(np.abs(mixed_jacobian))
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
