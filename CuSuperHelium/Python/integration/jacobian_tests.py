import rhs
from numpy.typing import NDArray
import numpy as np
import tqdm
def cot(a):
    return 1.0/np.tan(a)
def csc_squared(a):
    return 1.0/np.sin(a)**2

def calculate_jacobian_dx_block_velocities(Z : NDArray, Zp : NDArray, Zpp : NDArray, a : NDArray, adx : NDArray, adxp : NDArray, h : float):
    """
    Calculate the Jacobian matrix block for the derivatives with respect to x variables.

    Parameters:
    Z (NDArray): Current state vector.
    Zp (NDArray): First derivative wrt j - discretization variable - of the state vector.
    Zpp (NDArray): Second derivative wrt j - discretization variable - of the state vector.
    adx (NDArray): Vorticities's jacobian (NxN matrix) corresponding to da_k/dx_l for kth row, lth column.
    adxp (NDArray): vorticities's jacobian (NxN matrix) corresponding to (da_k/dx_l)' for kth row, lth column.
    h (float): depth

    Returns:
    NDArray: The Jacobian matrix block for dx.
    """
    n = len(Z)
    jacobian_dx = np.zeros((n, n), dtype=np.complex128)

    for l in tqdm.tqdm(range(n), desc="Calculating Jacobian dx block"):
        for k in range(n):
            jacobian_dx[l, k] += 0.5*adx[l, k]/Zp[k]
            jacobian_dx[l, k] -= 0.25/np.pi*(Zpp[k]/Zp[k]**2*adx[l, k] -2/Zp[k]*adxp[l, k])
            for j in range(n):
                if j != k:
                    jacobian_dx[l, k] += adx[l,j]*cot((Z[k]-Z[j])/2)
                    jacobian_dx[l, k] += adx[l,j]*cot((Z[k] - np.conj(Z[j])+2j*h)/2)
                if l == k and j != k:
                    jacobian_dx[l, k] -= 0.5*a[j]*csc_squared((Z[k] - Z[j])/2)
                if l == k:
                    jacobian_dx[l, k] -= 0.5*a[j]*csc_squared((Z[k] - np.conj(Z[j])+2j*h)/2)
            if l == k:
                jacobian_dx[l, k] += 0.5*a[l]*csc_squared((Z[k]- np.conj(Z[k]) + 2j*h)/2)
            else:
                jacobian_dx[l, k] += 0.5*a[l]*csc_squared((Z[k] - Z[l])/2)
                jacobian_dx[l, k] += 0.5*a[l]*csc_squared((Z[k] - np.conj(Z[l])+2j*h)/2) 
            # finally conjugate to obtain the change in velocity
            jacobian_dx[l, k] = np.conj(jacobian_dx[l, k])
    return jacobian_dx


def calculate_jacobian_vorticities_dx_block(Z : NDArray, Zp : NDArray, Zpp : NDArray, a : NDArray, h : float):
    """
    Calculate the Jacobian matrix block for the derivatives with respect to vorticity variables.

    Parameters:
    Z (NDArray): Current state vector.
    Zp (NDArray): First derivative wrt j - discretization variable - of the state vector.
    Zpp (NDArray): Second derivative wrt j - discretization variable - of the state vector.
    a (NDArray): Current vorticity vector.
    adx (NDArray): Vorticities's jacobian (NxN matrix) corresponding to da_k/dx_l for kth row, lth column.
    adxp (NDArray): vorticities's jacobian (NxN matrix) corresponding to (da_k/dx_l)' for kth row, lth column.
    h (float): depth

    Returns:
    NDArray: The Jacobian matrix block for vorticities.
    """
    n = len(Z)
    jacobian_vorticities = np.zeros((n, n))

    for l in range(n):
        M_matrix = np.zeros((n, n), dtype=np.complex128)
        b_vector = np.zeros(n, dtype=np.complex128)
        for k in range(n):
            for j in range(n):
                if j == k:
                    M_matrix[k, j] = 0.5 + 0.25/np.pi*np.imag(Zpp[k]/Zp[k])
                else:
                    M_matrix[k, j] = np.imag(Zp[k]*cot((Z[k]-Z[j])/2.0))
                M_matrix[k,j] += np.imag(cot(0.5*(Z[k] - np.conj(Z[j]) + 2j*h)))
                if l == k and j != k:
                    b_vector[k] -= a[j]*np.imag(0.5*Zp[k]*csc_squared(0.5*(Z[k]-Z[j])))
                    b_vector[k] -= a[j]*np.imag(0.5*csc_squared(0.5*(Z[k] - np.conj(Z[j]) + 2j*h)))
            if l == k:
                b_vector[k] += a[k]*np.imag(0.5*csc_squared(0.5*(Z[k] - np.conj(Z[k]) + 2j*h)))
            else:
                b_vector[k] += a[l]*np.imag(0.5*Zp[k]*csc_squared(0.5*(Z[k]-Z[l])))
                b_vector[k] += a[l]*np.imag(0.5*csc_squared(0.5*(Z[k] - np.conj(Z[l]) + 2j*h)))
        sol = np.linalg.solve(M_matrix, -b_vector)
        jacobian_vorticities[:, l] = sol
    return jacobian_vorticities

def calculate_jacobian_vorticities_dphi_block(Z : NDArray, Zp : NDArray, Zpp : NDArray, h : float):
    """
    Calculate the Jacobian matrix block for the derivatives with respect to potential variables.

    Parameters:
    Z (NDArray): Current state vector.
    Zp (NDArray): First derivative wrt j - discretization variable - of the state vector.
    Zpp (NDArray): Second derivative wrt j - discretization variable - of the state vector.
    a (NDArray): Current vorticity vector.
    h (float): depth

    Returns:
    NDArray: The Jacobian matrix block for potential.
    """
    n = len(Z)
    jacobian_potential = np.zeros((n, n))

    for l in tqdm.tqdm(range(n), desc="Calculating Jacobian dphi block"):
        M_matrix = np.zeros((n, n), dtype=np.float64)
        b_vector = np.zeros(n, dtype=np.float64)
        for k in range(n):
            for j in range(n):
                if j == k:
                    M_matrix[k, j] = 0.5 + 0.25/np.pi*np.imag(Zpp[k]/Zp[k])
                else:
                    M_matrix[k, j] = np.imag(Zp[k]*cot((Z[k]-Z[j])/2.0))
                M_matrix[k,j] += np.imag(cot(0.5*(Z[k] - np.conj(Z[j]) + 2j*h)))
        sol = np.linalg.solve(M_matrix, -b_vector)
        jacobian_potential[:, l] = sol
    return jacobian_potential

def calculate_jacobian_dphi_block(Z : NDArray, Zp : NDArray, Zpp : NDArray, h : float, adphi : NDArray, adphip : NDArray):
    """
    Calculate the Jacobian matrix block for the derivatives with respect to potential variables.

    Parameters:
    Z (NDArray): Current state vector.
    Zp (NDArray): First derivative wrt j - discretization variable - of the state vector.
    Zpp (NDArray): Second derivative wrt j - discretization variable - of the state vector.
    adphi (NDArray): Vorticities's jacobian (NxN matrix) corresponding to da_k/dphi_l for kth row, lth column.
    adphip (NDArray): vorticities's jacobian (NxN matrix) corresponding to (da_k/dphi_l)' for kth row, lth column.
    h (float): depth

    Returns:
    NDArray: The Jacobian matrix block for potential.
    """
    n = len(Z)
    jacobian_potential = np.zeros((n, n), dtype=np.complex128)

    for l in tqdm.tqdm(range(n), desc="Calculating Jacobian dphi block"):
        for k in range(n):
            jacobian_potential[l, k] += 0.5*adphi[l, k]/Zp[k]
            jacobian_potential[l, k] += 0.25/np.pi*(Zpp[k]/Zp[k]**2*adphi[l, k] -2/Zp[k]*adphip[l, k])
            for j in range(n):
                if j != k:
                    jacobian_potential[l, k] -= 1j/(4*np.pi) * adphi[l,j]*cot((Z[k]-Z[j])/2)
                    jacobian_potential[l, k] += 1j/(4*np.pi) * adphi[l,j]*cot((Z[k] - np.conj(Z[j])+2j*h)/2)
            # finally conjugate to obtain the change in velocity
            jacobian_potential[l, k] = np.conj(jacobian_potential[l, k])
    return jacobian_potential

def calculate_primed_adx(adphi : NDArray):
    adphip = np.empty_like(adphi, dtype=np.complex128)
    for i in range(adphi.shape[1]):
        res, out = rhs.calculate_derivativeFFT256(adphi[:,i].astype(np.complex128))
        # print(out.shape)
        adphip[:,i] = out
    return adphip


if __name__ == "__main__":
    N = 256
    
    depth = 15e-9
    initial_amplitude = 0.01*depth
    L = 1e-6 # 1 micron
    L0 = L / (2.0 * np.pi)
    g = 3*2.6e-24 / depth**4

    
    x = np.array([2.0*np.pi/N*x for x in range(N)])
    ampl = np.cos(x) * initial_amplitude / L0
    pot = initial_amplitude/L0 * np.sin(x) + 1j*np.zeros_like(x)
    Z = x + 1j*ampl

    res, vorticities, Zp, Zpp = rhs.calculate_vorticities256_from_vectors(Z.astype(np.complex128), pot.astype(np.complex128), L, 145, 0, depth)
    # jac_vorticities_dphi = calculate_jacobian_vorticities_dphi_block(Z, Zp, Zpp, depth/L0)


    from matplotlib import pyplot as plt
    # plt.figure()
    # plt.plot(np.real(Z), np.imag(Z), label='initial')
    # plt.figure()
    # # plt.plot(np.real(Z), np.real(pot), label='potential')
    # plt.plot(np.real(x), np.real(vorticities), label='vorticities')
    

    # plt.plot(np.real(x), np.imag(Zp), label="Z' y")
    # plt.plot(np.real(x), np.imag(Zpp), label="Z'' y")

    # # plt.plot(np.real(x), np.real(Zp), label="Z' x")
    # plt.plot(np.real(x), np.real(Zpp), label="Z'' x")

    # plt.legend()

    # plt.figure()
    # plt.contourf(jac_vorticities_dphi)
    # plt.colorbar()

    # plt.show()
    def rhs_voricities(t, y, L = 1e-6, depth = 15e-9):
        N = 256
        local_x = y[:N]
        local_ampl = y[N:2*N]
        local_pot = y[2*N:3*N]
        Z = local_x + 1j*local_ampl
        phi = local_pot.astype(np.complex128)
        res, vorticities, _, _ = rhs.calculate_vorticities256_from_vectors(Z, phi, L, 145, 0, depth)
        if res != 0:
            raise Exception("Error in calculation")
        return vorticities

    def rhs_func(t, y, L = 1e-6, depth = 15e-9):
        N = 256
        local_x = y[:N]
        local_ampl = y[N:2*N]
        local_pot = y[2*N:3*N]
        res, vx, vy, dphi = rhs.calculate_rhs256_from_vectors(local_x, local_ampl, local_pot, L, 145, 0, depth)
        if res != 0:
            raise Exception("Error in calculation")
        return np.concatenate((vx, vy, dphi))
    def g(y):
        return rhs_voricities(0, y, L, depth)
    def f(y):
        return rhs_func(0, y, L, depth)
    # jacobian_dx = (calculate_jacobian_dx_block_velocities(Z, Zp, Zpp, vorticities, np.zeros((N,N)), np.zeros((N,N)), depth/L0))
    # plt.figure()
    # plt.imshow(np.abs(jacobian_dx))
    # # plt.gca().invert_yaxis()
    # plt.colorbar()
    # plt.title("Jacobian dx block abs value")
    # plt.figure()
    # plt.contourf(np.imag(jacobian_dx))
    # # plt.gca().invert_yaxis()
    # plt.colorbar()
    # plt.title("Jacobian dx block (Imaginary)")
    # plt.figure()
    # plt.contourf(np.real(jacobian_dx))
    # # plt.gca().invert_yaxis()
    # plt.colorbar()
    # plt.title("Jacobian dx block (Real)")
    # # plt.show()
    y0 = np.concatenate((x, ampl, pot.astype(np.float64)))
    # jacobian_vorticities = rhs.jacobian_fd(g, y0, 1e-6)
    # primed = calculate_primed_adx(jacobian_vorticities)

    # jac_vort_dphi = jacobian_vorticities[:, 256*2:]
    # primed_dphi = primed[:, 256*2:]

    # jacobian_potential = calculate_jacobian_dphi_block(Z, Zp, Zpp, depth/L0, jac_vort_dphi, primed_dphi)
    jacobian_exp = rhs.jacobian_fd(f, y0, 5.5e-7)
    vals, vecs = np.linalg.eig(jacobian_exp)
    plt.figure()
    plt.plot(np.real(vals), np.imag(vals), 'o')
    plt.title("Eigenvalues of the Jacobian")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.grid()
    # plt.axis("equal")
    # # plt.show()
    # plt.figure()
    # plt.imshow(np.abs(jacobian_vorticities))
    # # plt.gca().invert_yaxis()
    # plt.colorbar()
    # plt.title("Jacobian of vorticities via FD")
    # # plt.title("Difference in Jacobian dx block")
    # plt.figure()
    # plt.imshow(np.abs(primed))
    # plt.colorbar()

    # # plt.figure()
    # # plt.imshow(np.abs(jacobian_potential))
    # # plt.colorbar()
    plt.figure()
    plt.title("Jacobian via FD")
    plt.imshow(np.abs(jacobian_exp))
    plt.colorbar()

    # plt.figure()
    # plt.imshow(np.abs(np.real(jacobian_potential)))
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(np.abs(np.imag(jacobian_potential)))
    # plt.colorbar()

    plt.show()
