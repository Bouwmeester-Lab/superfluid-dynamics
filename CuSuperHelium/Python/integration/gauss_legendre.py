import numpy as np
import tqdm
# ---- Gauss-Legendre s=2 coefficients (order 4) ----
sqrt3 = np.sqrt(3.0)
c1 = 0.5 - sqrt3/6.0
c2 = 0.5 + sqrt3/6.0
A  = np.array([[0.25,               0.25 - sqrt3/6.0],
               [0.25 + sqrt3/6.0,   0.25            ]], dtype=float)
b  = np.array([0.5, 0.5], dtype=float)
c  = np.array([c1, c2], dtype=float)


def _pack_blocks(blocks):
    """Pack a 2x2 list of (m x m) blocks into a (2m x 2m) array."""
    B11, B12, B21, B22 = blocks
    m = B11.shape[0]
    M = np.zeros((2*m, 2*m), dtype=B11.dtype)
    M[:m,   :m  ] = B11
    M[:m,   m:  ] = B12
    M[m: ,  :m  ] = B21
    M[m: ,  m:  ] = B22
    return M


def _solve_block_system(I, h, A, J1, J2, rhs):
    """
    Solve (I_2 ⊗ I - h * [a_ij * J_i]) ΔK = rhs
    where block (i,j) = δ_ij I - h a_ij J_i, and J_i is the Jacobian at stage i.
    """
    # Blocks are (m x m). Build the 2x2 block matrix explicitly.
    A11, A12, A21, A22 = A[0,0], A[0,1], A[1,0], A[1,1]
    B11 = I - h * A11 * J1
    B12 =    - h * A12 * J1
    B21 =    - h * A21 * J2
    B22 = I - h * A22 * J2
    M = _pack_blocks([B11, B12, B21, B22])
    return np.linalg.solve(M, rhs)


def gauss_legendre_s2_step(y, h, f, J, newton_tol=1e-10, newton_maxit=12, simplified=False):
    """
    One step of the 2-stage Gauss–Legendre IRK for autonomous ODE y' = f(y).
    
    Parameters
    ----------
    y : (m,) array
        Current state.
    h : float
        Step size.
    f : callable(y) -> (m,)
        RHS function.
    J : callable(y) -> (m,m)
        Jacobian of f at y.
    newton_tol : float
        Stop when ||residual||_2 <= newton_tol * (1 + ||K||_2).
    newton_maxit : int
        Max Newton iterations.
    simplified : bool
        If True, use simplified Newton: freeze J at y (same for both stages).
    
    Returns
    -------
    y_next : (m,) array
    info : dict
        {'nit': iterations, 'converged': bool, 'res_norm': float}
    """
    y = np.asarray(y)
    m = y.size
    
    # Predictor: use f(y) for both stages
    k = np.vstack([f(y), f(y)])   # shape (2, m)

    I = np.eye(m)

    # Precompute frozen Jacobian for simplified Newton
    if simplified:
        Jfrozen = J(y)

    def stage_states(k_):
        # y_i = y + h * sum_j a_ij k_j
        y1 = y + h * (A[0,0]*k_[0] + A[0,1]*k_[1])
        y2 = y + h * (A[1,0]*k_[0] + A[1,1]*k_[1])
        return y1, y2

    # Newton loop
    converged = False
    for it in range(1, newton_maxit+1):
        y1, y2 = stage_states(k)
        # Residuals R_i = k_i - f(y_i)
        R1 = k[0] - f(y1)
        R2 = k[1] - f(y2)
        R  = np.hstack([R1, R2])
        res_norm = np.linalg.norm(R)

        if res_norm <= newton_tol * (1.0 + np.linalg.norm(k)):
            converged = True
            break

        # Build/solve Newton system
        if simplified:
            # Both stage Jacobians use frozen J(y)
            J1 = Jfrozen
            J2 = Jfrozen
        else:
            # Full Newton: stage-dependent Jacobians
            J1 = J(y1)
            J2 = J(y2)

        # Right-hand side is -R
        rhs = -R
        dK  = _solve_block_system(I, h, A, J1, J2, rhs)
        dk1 = dK[:m]
        dk2 = dK[m:]

        # Update stages
        k[0] = k[0] + dk1
        k[1] = k[1] + dk2

    # Step update
    y_next = y + h * (b[0]*k[0] + b[1]*k[1])
    return y_next, {'nit': it, 'converged': converged, 'res_norm': float(res_norm)}


def integrate_gl2(f, J, y0, t0, t1, h, newton_tol=1e-10, newton_maxit=12, simplified=False, return_trajectory=True, **kwargs):
    """
    Fixed-step integrator using Gauss–Legendre s=2.
    Autonomous systems only.

    Returns times and states if return_trajectory=True; otherwise final state.
    """
    assert h > 0.0
    nsteps = int(np.ceil(np.abs(t1 - t0)/h))
    h_eff = (t1 - t0)/nsteps  # adjust to land exactly on t1

    y = np.asarray(y0, dtype=float).copy()
    if return_trajectory:
        T = np.linspace(t0, t1, nsteps+1)
        Y = np.empty((nsteps+1, y.size))
        Y[0] = y

    t = t0
    for n in tqdm.tqdm(range(nsteps), desc="Integrating", unit="step"):
        y, info = gauss_legendre_s2_step(y, h_eff, f, J, newton_tol, newton_maxit, simplified, **kwargs)
        if not info['converged']:
            raise RuntimeError(f"Newton failed at step {n} (t ≈ {t:.6g}): "
                               f"iter={info['nit']} res={info['res_norm']:.3e}")
        t += h_eff
        if return_trajectory:
            Y[n+1] = y

    return (T, Y) if return_trajectory else y

if __name__ == "__main__":
    import rhs
    L = 1e-6
    depth = 15e-9

    def rhs_func(t, y, L = 1e-6, depth = 15e-9):
        N = 256
        local_x = y[:N]
        local_ampl = y[N:2*N]
        local_pot = y[2*N:3*N]
        res, vx, vy, dphi = rhs.calculate_rhs256_from_vectors(local_x, local_ampl, local_pot, L, 145, 0, depth)
        if res != 0:
            raise Exception("Error in calculation")
        return np.concatenate((vx, vy, dphi))
    def f(y):
        return rhs_func(0, y, L, depth)
    
    def J(y):
        N = 256
        local_x = y[:N]
        local_ampl = y[N:2*N]
        local_pot = y[2*N:3*N]
        res, jacobian = rhs.calculate_jacobian(local_x, local_ampl, local_pot, L, 145, 0, depth)
        if jacobian is None or res != 0:
            raise Exception("Error in Jacobian calculation")
        return jacobian
    
    N = 256
    r = np.array([2.0*np.pi/N*x for x in range(N)])
    initial_amplitude = 0.01*depth
    L0 = L / (2.0 * np.pi)

    ampl = np.cos(r) * initial_amplitude / L0
    pot = initial_amplitude/L0 * np.sin(r)

    y0 = np.concatenate((r, ampl, pot))

    t0, t1, h = 0.0, 2000.0, 2
    T, Y = integrate_gl2(f, J, y0, t0, t1, h, simplified=False)
    y_fwd = Y[-1]
    Tback, Yback = integrate_gl2(f, J, y_fwd, t1, t0, h, simplified=False, newton_maxit = 20)  # integrate backward
    rev_err = np.linalg.norm(Yback[-1] - y0)
    print("Reversibility error:", rev_err)

    import matplotlib.pyplot as plt

    x = Y[:, :N]
    x_back = Yback[:, :N]
    ampl_back = Yback[:, N:2*N]

    ampl_end = Y[:, N:2*N]
    pot = Y[:, 2*N:3*N]
    #plt.plot(T, Y[-1, :], label="Amplitude after integration")
    plt.plot(x[-1, :], ampl_end[-1, :], label="Amplitude after integration")
    plt.plot(r, ampl, label="Initial Amplitude")
    plt.plot(x_back[-1, :], ampl_back[-1, :], label="Amplitude after backward integration")
    # plt.plot()
    # plt.plot(x[-1, :], pot[-1, :], label="Potential after integration")
    plt.legend()
    plt.show()
