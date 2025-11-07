import numpy as np
from dataclasses import dataclass
from tqdm import tqdm

# ---- Gauss-Legendre s=2 coefficients (order 4) ----
sqrt3 = np.sqrt(3.0)
c1 = 0.5 - sqrt3 / 6.0
c2 = 0.5 + sqrt3 / 6.0
A  = np.array([[0.25,               0.25 - sqrt3/6.0],
               [0.25 + sqrt3/6.0,   0.25            ]], dtype=float)
b  = np.array([0.5, 0.5], dtype=float)
c  = np.array([c1, c2], dtype=float)


@dataclass
class NewtonOptions:
    tol: float = 1e-10
    maxit: int = 12
    # line search parameters
    armijo_c: float = 1e-4     # sufficient decrease constant
    backtrack: float = 0.5     # step reduction factor
    min_alpha: float = 1e-6    # stop backtracking threshold
    # behavior
    allow_simplified_fallback: bool = True


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


def _solve_block_system(I, h, Acoef, J1, J2, rhs):
    """
    Solve (I_2 ⊗ I - h * [a_ij * J_i]) ΔK = rhs
    where block (i,j) = δ_ij I - h a_ij J_i, and J_i is the Jacobian at stage i.
    """
    A11, A12, A21, A22 = Acoef[0,0], Acoef[0,1], Acoef[1,0], Acoef[1,1]
    B11 = I - h * A11 * J1
    B12 =    - h * A12 * J1
    B21 =    - h * A21 * J2
    B22 = I - h * A22 * J2
    M = _pack_blocks([B11, B12, B21, B22])
    return np.linalg.solve(M, rhs)


def gauss_legendre_s2_step(y, h, f, J, newton: NewtonOptions):
    """
    One step of the 2-stage Gauss–Legendre IRK for autonomous ODE y' = f(y).

    Returns
    -------
    y_next : (m,) array
    info : dict
        {'nit': iterations, 'converged': bool, 'res_norm': float, 'simplified_used': bool}
    """
    y = np.asarray(y)
    m = y.size
    I = np.eye(m)

    # Predictor: both stages start from f(y)
    k = np.vstack([f(y), f(y)])   # shape (2, m)

    def stage_states(k_):
        # y_i = y + h * sum_j a_ij k_j
        y1 = y + h * (A[0,0]*k_[0] + A[0,1]*k_[1])
        y2 = y + h * (A[1,0]*k_[0] + A[1,1]*k_[1])
        return y1, y2

    # merit function: 0.5 * ||R||^2
    def residual_and_phi(k_):
        y1, y2 = stage_states(k_)
        R1 = k_[0] - f(y1)
        R2 = k_[1] - f(y2)
        R  = np.hstack([R1, R2])
        phi = 0.5 * np.dot(R, R)
        return R, phi, y1, y2

    # ---- Newton loop (with fallback + line search) ----
    simplified_used = False
    converged = False

    # Try full Newton first
    freeze_J = False
    J_frozen = None

    for it in range(1, newton.maxit + 1):
        R, phi, y1, y2 = residual_and_phi(k)
        res_norm = np.sqrt(2.0 * phi)
        if res_norm <= newton.tol * (1.0 + np.linalg.norm(k)):
            converged = True
            break

        if freeze_J:
            J1 = J_frozen
            J2 = J_frozen
        else:
            J1 = J(y1)
            J2 = J(y2)

        # Solve for Newton direction: (Jacobian of residual) dK = -R
        rhs = -R
        dK  = _solve_block_system(I, h, A, J1, J2, rhs)
        dk1 = dK[:m]
        dk2 = dK[m:]

        # Armijo backtracking on phi(k)
        alpha = 1.0
        # directional derivative surrogate: we use ||R||^2 decrease ~ newton_cauchy
        # A safe/proven Armijo condition: phi(k + a d) <= phi(k) - c * a * ||R||^2
        # (This is conservative but very robust for IRK residuals.)
        target = phi - newton.armijo_c * alpha * (res_norm**2)

        while True:
            k_trial = np.empty_like(k)
            k_trial[0] = k[0] + alpha * dk1
            k_trial[1] = k[1] + alpha * dk2
            R_trial, phi_trial, *_ = residual_and_phi(k_trial)

            if phi_trial <= target:
                k = k_trial
                break
            alpha *= newton.backtrack
            target = phi - newton.armijo_c * alpha * (res_norm**2)
            if alpha < newton.min_alpha:
                # Line-search failed: optionally switch to simplified Newton once
                if (not freeze_J) and newton.allow_simplified_fallback:
                    freeze_J = True
                    simplified_used = True
                    J_frozen = J(y)  # chord Newton about the step base point
                    # restart iteration counter softly (don’t discard progress)
                    break
                else:
                    # give up this Newton iteration
                    # take the best we had (none accepted), so abort
                    return None, {
                        'nit': it,
                        'converged': False,
                        'res_norm': float(res_norm),
                        'simplified_used': simplified_used
                    }

        # continue Newton with updated k

    # Step update
    if not converged:
        # one last residual calc for reporting
        R, phi, *_ = residual_and_phi(k)
        res_norm = np.sqrt(2.0 * phi)

    y_next = y + h * (b[0]*k[0] + b[1]*k[1])
    return y_next, {
        'nit': it,
        'converged': converged,
        'res_norm': float(res_norm),
        'simplified_used': simplified_used
    }


def integrate_gl2(
    f, J, y0, t0, t1,
    h,
    newton_opts: NewtonOptions = NewtonOptions(),
    return_trajectory=True,
    max_step_halves: int = 6,
    h_min: float = None,
    show_progress=True
):
    """
    Fixed-target integration over [t0, t1] with robust IRK solves:
    - damped/backtracking Newton per step
    - automatic fallback to simplified Newton inside the step
    - halve-and-retry step on nonlinear failure, down to h_min

    Parameters
    ----------
    f, J : callables
    y0 : array-like
    t0, t1 : floats
    h : float  (initial step; may be reduced on failures)
    newton_opts : NewtonOptions
    max_step_halves : int
        Max number of times we may halve h when a solve fails at a given time.
    h_min : float or None
        Absolute lower bound on h. Defaults to |t1-t0| / (2**20).

    Returns
    -------
    (T, Y) if return_trajectory else y_final
    """
    assert h > 0.0
    total_T = abs(t1 - t0)
    if h_min is None:
        h_min = total_T / (2**20)

    y = np.asarray(y0, dtype=float).copy()

    if return_trajectory:
        # unknown number of accepted steps if we do retries; store dynamically
        T_list = [float(t0)]
        Y_list = [y.copy()]

    t = float(t0)
    forward = np.sign(t1 - t0) if t1 != t0 else 1.0

    pbar = None
    if show_progress:
        pbar = tqdm(total=abs(t1 - t0), desc="Integrating", unit="t-units")
        last_t = t

    while (t - t1) * forward < 0.0:
        # choose h_eff so we land exactly on t1
        h_eff = min(h, abs(t1 - t)) * forward

        # attempt the step with at most 'max_step_halves' reductions on failure
        attempt_ok = False
        h_try = h_eff
        for _ in range(max_step_halves + 1):
            y_next, info = gauss_legendre_s2_step(y, h_try, f, J, newton_opts)
            if (y_next is not None) and info['converged']:
                attempt_ok = True
                break
            # halve and retry, unless too small
            if abs(h_try) <= h_min:
                break
            h_try *= 0.5

        if not attempt_ok:
            raise RuntimeError(
                f"IRK solve failed at t≈{t:.6g}; residual={info['res_norm']:.3e}; "
                f"last h_try={h_try:.3e}"
            )

        # accept step
        y = y_next
        new_t = t + h_try

        if pbar is not None:
            pbar.update(abs(new_t - t))

        t = new_t
        h = abs(h_try)  # keep the current (possibly reduced) step for the next iteration
        # update tqdm
        

        if return_trajectory:
            T_list.append(t)
            Y_list.append(y.copy())
            
    if pbar is not None:
        pbar.close()

    if return_trajectory:
        T = np.array(T_list, dtype=float)
        Y = np.vstack(Y_list)
        return T, Y
    else:
        return y

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
    T, Y = integrate_gl2(f, J, y0, t0, t1, h)
    y_fwd = Y[-1]
    Tback, Yback = integrate_gl2(f, J, y_fwd, t1, t0, h, newton_maxit = 20)  # integrate backward
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
