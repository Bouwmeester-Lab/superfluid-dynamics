"""Host-side finite-difference Jacobians, used to check the DLL's Jacobian and the integrators' stability."""
import numpy as np
import tqdm


def normalize(v):
    return v / np.max(np.abs(v))


def jacobian_fd(f, y0, eps=None, scales=None):
    """
    Dense Jacobian via central differences using scalar f: (n,) -> (n,)
    """
    f0 = f(y0)
    n = f0.shape[0]  # output dimension
    m = y0.shape[0]  # input dimension

    if eps is None:
        h0 = np.sqrt(np.finfo(float).eps) * (1.0 + np.abs(y0))
    else:
        h0 = np.full(m, float(eps))
    if scales is not None:
        s = np.asarray(scales, dtype=float)
        h0 = h0 * np.maximum(1.0, np.abs(s))

    J = np.empty((n, m), dtype=float)  # matrix should be output dimension x input dimension

    for j in tqdm.tqdm(range(m), desc="Calculating Jacobian"):
        ej = np.zeros(m); ej[j] = 1.0
        hj = h0[j]
        Fp = f(y0 + hj * ej)
        Fm = f(y0 - hj * ej)
        J[:, j] = (Fp - Fm) / (2.0 * hj)
    return J


def richardson_extrapolation(J1, J2, factor=4):
    return (factor * J2 - J1) / (factor - 1)


def richardson_jacobian(f, y0, h):
    J1 = jacobian_fd(f, y0, h)
    J2 = jacobian_fd(f, y0, h * 0.5)
    return richardson_extrapolation(J1, J2)
