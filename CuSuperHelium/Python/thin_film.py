import numpy as np


def thin_film_velocities_axisymmetric(
    r: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    h0: float,
    *,
    R: float | None = None,
    side_bc: str = "dirichlet",
    phi_wall: float = 0.0,
    eta_prime: np.ndarray | None = None,
):
    """
    Thin-film Dirichlet-to-Neumann map for an axisymmetric film.

    Parameters
    ----------
    r:
        Cell-center radial coordinates, usually r_i = (i + 1/2) * dr.
    eta:
        Free-surface displacement eta(r).
    phi:
        Surface potential phi(r).
    h0:
        Equilibrium film thickness. Same nondimensional units as r and eta.
    R:
        Outer radius. If None, assumes centered grid and sets R = r[-1] + dr/2.
    side_bc:
        "dirichlet" -> impose phi(R) = phi_wall.
        "neumann"   -> impose phi_r(R) = 0, no radial flux through wall.
    phi_wall:
        Boundary value for Dirichlet side condition.
    eta_prime:
        Optional eta_r. If None, computed with np.gradient.

    Returns
    -------
    q:
        Unit-normal velocity u · n_hat.
    vr:
        Leading-order radial velocity, approximately phi_r.
    vz:
        Vertical velocity at the surface, reconstructed from q.
    div_flux:
        The quantity (1/r) d/dr (r H phi_r).
    F:
        Radial flux at cell faces, F = r H phi_r.
    """

    r = np.asarray(r, dtype=float)
    eta = np.asarray(eta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    if r.ndim != 1:
        raise ValueError("r must be 1D.")
    if eta.shape != r.shape or phi.shape != r.shape:
        raise ValueError("eta and phi must have the same shape as r.")

    N = r.size
    if N < 2:
        raise ValueError("Need at least two radial points.")

    # Assume approximately uniform centered grid.
    dr_values = np.diff(r)
    dr = float(np.mean(dr_values))

    if not np.allclose(dr_values, dr, rtol=1e-10, atol=1e-14):
        raise ValueError("This simple implementation assumes a uniform centered grid.")

    if R is None:
        R = r[-1] + 0.5 * dr

    # Face coordinates.
    # For centered grid r_i = (i + 1/2) dr, faces are 0, dr, ..., R.
    r_face = np.empty(N + 1, dtype=float)
    r_face[0] = 0.0
    r_face[1:-1] = 0.5 * (r[:-1] + r[1:])
    r_face[-1] = R

    # Local film thickness.
    H = h0 + eta

    if np.any(H <= 0):
        raise ValueError("Film thickness H = h0 + eta must remain positive.")

    # eta_r for normal factor.
    if eta_prime is None:
        edge_order = 2 if N >= 3 else 1
        eta_prime = np.gradient(eta, r, edge_order=edge_order)
    else:
        eta_prime = np.asarray(eta_prime, dtype=float)
        if eta_prime.shape != r.shape:
            raise ValueError("eta_prime must have same shape as r.")

    # Face gradient of phi.
    dphi_dr_face = np.zeros(N + 1, dtype=float)

    # Axis regularity:
    # phi_r(0) = 0, so face flux at the axis is zero.
    dphi_dr_face[0] = 0.0

    # Interior faces.
    dphi_dr_face[1:N] = (phi[1:] - phi[:-1]) / (r[1:] - r[:-1])

    # Outer wall.
    side_bc = side_bc.lower()
    if side_bc == "dirichlet":
        # Last center is half a cell away from R for a centered grid.
        dphi_dr_face[N] = (phi_wall - phi[-1]) / (R - r[-1])
    elif side_bc == "neumann":
        dphi_dr_face[N] = 0.0
    else:
        raise ValueError("side_bc must be 'dirichlet' or 'neumann'.")

    # Thickness at faces.
    H_face = np.empty(N + 1, dtype=float)
    H_face[0] = H[0]
    H_face[1:N] = 0.5 * (H[:-1] + H[1:])
    H_face[N] = H[-1]

    # Radial flux F = r H phi_r.
    F = r_face * H_face * dphi_dr_face

    # Finite-volume divergence:
    # average of (1/r) dF/dr over radial cell [r_L, r_R]
    # denominator = integral r dr = (r_R^2 - r_L^2)/2.
    radial_cell_measure = 0.5 * (r_face[1:] ** 2 - r_face[:-1] ** 2)

    div_flux = (F[1:] - F[:-1]) / radial_cell_measure

    # Unit-normal velocity.
    norm = np.sqrt(1.0 + eta_prime * eta_prime)

    q = -div_flux / norm

    # Leading-order radial velocity at centers.
    # Average neighboring face gradients.
    vr = 0.5 * (dphi_dr_face[:-1] + dphi_dr_face[1:])

    # q = (vz - eta_r vr) / sqrt(1 + eta_r^2)
    # therefore vz = q sqrt(...) + eta_r vr.
    vz = q * norm + eta_prime * vr

    return q, vr, vz, div_flux, F