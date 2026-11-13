"""Exports return -1 (not an undefined value) for sizes they cannot handle, and accept every other size."""
import numpy as np
import pytest

L_SI, RHO, KAPPA, DEPTH_SI = 1.0e-6, 150.0, 0.0, 15.0e-9


def state(n: int, amplitude: float = 0.01):
    x = 2.0 * np.pi * np.arange(n) / n
    return x, amplitude * np.cos(x), amplitude * np.sin(x)


def test_rhs_from_vectors_rejects_too_few_points(sim):
    res, *_ = sim.calculate_rhs_from_vectors(*state(1), L_SI, RHO, KAPPA, DEPTH_SI)
    assert res == -1


@pytest.mark.parametrize("n", [3, 100, 300])
def test_rhs_from_vectors_accepts_any_n(sim, n):
    """Sizes are runtime values: non-powers of two that the old size dispatch rejected now work."""
    res, vx, vy, dphi = sim.calculate_rhs_from_vectors(*state(n), L_SI, RHO, KAPPA, DEPTH_SI)
    assert res == 0
    assert all(np.all(np.isfinite(v)) for v in (vx, vy, dphi))


def test_rhs_batched_matches_single_interfaces(sim):
    """A batch of 3 interfaces of 100 points gives the same RHS as three separate calls."""
    n, batch = 100, 3
    states = [state(n, amplitude=0.005 * (b + 1)) for b in range(batch)]
    x, y, phi = (np.concatenate([s[k] for s in states]) for k in range(3))
    res, vx, vy, dphi = sim.calculate_rhs_from_vectors_batched(x, y, phi, L_SI, RHO, KAPPA, DEPTH_SI, batch)
    assert res == 0
    for b, s in enumerate(states):
        res_b, vx_b, vy_b, dphi_b = sim.calculate_rhs_from_vectors(*s, L_SI, RHO, KAPPA, DEPTH_SI)
        assert res_b == 0
        part = slice(b * n, (b + 1) * n)
        # the batched LU (cuBLAS) and the single LU (cuSOLVER) round differently, so not bit for bit
        for batched, single in ((vx, vx_b), (vy, vy_b), (dphi, dphi_b)):
            np.testing.assert_allclose(batched[part], single, rtol=1e-9, atol=1e-12 * np.max(np.abs(single)))


def test_jacobian_rejects_too_few_points(sim):
    res, _ = sim.calculate_jacobian(*state(1), L_SI, RHO, KAPPA, DEPTH_SI)
    assert res == -1


def test_rhs_rejects_missing_material(sim):
    """The library has no default material: a zero Hamaker constant or film density is an error, not helium."""
    x, y, phi = state(64)
    res, *_ = sim.calculate_rhs_from_vectors(x, y, phi, L_SI, RHO, KAPPA, DEPTH_SI, hamaker_constant=0.0)
    assert res == -1
    res, *_ = sim.calculate_rhs_from_vectors(x, y, phi, L_SI, RHO, KAPPA, DEPTH_SI, film_density=0.0)
    assert res == -1


def test_sim_properties_fill_default_material():
    import cusuperhelium as rhs

    props = rhs.CSimulationProperties(L=L_SI, rho=RHO, kappa=KAPPA, depth=DEPTH_SI)
    assert props.hamaker_constant == rhs.HELIUM_FILM.hamaker_constant
    assert props.film_density == rhs.HELIUM_FILM.density

    other = rhs.FilmMaterial(hamaker_constant=1.0e-24, density=100.0)
    props = rhs.CSimulationProperties(L=L_SI, rho=RHO, kappa=KAPPA, depth=DEPTH_SI, material=other)
    assert (props.hamaker_constant, props.film_density) == (1.0e-24, 100.0)

    props = rhs.CSimulationProperties(L=L_SI, rho=RHO, kappa=KAPPA, depth=DEPTH_SI, hamaker_constant=2.0e-24)
    assert props.hamaker_constant == 2.0e-24
