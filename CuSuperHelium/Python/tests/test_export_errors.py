"""Exports must return -1 (not an undefined value) for sizes they do not support."""
import numpy as np

L_SI, RHO, KAPPA, DEPTH_SI = 1.0e-6, 150.0, 0.0, 15.0e-9


def state(n: int):
    x = 2.0 * np.pi * np.arange(n) / n
    return x, 0.01 * np.cos(x), 0.01 * np.sin(x)


def test_rhs_from_vectors_rejects_unsupported_n(sim):
    res, *_ = sim.calculate_rhs_from_vectors(*state(3), L_SI, RHO, KAPPA, DEPTH_SI)
    assert res == -1


def test_rhs256_batched_rejects_unsupported_batch_size(sim):
    batch = 3
    x, y, phi = (np.tile(a, batch) for a in state(256))
    res, *_ = sim.calculate_rhs256_from_vectors_batched(x, y, phi, L_SI, RHO, KAPPA, DEPTH_SI, batch)
    assert res == -1


def test_jacobian_rejects_unsupported_n(sim):
    res, _ = sim.calculate_jacobian(*state(16), L_SI, RHO, KAPPA, DEPTH_SI)
    assert res == -1


def test_rhs_rejects_missing_material(sim):
    """The library has no default material: a zero Hamaker constant or film density is an error, not helium."""
    x, y, phi = state(64)
    res, *_ = sim.calculate_rhs_from_vectors(x, y, phi, L_SI, RHO, KAPPA, DEPTH_SI, hamaker_constant=0.0)
    assert res == -1
    res, *_ = sim.calculate_rhs_from_vectors(x, y, phi, L_SI, RHO, KAPPA, DEPTH_SI, film_density=0.0)
    assert res == -1


def test_sim_properties_fill_default_material():
    import integration.rhs as rhs

    props = rhs.CSimulationProperties(L=L_SI, rho=RHO, kappa=KAPPA, depth=DEPTH_SI)
    assert props.hamaker_constant == rhs.HELIUM_FILM.hamaker_constant
    assert props.film_density == rhs.HELIUM_FILM.density

    other = rhs.FilmMaterial(hamaker_constant=1.0e-24, density=100.0)
    props = rhs.CSimulationProperties(L=L_SI, rho=RHO, kappa=KAPPA, depth=DEPTH_SI, material=other)
    assert (props.hamaker_constant, props.film_density) == (1.0e-24, 100.0)

    props = rhs.CSimulationProperties(L=L_SI, rho=RHO, kappa=KAPPA, depth=DEPTH_SI, hamaker_constant=2.0e-24)
    assert props.hamaker_constant == 2.0e-24
