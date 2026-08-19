import numpy as np
import pytest

from src import scenarios, units
from src.integrator import DT_BASE, Integrator


def _run(world, days, integrator=None):
    integrator = integrator or Integrator(world.capacity)
    integrator.refresh(world)
    done = 0.0
    while done < days:
        integrator.choose_k(world)
        dt = min(integrator.dt, days - done)
        integrator.step(world, dt)
        done += dt
    return integrator


def test_two_body_conserves_energy_over_100_orbits():
    """Velocity Verlet exists to stop planets spiralling in.  This is that test."""
    world = scenarios.two_body(separation=1.0, secondary_mass=units.EARTH_MASS)
    integrator = Integrator(world.capacity)
    integrator.refresh(world)
    e0 = world.total_energy()
    _run(world, 100 * units.DAYS_PER_YEAR, integrator)
    drift = abs((world.total_energy() - e0) / e0)
    assert drift < 1e-6, "energy drift %.3g" % drift


def test_earth_returns_to_its_start_after_one_year():
    world = scenarios.solar_system(moons=False)
    names = [b.name for b in world.bodies]
    earth = names.index("Earth")
    start = world.pos[earth].copy()
    _run(world, units.DAYS_PER_YEAR)
    offset = np.linalg.norm(world.pos[earth] - start)
    # 0.5% of the orbit's circumference.
    assert offset < 0.005 * 2 * np.pi, "Earth missed its start by %.4f AU" % offset


def test_neptune_year_is_165_years():
    from src import orbits
    world = scenarios.solar_system(moons=False)
    names = [b.name for b in world.bodies]
    nep = names.index("Neptune")
    sun = names.index("Sun")
    mu = units.G * (world.mass[sun] + world.mass[nep])
    el = orbits.elements_from_state(
        mu, world.pos[nep] - world.pos[sun], world.vel[nep] - world.vel[sun])
    years = el["period"] / units.DAYS_PER_YEAR
    assert years == pytest.approx(164.8, rel=0.005)


def test_choose_k_moves_at_most_one_notch():
    world = scenarios.solar_system(moons=True)
    integrator = Integrator(world.capacity)
    integrator.refresh(world)
    previous = integrator.k
    for _ in range(20):
        k = integrator.choose_k(world)
        assert abs(k - previous) <= 1
        previous = k


def test_timestep_is_always_a_power_of_two_fraction():
    world = scenarios.solar_system()
    integrator = Integrator(world.capacity)
    integrator.refresh(world)
    integrator.choose_k(world)
    ratio = DT_BASE / integrator.dt
    assert ratio == pytest.approx(round(ratio))
    assert (round(ratio) & (round(ratio) - 1)) == 0


def test_momentum_is_conserved_by_stepping():
    world = scenarios.solar_system(moons=False)
    p0 = world.total_momentum().copy()
    _run(world, 500.0)
    assert np.allclose(world.total_momentum(), p0, atol=1e-14)
