import time

import numpy as np
import pytest

from src import units
from src.sim import Simulation


def test_advance_returns_the_days_it_managed():
    sim = Simulation()
    done = sim.advance(1.0, budget_s=1.0)
    assert done == pytest.approx(1.0, rel=1e-9)
    assert sim.world.time == pytest.approx(1.0, rel=1e-9)


def test_paused_simulation_does_not_move():
    sim = Simulation()
    sim.paused = True
    before = sim.world.pos.copy()
    assert sim.advance(10.0) == 0.0
    assert np.array_equal(sim.world.pos, before)


def test_wall_clock_budget_is_respected():
    """Without this the window freezes at high time scales and reads as a crash."""
    sim = Simulation()
    start = time.perf_counter()
    done = sim.advance(50_000.0, budget_s=0.02)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, "budget ignored, took %.3fs" % elapsed
    assert done < 50_000.0
    assert sim.last_ratio < 1.0


def test_ratio_is_one_when_the_budget_is_ample():
    sim = Simulation()
    sim.advance(0.5, budget_s=5.0)
    assert sim.last_ratio == pytest.approx(1.0, rel=1e-9)


def test_trails_record_points_as_bodies_move():
    sim = Simulation()
    earth = next(b for b in sim.world.bodies if b.name == "Earth")
    assert earth.trail.count == 0
    for _ in range(40):
        sim.advance(3.0, budget_s=5.0)
        sim.update_trails()
    assert earth.trail.count > 3
    assert earth.trail.ordered().shape[1] == 2


def test_reset_restores_the_solar_system():
    sim = Simulation()
    sim.advance(100.0, budget_s=5.0)
    sim.world.remove(sim.world.n - 1)
    sim.reset()
    assert sim.world.time == 0.0
    assert any(b.name == "Neptune" for b in sim.world.bodies)
    assert np.allclose(sim.world.total_momentum(), 0.0, atol=1e-18)


def test_saturn_has_rings():
    sim = Simulation()
    saturn = next(b for b in sim.world.bodies if b.name == "Saturn")
    assert saturn.rings is True


def test_energy_error_does_not_run_away():
    """The long-horizon drift check, run without moons.

    Io's dynamical time is a quarter of a day against Mercury's fourteen, so
    including the moons costs two orders of magnitude in substeps and buys
    nothing here: this is a property of the integrator, and the planets alone
    exercise it over a far longer horizon for a fraction of the time.

    What is asserted is that the error stays small *and grows sublinearly*.  A
    dissipative integrator -- RK4, say -- loses energy at a steady rate, so its
    error doubles when the run doubles and planets visibly spiral inward.
    Verlet's does not, and that is the whole reason it was chosen.

    Measured on this system: 1.7e-7 at a fixed 0.5-day step, rising to 1.7e-6
    once ``choose_k`` is in play, because Mercury's eccentricity swings the
    required step across a power-of-two boundary twice an orbit and each switch
    costs a little symplecticity.  Both are far below anything observable.
    """
    from src import scenarios
    sim = Simulation(world=scenarios.solar_system(moons=False))
    e0 = sim.world.total_energy()

    def drift():
        return abs((sim.world.total_energy() - e0) / e0)

    for _ in range(100):
        sim.advance(200.0, budget_s=10.0)
    assert sim.world.time == pytest.approx(20_000.0, rel=1e-6)
    first = drift()

    for _ in range(100):
        sim.advance(200.0, budget_s=10.0)
    second = drift()

    assert first < 1e-5, "energy drift %.3g over 55 years" % first
    assert second < 2.0 * first, (
        "error is growing linearly (%.3g -> %.3g): the integrator is dissipating"
        % (first, second))


def test_energy_is_stable_with_moons_present():
    """Moons drive the block timestep down by two orders of magnitude; this
    checks that the finer steps stay just as well behaved."""
    sim = Simulation()
    e0 = sim.world.total_energy()
    for _ in range(20):
        sim.advance(20.0, budget_s=10.0)
    drift = abs((sim.world.total_energy() - e0) / e0)
    assert drift < 1e-5, "energy drift %.3g" % drift


# --- impact events -------------------------------------------------------------------

def _colliding_world():
    from src import materials
    from src.world import World
    rock = materials.get("rock")
    r = units.EARTH_RADIUS
    world = World(8)
    world.add("A", rock, units.EARTH_MASS, r, np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    world.add("B", rock, units.EARTH_MASS, r, np.array([0.5 * r, 0.0]), np.array([0.0, 0.0]))
    return world


def test_a_collision_records_an_impact_event():
    from src.collisions import ACCRETE, DISRUPT, GRAZE
    sim = Simulation(world=_colliding_world())
    sim.advance(1.0, budget_s=1.0)
    assert len(sim.impacts) >= 1
    event = sim.impacts[0]
    assert event.outcome in (ACCRETE, GRAZE, DISRUPT)
    assert event.target_radius > 0.0
    assert len(event.tones) == 3


def test_impacts_are_cleared_at_the_start_of_each_advance():
    sim = Simulation(world=_colliding_world())
    sim.advance(1.0, budget_s=1.0)
    assert sim.impacts, "test setup must actually produce an event"
    sim.advance(0.01, budget_s=1.0)
    assert sim.impacts == []


def test_impact_events_hold_only_plain_data():
    sim = Simulation(world=_colliding_world())
    sim.advance(1.0, budget_s=1.0)
    assert sim.impacts
    for event in sim.impacts:
        assert isinstance(event.pos, tuple) and len(event.pos) == 2
        assert isinstance(event.vel, tuple) and len(event.vel) == 2
        assert isinstance(event.energy, float)
        assert isinstance(event.outcome, str)
        assert isinstance(event.target_radius, float)
        assert isinstance(event.projectile_radius, float)
        assert isinstance(event.tones, tuple)
        assert isinstance(event.on_star, bool)


# --- temperature ------------------------------------------------------------------

def test_a_new_simulation_opens_in_thermal_balance():
    """Not at absolute zero.  A scenario that had to warm up over its first
    simulated months would be showing a transient nobody asked for."""
    from src import thermal
    sim = Simulation()
    world = sim.world
    earth = next(b for b in world.bodies if b.name == "Earth")
    i = world.index_of(earth.id)
    assert float(world.temperature[i]) == pytest.approx(
        thermal.equilibrium_temperature(world, i), rel=1e-6)
    assert float(world.temperature[i]) > units.CMB_TEMPERATURE_K


def test_advancing_the_simulation_relaxes_a_body_that_was_heated():
    sim = Simulation()
    world = sim.world
    earth = next(b for b in world.bodies if b.name == "Earth")
    i = world.index_of(earth.id)
    settled = float(world.temperature[i])
    world.temperature[i] = 2000.0
    sim.advance(40.0, budget_s=5.0)
    i = world.index_of(earth.id)
    cooled = float(world.temperature[i])
    assert cooled < 2000.0
    assert cooled > settled - 30.0


def test_a_paused_simulation_does_not_change_any_temperature():
    sim = Simulation()
    sim.paused = True
    before = sim.world.temperature[:sim.world.n].copy()
    sim.advance(10.0)
    assert np.allclose(sim.world.temperature[:sim.world.n], before)


def test_reset_settles_the_new_world():
    """Every body at its equilibrium -- except the ones whose own physics fixes
    it.  The Sun's equilibrium *as a receiver* is 2.85 K, the temperature the
    background alone would put it at; what it actually sits at is the 5772 K
    fusion gives it.  Comparing a star against the receiver number is the
    mistake this test exists to catch.
    """
    from src import thermal
    sim = Simulation()
    sim.world.temperature[:sim.world.n] = 1.0
    sim.reset("solar_system")
    world = sim.world
    target = thermal.equilibrium_temperatures(world)
    for i in range(world.n):
        pinned = thermal.pinned_temperature(world, i)
        expected = pinned if pinned is not None else float(target[i])
        assert float(world.temperature[i]) == pytest.approx(expected, rel=1e-6), \
            world.bodies[i].name


# --- maneuver nodes -----------------------------------------------------------------


def _craft_in_orbit(sim, radius=1.0, fuel_ms=9300.0):
    """A craft on a circular orbit about the star, with a full tank."""
    from src import materials
    from src.body import KIND_CRAFT
    world = sim.world
    speed = (units.G * float(world.mass[0]) / radius) ** 0.5
    body = world.add("Shuttle", materials.SPACECRAFT, units.EARTH_MASS * 1e-18,
                     units.CRAFT_RADIUS, np.array([radius, 0.0]),
                     np.array([0.0, speed]), kind=KIND_CRAFT)
    body.fuel_dv = units.from_ms(fuel_ms)
    sim.refresh()
    return body


def _solar_sim():
    from src import scenarios
    return Simulation(world=scenarios.solar_system(moons=False))


def test_a_node_fires_within_a_substep_of_its_due_time():
    from src import maneuver
    sim = _solar_sim()
    body = _craft_in_orbit(sim)
    body.plan = (maneuver.Maneuver(due_time=sim.world.time + 3.0,
                                   prograde=units.from_ms(500.0)),)
    sim.refresh()
    for _ in range(200):
        sim.advance(1.0, budget_s=1.0)
        if not body.plan:
            break
    assert body.plan == ()
    assert sim.world.time < 10.0


def test_a_node_does_not_fire_while_paused():
    from src import maneuver
    sim = _solar_sim()
    body = _craft_in_orbit(sim)
    body.plan = (maneuver.Maneuver(due_time=0.0, prograde=units.from_ms(500.0)),)
    sim.refresh()
    sim.paused = True
    sim.advance(10.0, budget_s=1.0)
    assert len(body.plan) == 1


def test_a_node_fires_at_the_fastest_time_scale_too():
    """At ten years a second a frame is sixty days.  A node checked only at
    frame boundaries would be missed by weeks, so this compares two runs
    whose nodes fall at opposite ends of the SAME frame: if burns were applied
    at frame granularity both would land on the identical final state.
    """
    from src import maneuver, orbits

    def fly(lead_days):
        sim = _solar_sim()
        body = _craft_in_orbit(sim)
        body.plan = (maneuver.Maneuver(due_time=sim.world.time + lead_days,
                                       prograde=units.from_ms(500.0)),)
        sim.refresh()
        sim.advance(3652.5 / 60.0, budget_s=5.0)      # one 60.875-day frame
        assert body.plan == (), "the node never fired at all"
        index = sim.world.index_of(body.id)
        return np.array(sim.world.pos[index]), np.array(sim.world.vel[index])

    early_pos, early_vel = fly(2.0)
    late_pos, late_vel = fly(58.0)

    # Burning 56 days apart inside one frame must leave the craft measurably
    # elsewhere.  Frame-granularity execution would make these identical.
    assert float(np.hypot(*(early_pos - late_pos))) > 1e-4
    assert float(np.hypot(*(early_vel - late_vel))) > 0.0


def test_firing_a_burn_actually_changes_the_orbit():
    from src import maneuver, orbits
    sim = _solar_sim()
    body = _craft_in_orbit(sim)
    index = sim.world.index_of(body.id)
    mu = units.G * float(sim.world.mass[0])
    before = orbits.elements_from_state(
        mu, sim.world.pos[index] - sim.world.pos[0],
        sim.world.vel[index] - sim.world.vel[0])["a"]
    body.plan = (maneuver.Maneuver(due_time=sim.world.time + 1.0,
                                   prograde=units.from_ms(2000.0)),)
    sim.refresh()
    for _ in range(50):
        sim.advance(1.0, budget_s=1.0)
        if not body.plan:
            break
    index = sim.world.index_of(body.id)
    after = orbits.elements_from_state(
        mu, sim.world.pos[index] - sim.world.pos[0],
        sim.world.vel[index] - sim.world.vel[0])["a"]
    assert after > before * 1.05


def test_fuel_is_deducted_by_the_impulse_actually_applied():
    from src import maneuver
    sim = _solar_sim()
    body = _craft_in_orbit(sim, fuel_ms=9300.0)
    body.plan = (maneuver.Maneuver(due_time=sim.world.time + 1.0,
                                   prograde=units.from_ms(1500.0)),)
    sim.refresh()
    for _ in range(50):
        sim.advance(1.0, budget_s=1.0)
        if not body.plan:
            break
    assert abs(units.to_ms(body.fuel_dv) - 7800.0) < 1.0


def test_removing_a_craft_leaves_the_cached_burn_time_consistent():
    from src import maneuver
    sim = _solar_sim()
    body = _craft_in_orbit(sim)
    body.plan = (maneuver.Maneuver(due_time=sim.world.time + 5.0),)
    sim.refresh()
    sim.world.remove(sim.world.index_of(body.id))
    sim.refresh()
    assert sim._next_burn_time == float("inf")
    sim.advance(20.0, budget_s=1.0)          # must not raise
