"""The ruler and the orbit timer: numbers a child can check against a book."""

import math

import numpy as np
import pytest

from src import materials, measure, scenarios, units
from src.sim import Simulation
from src.world import World


def _pair(separation_au, vel_b=(0.0, 0.0), radius=0.0):
    world = World()
    rock = materials.get("rock")
    world.add("A", rock, units.EARTH_MASS, radius, np.zeros(2), np.zeros(2))
    world.add("B", rock, units.EARTH_MASS, radius,
              np.array([separation_au, 0.0]), np.array(vel_b))
    return world


# --- the ruler ---------------------------------------------------------------------

def test_light_crosses_one_au_in_499_seconds():
    # Deleting the division by units.C makes this fail.
    s = measure.separation(_pair(1.0), 0, 1)
    assert s["centres"] == pytest.approx(1.0)
    assert s["light_days"] * units.SEC_PER_DAY == pytest.approx(499.005, abs=0.01)


def test_the_gap_is_surface_to_surface_and_never_negative():
    # Leaving out either radius, or the max(0, ...), makes this fail.
    assert measure.separation(_pair(1.0, radius=0.1), 0, 1)["gap"] == pytest.approx(0.8)
    assert measure.separation(_pair(0.1, radius=0.1), 0, 1)["gap"] == 0.0


def test_closing_is_positive_while_they_approach():
    # B sits on +x and moves towards A.  Flipping the sign makes this fail.
    approaching = measure.separation(_pair(1.0, vel_b=(-0.01, 0.0)), 0, 1)
    receding = measure.separation(_pair(1.0, vel_b=(0.01, 0.0)), 0, 1)
    assert approaching["closing"] == pytest.approx(0.01)
    assert receding["closing"] == pytest.approx(-0.01)


def test_sideways_motion_is_relative_speed_but_not_closing():
    s = measure.separation(_pair(1.0, vel_b=(0.0, 0.02)), 0, 1)
    assert s["closing"] == pytest.approx(0.0, abs=1e-15)
    assert s["relative"] == pytest.approx(0.02)


# --- Kepler ------------------------------------------------------------------------

def test_kepler_gives_a_year_at_one_au():
    world = scenarios.two_body()
    period, a = measure.kepler_period(world, 1, 0)
    assert a == pytest.approx(1.0, rel=1e-6)
    assert period == pytest.approx(365.256, rel=1e-4)


# --- the timer ---------------------------------------------------------------------

def _time(world, index, frame_days, laps=1, limit_days=3000.0):
    """Advance in frame-sized chunks, sampling once a chunk, like the app."""
    sim = Simulation(world=world)
    timer = measure.PeriodTimer(sim.world, index)
    while len(timer.laps) < laps + 1 and sim.world.time < limit_days:
        sim.advance(frame_days, budget_s=1e9)
        timer.sample(sim.world)
        if timer.status != "timing":
            break
    return sim, timer


def test_a_lap_matches_kepler_to_a_twentieth_of_a_percent():
    """7-day frames: without interpolating the crossing, the lap lands on a
    frame boundary and the error is ~1.6%.  With it, measured 0.035%.
    Replacing the interpolated crossing with world.time makes this fail."""
    sim, timer = _time(scenarios.two_body(), 1, 7.0)
    kepler, _ = measure.kepler_period(sim.world, 1, 0)
    assert timer.status == "timing"
    assert timer.measured == pytest.approx(kepler, rel=5e-4)


def test_a_retrograde_orbit_is_timed_too():
    # Counting laps on the signed angle instead of its magnitude makes this fail.
    world = scenarios.two_body()
    world.vel[:world.n] *= -1.0
    sim, timer = _time(world, 1, 7.0)
    assert timer.swept < 0.0
    assert timer.measured == pytest.approx(365.256, rel=1e-3)


def test_a_frame_longer_than_a_quarter_orbit_is_refused():
    """200-day frames on a 365-day orbit: the angle between samples cannot be
    unwrapped.  Without the guard the timer reported 600 days as a lap."""
    sim, timer = _time(scenarios.two_body(), 1, 200.0)
    assert timer.status == "too fast"
    assert timer.measured is None


def test_an_eccentric_orbit_is_refused_past_the_periapsis_scaled_limit():
    """e=0.5: the flat quarter-period limit (~91 days) would let an 80-day
    frame straddling periapsis sweep more than half a turn and count
    backwards, reporting a wildly wrong lap.  The limit scaled by periapsis
    rate (~26 days here) refuses it instead."""
    sim, timer = _time(scenarios.two_body(eccentricity=0.5), 1, 80.0)
    assert timer.status == "too fast"
    assert timer.measured is None


def test_an_eccentric_orbit_is_timed_within_the_scaled_limit():
    """5-day frames sit comfortably inside the e=0.5 limit (~26 days).
    Measured error is ~0.12%: linear interpolation between samples is a
    coarser fit near a fast periapsis passage than it is on a circle, where
    the same interpolation measures to ~0.035% (see the 7-day-frame test
    above)."""
    sim, timer = _time(scenarios.two_body(eccentricity=0.5), 1, 5.0)
    kepler, _ = measure.kepler_period(sim.world, 1, 0)
    assert timer.status == "timing"
    assert timer.measured == pytest.approx(kepler, rel=2e-3)


def test_the_timer_recovers_once_time_slows_down():
    # Leaving out the status reset to "timing" in sample() makes this fail.
    sim = Simulation(world=scenarios.two_body())
    timer = measure.PeriodTimer(sim.world, 1)
    sim.advance(200.0, budget_s=1e9)
    timer.sample(sim.world)
    assert timer.status == "too fast"
    while len(timer.laps) < 2 and sim.world.time < 2000.0:
        sim.advance(5.0, budget_s=1e9)
        timer.sample(sim.world)
    assert timer.status == "timing"
    assert timer.measured == pytest.approx(365.256, rel=1e-3)


def test_earth_takes_a_sidereal_year():
    """The number in the book: 365.256 days.  Planets only, 6-day frames,
    which is what the one-year-a-second setting gives."""
    world = scenarios.solar_system(moons=False)
    earth = [b.name for b in world.bodies].index("Earth")
    sim, timer = _time(world, earth, 6.0)
    assert timer.measured == pytest.approx(365.256, rel=5e-4)


def test_timing_the_earth_weighs_the_sun():
    """a^3 / P^2 in AU and years is the pair's mass in suns.  Using the
    period in days instead of years makes this fail."""
    sim, timer = _time(scenarios.two_body(), 1, 7.0)
    assert timer.mass_from_orbit(sim.world) == pytest.approx(
        1.0 + units.EARTH_MASS, rel=2e-3)


def test_no_mass_before_the_first_lap():
    sim = Simulation(world=scenarios.two_body())
    timer = measure.PeriodTimer(sim.world, 1)
    sim.advance(30.0, budget_s=1e9)
    timer.sample(sim.world)
    assert timer.measured is None
    assert timer.mass_from_orbit(sim.world) is None
    assert 0.05 < timer.fraction_round < 0.12      # 30 of 365 days


def test_the_sun_is_not_in_orbit():
    world = scenarios.two_body()
    timer = measure.PeriodTimer(world, 0)
    assert timer.attractor_id is None
    assert timer.status == "not in orbit"
    timer.sample(world)                           # must not raise
    assert timer.status == "not in orbit"


def test_a_removed_body_is_lost_not_a_crash():
    # Deleting the `index is None` check makes this raise.
    sim = Simulation(world=scenarios.two_body())
    timer = measure.PeriodTimer(sim.world, 1)
    sim.world.remove(1)
    sim.refresh()
    sim.advance(1.0, budget_s=1e9)
    timer.sample(sim.world)
    assert timer.status == "lost"


def test_a_body_captured_by_another_star_has_moved_on():
    """Carry the secondary to a second, far heavier star and set it orbiting
    there.  Deleting the dominant_attractor comparison makes this fail."""
    sim = Simulation(world=scenarios.two_body())
    world = sim.world
    timer = measure.PeriodTimer(world, 1)
    far = np.array([50.0, 0.0])
    world.add("Other", materials.get("hydrogen"), 5.0, units.SUN_RADIUS,
              far, np.zeros(2), kind=world.bodies[0].kind)
    world.pos[1] = far + np.array([0.5, 0.0])
    world.vel[1] = np.array([0.0, math.sqrt(units.G * 5.0 / 0.5)])
    sim.refresh()
    sim.advance(1.0, budget_s=1e9)
    timer.sample(world)
    assert timer.status == "moved on"


# --- the lap-mean a (F1 + F2) ------------------------------------------------------

def test_lap_a_and_lap_kepler_are_none_before_the_first_lap():
    # Deleting the __init__/_restart initialisation of lap_a makes this raise.
    sim = Simulation(world=scenarios.two_body())
    timer = measure.PeriodTimer(sim.world, 1)
    assert timer.lap_a is None
    assert timer.lap_kepler is None


def test_lap_a_and_lap_kepler_freeze_at_the_crossing():
    """A clean two-body orbit's instantaneous a is constant, so the frozen lap
    value should read the same as Kepler's own number.  Not freezing (leaving
    lap_a/lap_kepler at None) makes this fail."""
    sim, timer = _time(scenarios.two_body(), 1, 7.0)
    assert timer.lap_a == pytest.approx(1.0, rel=5e-3)
    assert timer.lap_kepler == pytest.approx(365.256, rel=5e-4)


def _sun_earth_moon():
    """Sun, Earth and Moon only, trimmed from the real scenario so the Moon is
    perturbed by the Sun exactly as it is in the game, without Io's tiny
    dynamical time forcing the timestep down."""
    world = scenarios.solar_system(moons=True)
    keep = {"Sun", "Earth", "Moon"}
    for body in list(world.bodies):
        if body.name not in keep:
            world.remove(world.index_of(body.id))
    return world


def test_the_moons_mass_no_longer_wanders_through_a_lap():
    """Before this fix, the *instantaneous* a wanders across a lap (measured
    directly here: recomputing a from the live state over the ten days
    sampled below swings a^3/measured^2 by ~5.5%, matching the reviewer's
    27.22->28.09 d Kepler swing and 0.971->1.03 Earth+Moon-mass swing) even
    though the measured lap length itself is steady.  With the lap's mean a
    frozen at the crossing there is nothing left inside a lap to wander:
    a^3/measured^2 must read bit-identically at every frame sampled through
    the following lap.  Using the live osculating a instead of the frozen
    lap value in mass_from_orbit makes this fail.

    It settles about 0.8-1.2% below the true Earth+Moon mass here, not the
    finding's proposed +-0.5% -- real solar perturbation on the Moon (the
    spec's own "comes out about 1% slow against its own two-body prediction"),
    which mass_from_orbit's a^3/P^2 roughly doubles from the period error.
    Flagged for the controller in final-fix-report.md rather than loosened."""
    world = _sun_earth_moon()
    sim = Simulation(world=world)
    earth = world.index_of(next(b for b in world.bodies if b.name == "Earth").id)
    moon = world.index_of(next(b for b in world.bodies if b.name == "Moon").id)
    target = float(world.mass[earth] + world.mass[moon])
    timer = measure.PeriodTimer(sim.world, moon)
    # Two laps at 0.5-day frames, as the reviewer's experiment used.
    while len(timer.laps) < 3 and sim.world.time < 100.0:
        sim.advance(0.5, budget_s=1e9)
        timer.sample(sim.world)
    assert timer.status == "timing"
    samples = []
    for _ in range(20):
        sim.advance(0.5, budget_s=1e9)
        timer.sample(sim.world)
        mass = timer.mass_from_orbit(sim.world)
        assert mass is not None
        samples.append(mass)
    assert samples == [samples[0]] * len(samples)          # no wander at all
    assert samples[0] == pytest.approx(target, rel=0.02)   # right ballpark


def test_mass_from_orbit_is_frozen_and_positive_once_unbound():
    """After a lap, tripling the secondary's heliocentric speed makes the
    orbit unbound.  mass_from_orbit must keep reporting the frozen, positive
    lap value rather than a fresh (and here negative) a^3/P^2.  Recomputing
    from the live osculating a instead of the frozen lap_a makes this fail."""
    sim, timer = _time(scenarios.two_body(), 1, 7.0)
    assert timer.status == "timing"
    frozen = timer.mass_from_orbit(sim.world)
    assert frozen is not None and frozen > 0.0

    sim.world.vel[1] *= 3.0
    sim.refresh()
    sim.advance(1.0, budget_s=1e9)
    timer.sample(sim.world)
    assert timer.status == "not in orbit"

    mass = timer.mass_from_orbit(sim.world)
    assert mass == pytest.approx(frozen)
    assert mass > 0.0 and math.isfinite(mass)
