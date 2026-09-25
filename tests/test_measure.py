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
