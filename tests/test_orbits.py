import math

import numpy as np
import pytest

from src import orbits, units


@pytest.mark.parametrize("a,e,lon_peri,nu", [
    (1.0, 0.0, 0.0, 0.0),
    (1.5, 0.2, 0.7, 1.3),
    (5.2, 0.048, 0.257, -2.1),
    (30.1, 0.0086, 0.785, 3.0),
])
def test_elements_state_roundtrip(a, e, lon_peri, nu):
    mu = units.G * 1.0
    pos, vel = orbits.state_from_elements(mu, a, e, lon_peri, nu)
    el = orbits.elements_from_state(mu, pos, vel)
    assert el["a"] == pytest.approx(a, rel=1e-10)
    assert el["e"] == pytest.approx(e, abs=1e-10)
    assert el["bound"] is True
    # Angles wrap, so compare their difference reduced to (-pi, pi].
    d = (el["true_anomaly"] - nu + math.pi) % (2 * math.pi) - math.pi
    assert d == pytest.approx(0.0, abs=1e-9)


def test_kepler_solver_matches_definition():
    """M = E - e sin E must hold for the E implied by the returned true anomaly."""
    e = 0.35
    for mean in (0.0, 0.5, 1.9, 3.0, 5.5):
        nu = orbits.true_anomaly_from_mean(mean, e)
        big_e = 2.0 * math.atan2(
            math.sqrt(1.0 - e) * math.sin(nu * 0.5),
            math.sqrt(1.0 + e) * math.cos(nu * 0.5),
        )
        recovered = big_e - e * math.sin(big_e)
        d = (recovered - mean + math.pi) % (2 * math.pi) - math.pi
        assert d == pytest.approx(0.0, abs=1e-10)


def test_dominant_attractor_of_moon_is_earth():
    from src import scenarios
    world = scenarios.solar_system()
    names = [b.name for b in world.bodies]
    moon = names.index("Moon")
    assert names[orbits.dominant_attractor(world, moon)] == "Earth"


def test_dominant_attractor_of_earth_is_sun():
    from src import scenarios
    world = scenarios.solar_system()
    names = [b.name for b in world.bodies]
    earth = names.index("Earth")
    assert names[orbits.dominant_attractor(world, earth)] == "Sun"


def test_attractor_is_not_just_the_strongest_pull():
    """The Sun pulls on the Moon about twice as hard as Earth does.  Picking the
    largest force therefore gives the wrong answer, which is the whole reason
    dominant_attractor uses spheres of influence instead."""
    import numpy as np
    from src import scenarios, units
    world = scenarios.solar_system()
    names = [b.name for b in world.bodies]
    moon, earth, sun = names.index("Moon"), names.index("Earth"), names.index("Sun")

    def pull(source):
        d = world.pos[source] - world.pos[moon]
        return units.G * world.mass[source] / float(np.dot(d, d))

    assert pull(sun) > pull(earth), "premise of this test no longer holds"
    assert names[orbits.dominant_attractor(world, moon)] == "Earth"


def test_jovian_moons_orbit_jupiter_not_each_other():
    """Io sits inside Ganymede's Hill radius but is nowhere near bound to it."""
    from src import scenarios
    world = scenarios.solar_system()
    names = [b.name for b in world.bodies]
    for moon in ("Io", "Europa", "Ganymede", "Callisto"):
        assert names[orbits.dominant_attractor(world, names.index(moon))] == "Jupiter"


def test_titan_orbits_saturn():
    from src import scenarios
    world = scenarios.solar_system()
    names = [b.name for b in world.bodies]
    assert names[orbits.dominant_attractor(world, names.index("Titan"))] == "Saturn"


def test_every_planet_orbits_the_sun():
    from src import scenarios
    world = scenarios.solar_system()
    names = [b.name for b in world.bodies]
    for planet in ("Mercury", "Venus", "Earth", "Mars",
                   "Jupiter", "Saturn", "Uranus", "Neptune"):
        assert names[orbits.dominant_attractor(world, names.index(planet))] == "Sun"


def test_the_sun_orbits_nothing():
    from src import scenarios
    world = scenarios.solar_system()
    names = [b.name for b in world.bodies]
    assert orbits.dominant_attractor(world, names.index("Sun")) is None


def test_periapsis_is_positive_for_an_escaping_body():
    """r(1-e)/(1+e) goes negative once e > 1, which made the inspector report a
    negative closest approach and left the aiming preview with nothing to draw."""
    import numpy as np
    mu = units.G
    pos = np.array([1.0, 0.0])
    for factor in (1.05, 1.3, 3.0):
        vel = np.array([0.0, math.sqrt(2.0 * mu) * factor])
        el = orbits.elements_from_state(mu, pos, vel)
        assert el["bound"] is False
        assert el["e"] > 1.0
        assert el["periapsis"] > 0.0, "e=%.3f gave periapsis %.3g" % (
            el["e"], el["periapsis"])
        assert el["periapsis"] <= 1.0 + 1e-12


def test_semi_latus_rectum_is_consistent_for_bound_orbits():
    for a, e in ((1.0, 0.0), (2.5, 0.3), (0.8, 0.7)):
        pos, vel = orbits.state_from_elements(units.G, a, e, 0.3, 1.1)
        el = orbits.elements_from_state(units.G, pos, vel)
        assert el["p"] == pytest.approx(a * (1 - e * e), rel=1e-9)


def test_bound_is_a_real_bool_even_with_numpy_inputs():
    """mu is usually a numpy scalar; a leaking np.bool_ breaks `is True`."""
    mu = np.float64(units.G)
    pos, vel = orbits.state_from_elements(mu, 1.0, 0.1, 0.0, 0.0)
    el = orbits.elements_from_state(mu, np.asarray(pos), np.asarray(vel))
    assert el["bound"] is True
    assert type(el["bound"]) is bool


# --- intercept solving --------------------------------------------------------------

def test_propagate_matches_state_from_elements_after_a_known_fraction_of_a_period():
    mu = units.G * 1.0
    a, e = 1.0, 0.2
    pos0, vel0 = orbits.state_from_elements(mu, a, e, lon_peri=0.4, true_anomaly=0.0)
    period = 2.0 * math.pi * math.sqrt(a ** 3 / mu)
    pos1, vel1 = orbits.propagate(mu, pos0, vel0, period)
    assert pos1 == pytest.approx(pos0, abs=1e-9)
    assert vel1 == pytest.approx(vel0, abs=1e-9)


def test_propagate_leaves_an_unbound_body_unmoved():
    mu = units.G * 1.0
    pos = np.array([1.0, 0.0])
    vel = np.array([0.0, 5.0])          # far past escape velocity
    pos1, vel1 = orbits.propagate(mu, pos, vel, 10.0)
    assert pos1 == pytest.approx(pos)
    assert vel1 == pytest.approx(vel)


def test_intercept_angle_hits_a_stationary_point_exactly():
    mu = units.G * 1.0
    launch = np.array([1.0, 0.0])
    target = np.array([0.3, 1.6])
    speed = 1.3 * math.sqrt(mu / 1.0)   # comfortably above local circular speed
    result = orbits.solve_intercept_angle(mu, launch, speed, target)
    assert result["hit"]
    v = result["velocity"]
    assert abs(np.hypot(*v) - speed) < 1e-9
    el = orbits.elements_from_state(mu, launch, v)
    ang = math.atan2(target[1], target[0])
    nu = ang - el["lon_peri"]
    r_on_orbit = el["p"] / (1.0 + el["e"] * math.cos(nu))
    assert r_on_orbit == pytest.approx(float(np.hypot(*target)), abs=1e-6)


def test_intercept_reports_a_miss_when_the_target_is_out_of_reach():
    mu = units.G * 1.0
    launch = np.array([1.0, 0.0])
    target = np.array([500.0, 0.0])     # far beyond what a feeble throw can reach
    speed = 0.05 * math.sqrt(mu / 1.0)
    result = orbits.solve_intercept_angle(mu, launch, speed, target, samples=180)
    assert not result["hit"]
    assert result["flight_time"] is None
    assert result["miss"] > 0.0


def test_solve_intercept_leads_a_moving_target_onto_a_real_collision():
    """The fixed-point solve for a moving target must land ON the target at
    the predicted flight time, not just on its position when the solve began."""
    mu = units.G * 1.0
    # Target on a circular orbit at r=2.
    r_t = 2.0
    v_t = math.sqrt(mu / r_t)
    target_pos = np.array([0.0, r_t])
    target_vel = np.array([-v_t, 0.0])

    launch = np.array([1.0, 0.0])
    speed = 1.3 * math.sqrt(mu / 1.0)
    result = orbits.solve_intercept(mu, launch, speed, target_pos, target_vel)
    assert result["hit"]
    t = result["flight_time"]
    assert t is not None and t > 0.0

    # Fly the projectile forward exactly `t` days on its own solved orbit...
    proj_pos, _ = orbits.propagate(mu, launch, result["velocity"], t)
    # ...and fly the real target forward the same amount, independently.
    real_target_pos, _ = orbits.propagate(mu, target_pos, target_vel, t)
    assert proj_pos == pytest.approx(real_target_pos, abs=1e-3)
