import math

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
    import numpy as np
    mu = np.float64(units.G)
    pos, vel = orbits.state_from_elements(mu, 1.0, 0.1, 0.0, 0.0)
    el = orbits.elements_from_state(mu, np.asarray(pos), np.asarray(vel))
    assert el["bound"] is True
    assert type(el["bound"]) is bool
