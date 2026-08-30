import numpy as np
import pytest

from src import edits, materials, units
from src.body import KIND_PLANET, KIND_STAR
from src.sim import Simulation


@pytest.fixture
def sim():
    return Simulation()


def _index(sim, name):
    return next(i for i, b in enumerate(sim.world.bodies) if b.name == name)


def test_mass_scrub_is_logarithmic():
    """Mass spans twenty orders of magnitude; a linear scrub would be useless."""
    ratio = edits.scrub_mass(1e-6, 1) / 1e-6
    assert ratio > 1.0
    assert edits.scrub_mass(1e-3, 1) / 1e-3 == pytest.approx(ratio, rel=1e-9)


def test_mass_scrub_is_reversible():
    start = units.EARTH_MASS
    assert edits.scrub_mass(edits.scrub_mass(start, 5), -5) == pytest.approx(start)


def test_mass_scrub_clamps():
    assert edits.scrub_mass(1.0, 10_000) == pytest.approx(edits.MAX_MASS)
    assert edits.scrub_mass(1.0, -10_000) == pytest.approx(edits.MIN_MASS)


def test_density_scrub_clamps_to_the_material_range():
    assert edits.scrub_density_gcm3(1.0, 10_000) == pytest.approx(
        materials.MAX_DENSITY_GCM3)
    assert edits.scrub_density_gcm3(1.0, -10_000) == pytest.approx(
        materials.MIN_DENSITY_GCM3)


def test_setting_mass_keeps_density_and_grows_the_radius(sim):
    i = _index(sim, "Earth")
    before_density = edits.body_density(sim.world, i)
    before_radius = float(sim.world.radius[i])
    edits.set_mass(sim.world, i, units.EARTH_MASS * 8.0)
    assert edits.body_density(sim.world, i) == pytest.approx(before_density, rel=1e-9)
    # Eight times the mass at constant density is exactly twice the radius.
    assert sim.world.radius[i] == pytest.approx(before_radius * 2.0, rel=1e-9)


def test_squeezing_a_body_denser_really_does_shrink_it(sim):
    # Saturn, because eight times Earth's 5.51 g/cm3 is past the 25 g/cm3
    # ceiling and the clamp would (correctly) stop the radius halving.
    i = _index(sim, "Saturn")
    before_radius = float(sim.world.radius[i])
    before_mass = float(sim.world.mass[i])
    edits.set_density(sim.world, i, edits.body_density(sim.world, i) * 8.0)
    assert sim.world.radius[i] == pytest.approx(before_radius / 2.0, rel=1e-9)
    assert sim.world.mass[i] == pytest.approx(before_mass, rel=1e-9)


def test_density_cannot_be_pushed_past_degenerate_matter(sim):
    """Earth is already 5.5 g/cm3; the ceiling has to hold or the radius law
    starts producing bodies denser than anything real."""
    i = _index(sim, "Earth")
    edits.set_density(sim.world, i, units.from_gcm3(1e6))
    assert units.to_gcm3(edits.body_density(sim.world, i)) == pytest.approx(
        materials.MAX_DENSITY_GCM3, rel=1e-9)
    edits.set_density(sim.world, i, units.from_gcm3(1e-9))
    assert units.to_gcm3(edits.body_density(sim.world, i)) == pytest.approx(
        materials.MIN_DENSITY_GCM3, rel=1e-9)


def test_the_triple_is_always_consistent(sim):
    """The whole invariant of the module, checked over a random walk."""
    rng = np.random.default_rng(3)
    i = _index(sim, "Mars")
    for _ in range(200):
        choice = rng.integers(0, 3)
        if choice == 0:
            edits.set_mass(sim.world, i,
                           edits.scrub_mass(sim.world.mass[i], int(rng.integers(-9, 10))))
        elif choice == 1:
            gcm3 = units.to_gcm3(edits.body_density(sim.world, i))
            edits.set_density(sim.world, i, units.from_gcm3(
                edits.scrub_density_gcm3(gcm3, int(rng.integers(-9, 10)))))
        else:
            edits.cycle_material(sim.world, i, 1 if rng.random() < 0.5 else -1)

        mass = float(sim.world.mass[i])
        radius = float(sim.world.radius[i])
        assert mass > 0.0 and radius > 0.0
        assert np.isfinite(mass) and np.isfinite(radius)
        rebuilt = units.radius_from_mass_density(
            mass, units.density_from_mass_radius(mass, radius))
        assert rebuilt == pytest.approx(radius, rel=1e-9)


def test_changing_material_snaps_density_to_that_substance(sim):
    i = _index(sim, "Earth")
    edits.set_material(sim.world, i, materials.get("iron"))
    assert sim.world.bodies[i].material.key == "iron"
    assert edits.body_density(sim.world, i) == pytest.approx(
        materials.get("iron").density, rel=1e-9)


def test_cycling_material_wraps_around(sim):
    i = _index(sim, "Earth")
    seen = set()
    for _ in range(len(materials.MATERIALS) + 1):
        edits.cycle_material(sim.world, i)
        seen.add(sim.world.bodies[i].material.key)
    assert seen == {m.key for m in materials.MATERIALS}


def test_crossing_the_ignition_threshold_lights_a_star(sim):
    i = _index(sim, "Jupiter")
    assert sim.world.bodies[i].kind != KIND_STAR
    edits.set_mass(sim.world, i, units.STAR_MASS_THRESHOLD * 1.2)
    assert sim.world.bodies[i].kind == KIND_STAR


def test_dropping_below_the_threshold_puts_the_star_out(sim):
    i = _index(sim, "Sun")
    edits.set_mass(sim.world, i, units.EARTH_MASS)
    assert sim.world.bodies[i].kind == KIND_PLANET


def test_ignition_ratio_says_how_far_from_a_star_it_is(sim):
    i = _index(sim, "Jupiter")
    ratio = edits.ignition_ratio(sim.world, i)
    assert ratio == pytest.approx(units.STAR_MASS_THRESHOLD / units.JUPITER_MASS,
                                  rel=0.01)
    assert 80.0 < ratio < 90.0, "Jupiter should be about 84x too light"


def test_editing_mass_changes_the_forces_on_everything_else(sim):
    """Mass edits change the field for every body at once; stale accelerations
    would leave the system a step behind."""
    i = _index(sim, "Jupiter")
    before = sim.world.acc[_index(sim, "Earth")].copy()
    edits.set_mass(sim.world, i, units.JUPITER_MASS * 500.0)
    sim.refresh()
    after = sim.world.acc[_index(sim, "Earth")]
    assert not np.allclose(before, after)


def test_a_much_heavier_jupiter_wrecks_the_solar_system(sim):
    """The payoff: it must actually do something dramatic."""
    j = _index(sim, "Jupiter")
    earth = sim.world.bodies[_index(sim, "Earth")]
    edits.set_mass(sim.world, j, 0.05)          # 50x Jupiter, still not a star
    sim.refresh()
    from src import orbits
    before = orbits.elements_from_state(
        units.G, sim.world.pos[_index(sim, "Earth")], sim.world.vel[_index(sim, "Earth")])
    for _ in range(300):
        sim.advance(20.0, budget_s=10.0)
    index = sim.world.index_of(earth.id)
    if index is None:
        return                                   # swallowed entirely: dramatic enough
    after = orbits.elements_from_state(units.G, sim.world.pos[index],
                                       sim.world.vel[index])
    assert abs(after["e"] - before["e"]) > 0.01, "Earth's orbit barely noticed"


# --- temperature ------------------------------------------------------------------

def test_a_planet_can_be_given_a_temperature(sim):
    i = _index(sim, "Earth")
    edits.set_temperature(sim.world, i, 900.0)
    assert float(sim.world.temperature[i]) == pytest.approx(900.0)


def test_temperature_is_clamped_to_the_background_and_the_ceiling(sim):
    i = _index(sim, "Earth")
    edits.set_temperature(sim.world, i, -50.0)
    assert float(sim.world.temperature[i]) == pytest.approx(units.MIN_TEMPERATURE_K)
    edits.set_temperature(sim.world, i, 1.0e12)
    assert float(sim.world.temperature[i]) == pytest.approx(units.MAX_TEMPERATURE_K)


def test_temperature_scrubs_by_ratio_so_it_can_cross_five_orders_of_magnitude():
    """Linear steps cannot serve both a 3 K comet and a 100,000 K ember; the
    same argument that makes the mass scrub logarithmic."""
    up = edits.scrub_temperature(100.0, 1)
    assert up == pytest.approx(100.0 * edits.TEMPERATURE_NOTCH)
    assert edits.scrub_temperature(up, -1) == pytest.approx(100.0)


def test_temperature_scrub_clamps_at_both_ends():
    assert edits.scrub_temperature(300.0, 10_000) == pytest.approx(
        units.MAX_TEMPERATURE_K)
    assert edits.scrub_temperature(300.0, -10_000) == pytest.approx(
        units.MIN_TEMPERATURE_K)


def test_a_star_refuses_to_have_its_temperature_set(sim):
    """Its temperature is fusion's business and follows its mass.  The Edit
    panel offers no Temp row for one, and the setter enforces the same rule so
    no other caller can route around the missing button."""
    i = _index(sim, "Sun")
    before = float(sim.world.temperature[i])
    edits.set_temperature(sim.world, i, 10.0)
    assert float(sim.world.temperature[i]) == pytest.approx(before)


def test_a_planet_that_ignites_takes_its_stars_temperature_at_once(sim):
    i = _index(sim, "Earth")
    edits.set_mass(sim.world, i, units.STAR_MASS_THRESHOLD * 1.1)
    assert float(sim.world.temperature[i]) == pytest.approx(
        units.star_temperature(float(sim.world.mass[i])), rel=1e-9)


def test_a_collapsed_star_takes_its_hawking_temperature(sim):
    i = _index(sim, "Earth")
    edits.set_mass(sim.world, i, 10.0)
    edits.collapse(sim.world, i)
    assert float(sim.world.temperature[i]) == pytest.approx(
        units.hawking_temperature_k(10.0), rel=1e-9)


def test_a_planet_is_offered_four_fields_and_a_star_only_three(sim):
    earth = sim.world.bodies[_index(sim, "Earth")]
    sun = sim.world.bodies[_index(sim, "Sun")]
    assert edits.edit_fields(earth) == ("mass", "density", "temp", "material")
    assert edits.edit_fields(sun) == ("mass", "density", "material")
    assert "temp" not in edits.edit_fields(sun)
