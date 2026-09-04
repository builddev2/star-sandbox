import numpy as np
import pytest

from src import edits, materials, units
from src.body import KIND_BLACK_HOLE, KIND_CRAFT, KIND_PLANET, KIND_STAR
from src.sim import Simulation
from src.world import World


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


def test_a_craft_is_offered_no_edit_fields():
    """A 37 m hull at any realistic craft mass derives a density of a few
    thousandths of a g/cm3, nothing like SPACECRAFT's 2.70 -- so the mass,
    density and material edits would misbehave (an inspector readout that
    lies, set_mass inflating the hull, set_density relabelling it "Rock" and
    losing the SPACECRAFT material entirely). A craft's real editable state is
    fuel and flight plan, both in the Fly panel, not here."""
    world = World()
    craft = world.add("Shuttle", materials.SPACECRAFT, units.EARTH_MASS * 1e-18,
                      units.CRAFT_RADIUS, [1.0, 0.0], [0.0, 0.0], kind=KIND_CRAFT)
    assert edits.edit_fields(craft) == ()


# --- locking ----------------------------------------------------------------------

def test_every_body_starts_with_nothing_locked(sim):
    for body in sim.world.bodies:
        assert body.locked_fields == frozenset()


def test_the_lock_set_is_a_frozenset(sim):
    """Rebinding is the only way to change it, so two bodies can never end up
    sharing one mutable set."""
    body = sim.world.bodies[_index(sim, "Earth")]
    assert isinstance(body.locked_fields, frozenset)


def test_a_body_added_to_the_world_starts_unlocked(sim):
    """Every fragment path -- shatter, graze, cratering ejecta, tidal debris --
    goes through World.add, so this is also what says debris never inherits its
    parent's locks."""
    body = sim.world.add("Pebble", materials.get("rock"), 1e-9, 1e-6,
                         (5.0, 0.0), (0.0, 1.0))
    assert body.locked_fields == frozenset()


def test_toggle_lock_flips_one_field_and_leaves_the_others(sim):
    body = sim.world.bodies[_index(sim, "Earth")]
    assert not edits.is_locked(body, "mass")
    edits.toggle_lock(body, "mass")
    assert edits.is_locked(body, "mass")
    assert not edits.is_locked(body, "density")
    edits.toggle_lock(body, "mass")
    assert not edits.is_locked(body, "mass")


def test_unlock_all_empties_the_set(sim):
    body = sim.world.bodies[_index(sim, "Earth")]
    for field in ("mass", "density", "temp", "material"):
        edits.toggle_lock(body, field)
    edits.unlock_all(body)
    assert body.locked_fields == frozenset()


def test_unlock_all_with_nothing_locked_is_harmless(sim):
    body = sim.world.bodies[_index(sim, "Earth")]
    edits.unlock_all(body)
    assert body.locked_fields == frozenset()


def test_a_lock_survives_the_body_changing_kind(sim):
    """A star that cools back into a planet finds its old temperature lock
    still in place.  A kind change is not a reset -- while the body is a star
    the lock is simply inert, because the panel offers no Temp row for one."""
    i = _index(sim, "Earth")
    body = sim.world.bodies[i]
    edits.toggle_lock(body, "temp")
    edits.set_mass(sim.world, i, units.STAR_MASS_THRESHOLD * 1.1)
    assert body.kind == KIND_STAR
    edits.set_mass(sim.world, i, units.EARTH_MASS)
    assert body.kind == KIND_PLANET
    assert edits.is_locked(body, "temp")


def _black_hole(sim):
    """Ten solar masses of Earth, crushed.  Spin and disk take their defaults,
    0.7 and 0.6, so a test can set something else and see it."""
    i = _index(sim, "Earth")
    edits.set_mass(sim.world, i, 10.0)
    edits.collapse(sim.world, i)
    return i


def test_a_locked_mass_refuses_to_change(sim):
    i = _index(sim, "Earth")
    body = sim.world.bodies[i]
    mass = float(sim.world.mass[i])
    radius = float(sim.world.radius[i])
    edits.toggle_lock(body, "mass")
    edits.set_mass(sim.world, i, mass * 10.0)
    assert float(sim.world.mass[i]) == pytest.approx(mass)
    assert float(sim.world.radius[i]) == pytest.approx(radius)
    edits.toggle_lock(body, "mass")
    edits.set_mass(sim.world, i, mass * 10.0)
    assert float(sim.world.mass[i]) == pytest.approx(mass * 10.0)


def test_a_locked_mass_cannot_be_ignited(sim):
    """The guard is at the top of the setter, so ``_refresh_kind`` is never
    reached and the body does not quietly change kind on the way out."""
    i = _index(sim, "Jupiter")
    body = sim.world.bodies[i]
    edits.toggle_lock(body, "mass")
    edits.set_mass(sim.world, i, units.STAR_MASS_THRESHOLD * 2.0)
    assert body.kind == KIND_PLANET


def test_a_locked_density_refuses_to_change(sim):
    # Saturn: 0.687 g/cm3, so four times that is still inside the material range
    # and the clamp cannot be what holds the value still.
    i = _index(sim, "Saturn")
    body = sim.world.bodies[i]
    density = edits.body_density(sim.world, i)
    radius = float(sim.world.radius[i])
    material = body.material.key
    edits.toggle_lock(body, "density")
    edits.set_density(sim.world, i, density * 4.0)
    assert edits.body_density(sim.world, i) == pytest.approx(density, rel=1e-9)
    assert float(sim.world.radius[i]) == pytest.approx(radius)
    # set_density also relabels the composition; a refused edit relabels nothing.
    assert body.material.key == material
    edits.toggle_lock(body, "density")
    edits.set_density(sim.world, i, density * 4.0)
    assert edits.body_density(sim.world, i) == pytest.approx(density * 4.0, rel=1e-9)


def test_a_locked_temperature_refuses_to_change(sim):
    i = _index(sim, "Earth")
    body = sim.world.bodies[i]
    before = float(sim.world.temperature[i])
    edits.toggle_lock(body, "temp")
    edits.set_temperature(sim.world, i, 900.0)
    assert float(sim.world.temperature[i]) == pytest.approx(before)
    edits.toggle_lock(body, "temp")
    edits.set_temperature(sim.world, i, 900.0)
    assert float(sim.world.temperature[i]) == pytest.approx(900.0)


def test_a_locked_spin_refuses_to_change(sim):
    i = _black_hole(sim)
    body = sim.world.bodies[i]
    before = body.spin
    edits.toggle_lock(body, "spin")
    edits.set_spin(sim.world, i, 0.1)
    assert body.spin == pytest.approx(before)
    edits.toggle_lock(body, "spin")
    edits.set_spin(sim.world, i, 0.1)
    assert body.spin == pytest.approx(0.1)


def test_a_locked_disk_refuses_to_change(sim):
    i = _black_hole(sim)
    body = sim.world.bodies[i]
    before = body.disk
    edits.toggle_lock(body, "disk")
    edits.set_disk(sim.world, i, 0.2)
    assert body.disk == pytest.approx(before)
    edits.toggle_lock(body, "disk")
    edits.set_disk(sim.world, i, 0.2)
    assert body.disk == pytest.approx(0.2)


def test_a_locked_material_refuses_to_change(sim):
    i = _index(sim, "Earth")
    body = sim.world.bodies[i]
    key = body.material.key
    radius = float(sim.world.radius[i])
    edits.toggle_lock(body, "material")
    edits.set_material(sim.world, i, materials.get("iron"))
    assert body.material.key == key
    assert float(sim.world.radius[i]) == pytest.approx(radius)
    edits.cycle_material(sim.world, i, 1)
    assert body.material.key == key
    edits.toggle_lock(body, "material")
    edits.cycle_material(sim.world, i, 1)
    assert body.material.key != key


def test_a_lock_guards_only_its_own_field(sim):
    """The worked example the design is built around: pin the mass, then scrub
    density, temperature and composition freely."""
    i = _index(sim, "Earth")
    body = sim.world.bodies[i]
    mass = float(sim.world.mass[i])
    edits.toggle_lock(body, "mass")
    edits.set_density(sim.world, i, edits.body_density(sim.world, i) * 1.5)
    edits.set_temperature(sim.world, i, 500.0)
    edits.set_material(sim.world, i, materials.get("iron"))
    assert float(sim.world.mass[i]) == pytest.approx(mass)
    assert body.material.key == "iron"
    assert float(sim.world.temperature[i]) == pytest.approx(500.0)


def test_a_density_lock_does_not_chase_density_through_a_material_swap(sim):
    """Decided during design: a lock blocks direct edits to its own field and
    nothing else.  set_material is editing material, not density."""
    i = _index(sim, "Earth")
    body = sim.world.bodies[i]
    edits.toggle_lock(body, "density")
    edits.set_material(sim.world, i, materials.get("iron"))
    assert edits.body_density(sim.world, i) == pytest.approx(
        materials.get("iron").density, rel=1e-9)


def test_collapse_is_not_stopped_by_a_locked_mass(sim):
    """Collapse conserves mass, so it is not a mass edit -- and a lock reaches
    no further than the setters it guards."""
    i = _index(sim, "Earth")
    body = sim.world.bodies[i]
    edits.set_mass(sim.world, i, 10.0)
    edits.toggle_lock(body, "mass")
    edits.collapse(sim.world, i)
    assert body.kind == KIND_BLACK_HOLE
    assert float(sim.world.mass[i]) == pytest.approx(10.0)


def test_a_star_still_takes_its_pinned_temperature_with_temp_locked(sim):
    """The pin is not an edit: ``_refresh_kind`` writes the array directly, so a
    body that ignites reads as a star at once, whatever its lock says."""
    i = _index(sim, "Earth")
    body = sim.world.bodies[i]
    edits.toggle_lock(body, "temp")
    edits.set_mass(sim.world, i, units.STAR_MASS_THRESHOLD * 1.1)
    assert float(sim.world.temperature[i]) == pytest.approx(
        units.star_temperature(float(sim.world.mass[i])), rel=1e-9)
