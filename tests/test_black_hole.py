"""Black hole physics: formulas, collapse, collisions and tides.

Pure numpy -- no pygame anywhere in this file, so it exercises the same
layer tests/test_layering.py protects.
"""

import math

import pytest

from src import units


def test_speed_of_light_in_simulation_units():
    """c is derived from the conversion factors, never written down twice."""
    assert units.C == pytest.approx(173.1446326742, abs=1e-9)


def test_schwarzschild_radius_matches_the_textbook_value():
    """2GM/c^2 for one solar mass is 2.953 km."""
    r_s = units.schwarzschild_radius(1.0)
    assert units.to_km(r_s) == pytest.approx(2.95325, rel=1e-5)


def test_schwarzschild_radius_is_linear_in_mass():
    assert units.schwarzschild_radius(10.0) == pytest.approx(
        10.0 * units.schwarzschild_radius(1.0))


def test_escape_velocity_at_the_horizon_is_exactly_c():
    """Falls out of the existing formula with no special case, which is a
    consistency check on the whole unit system."""
    for mass in (1.0, 10.0, 150.0):
        r_s = units.schwarzschild_radius(mass)
        assert units.escape_velocity(mass, r_s) == pytest.approx(units.C, rel=1e-12)


def test_density_scales_as_inverse_mass_squared():
    """Bigger black holes are less dense -- the counterintuitive fact."""
    light = units.density_from_mass_radius(1.0, units.schwarzschild_radius(1.0))
    heavy = units.density_from_mass_radius(100.0, units.schwarzschild_radius(100.0))
    assert heavy / light == pytest.approx(1.0 / 100.0 ** 2, rel=1e-9)


def test_shadow_is_larger_than_the_photon_sphere_is_larger_than_the_horizon():
    r_s = units.schwarzschild_radius(10.0)
    assert units.photon_sphere_radius(10.0) == pytest.approx(1.5 * r_s)
    assert units.shadow_radius(10.0) == pytest.approx(math.sqrt(27.0) / 2.0 * r_s)
    assert units.shadow_radius(10.0) > units.photon_sphere_radius(10.0) > r_s


@pytest.mark.parametrize("spin,gm_c2", [
    (0.0, 6.0000),
    (0.5, 4.2330),
    (0.9, 2.3209),
    (0.998, 1.2370),
])
def test_isco_matches_bardeen_press_teukolsky(spin, gm_c2):
    """Prograde Kerr ISCO, in units of GM/c^2 -- which is half of r_s."""
    mass = 7.0
    expected = gm_c2 * 0.5 * units.schwarzschild_radius(mass)
    assert units.isco_radius(mass, spin) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("spin,efficiency", [
    (0.0, 0.0572),
    (0.998, 0.3210),
])
def test_accretion_efficiency_rises_with_spin(spin, efficiency):
    assert units.accretion_efficiency(spin) == pytest.approx(efficiency, abs=5e-4)


def test_hawking_temperature_is_absurdly_cold_for_a_stellar_hole():
    """6e-8 K -- colder than the CMB, so such a hole grows rather than evaporates."""
    assert units.hawking_temperature_k(1.0) == pytest.approx(6.169e-8, rel=1e-3)
    assert units.hawking_temperature_k(10.0) == pytest.approx(6.169e-9, rel=1e-3)


def test_evaporation_time_scales_as_mass_cubed():
    assert units.evaporation_years(10.0) / units.evaporation_years(1.0) == \
        pytest.approx(1000.0, rel=1e-9)


def test_apparent_radius_is_big_enough_to_draw():
    """The horizon is invisible; the apparent radius stands in for the disk.

    A 10 solar mass hole must read at least as large as the Sun does, or it
    draws smaller than a fragment of ejecta.
    """
    assert units.apparent_radius(10.0) > units.SUN_RADIUS
    assert units.apparent_radius(3.0) > units.SUN_RADIUS


# --- the kind and the material -------------------------------------------------

def test_black_hole_kind_exists_and_bodies_report_it():
    from src.body import KIND_BLACK_HOLE, Body
    from src import materials
    b = Body(1, "Hole", materials.SINGULARITY, kind=KIND_BLACK_HOLE)
    assert b.is_black_hole
    assert not b.is_star


def test_ordinary_bodies_are_not_black_holes():
    from src.body import Body
    from src import materials
    assert not Body(1, "Earth", materials.get("rock")).is_black_hole


def test_new_bodies_start_with_no_spin_and_no_disk():
    from src.body import Body
    from src import materials
    b = Body(1, "Earth", materials.get("rock"))
    assert b.spin == 0.0
    assert b.disk == 0.0


def test_singularity_is_reachable_by_key_but_not_cycleable():
    """It must not appear in MATERIALS, or the Edit tool's material cycle and
    nearest_by_density could both land on it."""
    from src import materials
    assert materials.get("singularity") is materials.SINGULARITY
    assert materials.SINGULARITY not in materials.MATERIALS
    assert all(m.key != "singularity" for m in materials.MATERIALS)


def test_nearest_by_density_can_never_return_the_singularity():
    from src import materials
    for density_gcm3 in (0.01, 1.0, 25.0, 1e12):
        chosen = materials.nearest_by_density(units.from_gcm3(density_gcm3))
        assert chosen.key != "singularity"


def test_singularity_never_boils_and_never_shatters():
    from src import materials
    assert materials.SINGULARITY.vapour_temperature_k == math.inf
    assert materials.SINGULARITY.toughness > 1e6


# --- collapse and editing ------------------------------------------------------

@pytest.fixture
def world_with_sun():
    from src import materials, units as u
    from src.body import KIND_STAR
    from src.world import World
    import numpy as np
    w = World()
    w.add("Sun", materials.get("hydrogen"), 1.0, u.SUN_RADIUS,
          np.zeros(2), np.zeros(2), kind=KIND_STAR)
    return w


def test_collapse_turns_a_star_into_a_black_hole(world_with_sun):
    from src import edits
    from src.body import KIND_BLACK_HOLE
    edits.collapse(world_with_sun, 0)
    body = world_with_sun.bodies[0]
    assert body.kind == KIND_BLACK_HOLE
    assert body.material.key == "singularity"
    assert body.disk > 0.0


def test_collapse_conserves_mass_exactly(world_with_sun):
    """Collapsing the Sun gives a 1 solar mass hole, which nature could not
    make -- but inventing two extra solar masses to respect the TOV limit is
    worse than allowing it."""
    from src import edits
    edits.collapse(world_with_sun, 0)
    assert float(world_with_sun.mass[0]) == pytest.approx(1.0)


def test_collapse_sets_the_radius_to_the_horizon(world_with_sun):
    from src import edits
    edits.collapse(world_with_sun, 0)
    assert float(world_with_sun.radius[0]) == pytest.approx(
        units.schwarzschild_radius(1.0))
    assert float(world_with_sun.contact_radius[0]) == pytest.approx(
        float(world_with_sun.radius[0]))


def test_collapse_clears_the_trail_and_the_rings(world_with_sun):
    from src import edits
    world_with_sun.bodies[0].rings = True
    world_with_sun.bodies[0].trail.maybe_append((1.0, 2.0), None)
    edits.collapse(world_with_sun, 0)
    assert not world_with_sun.bodies[0].rings
    assert world_with_sun.bodies[0].trail.count == 0


def test_collapse_is_idempotent(world_with_sun):
    from src import edits
    edits.collapse(world_with_sun, 0)
    world_with_sun.bodies[0].spin = 0.1
    edits.collapse(world_with_sun, 0)
    assert world_with_sun.bodies[0].spin == 0.1     # not reset a second time


def test_a_star_scrubbed_past_the_threshold_collapses(world_with_sun):
    from src import edits
    from src.body import KIND_BLACK_HOLE
    edits.set_mass(world_with_sun, 0, units.COLLAPSE_MASS_THRESHOLD)
    assert world_with_sun.bodies[0].kind == KIND_BLACK_HOLE


def test_a_star_below_the_threshold_stays_a_star(world_with_sun):
    from src import edits
    from src.body import KIND_STAR
    edits.set_mass(world_with_sun, 0, units.COLLAPSE_MASS_THRESHOLD * 0.99)
    assert world_with_sun.bodies[0].kind == KIND_STAR


def test_collapse_is_one_way(world_with_sun):
    """Losing mass leaves a smaller black hole, never a star again."""
    from src import edits
    from src.body import KIND_BLACK_HOLE
    edits.collapse(world_with_sun, 0)
    edits.set_mass(world_with_sun, 0, 1e-6)
    assert world_with_sun.bodies[0].kind == KIND_BLACK_HOLE


def test_black_hole_radius_follows_the_horizon_not_the_density(world_with_sun):
    from src import edits
    edits.collapse(world_with_sun, 0)
    edits.set_mass(world_with_sun, 0, 20.0)
    assert float(world_with_sun.radius[0]) == pytest.approx(
        units.schwarzschild_radius(20.0))
    assert float(world_with_sun.contact_radius[0]) == pytest.approx(
        float(world_with_sun.radius[0]))


def test_max_mass_is_the_collapse_threshold():
    from src import edits
    assert edits.MAX_MASS == units.COLLAPSE_MASS_THRESHOLD


def test_spin_scrub_is_clamped_to_the_thorne_limit():
    from src import edits
    assert edits.scrub_spin(0.99, +100) == pytest.approx(units.MAX_SPIN)
    assert edits.scrub_spin(0.01, -100) == 0.0


def test_disk_scrub_is_clamped_to_zero_and_one():
    from src import edits
    assert edits.scrub_disk(0.9, +100) == 1.0
    assert edits.scrub_disk(0.1, -100) == 0.0


def test_edit_fields_depend_on_the_kind(world_with_sun):
    from src import edits
    star = world_with_sun.bodies[0]
    assert edits.edit_fields(star) == ("mass", "density", "material")
    edits.collapse(world_with_sun, 0)
    assert edits.edit_fields(star) == ("mass", "spin", "disk")


def test_cycle_material_refuses_to_touch_a_black_hole(world_with_sun):
    """SINGULARITY is not in MATERIALS, so cycling would raise ValueError."""
    from src import edits
    edits.collapse(world_with_sun, 0)
    edits.cycle_material(world_with_sun, 0, 1)
    assert world_with_sun.bodies[0].material.key == "singularity"
