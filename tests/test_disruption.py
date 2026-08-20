import math

import numpy as np

from src import collisions, materials, units
from src.world import World


def _si(q):
    """Sim specific energy -> J/kg."""
    return q * units.SPECIFIC_ENERGY_SI


def test_strength_branch_matches_published_basalt_fit():
    # Benz & Asphaug 1999 basalt, converted to SI: 608 * R**-0.38 J/kg.  Kept
    # to sizes where the gravity term is still negligible (it is already 19%
    # of the total by R=100m for this density).
    rho = units.from_gcm3(3.0)
    for r_m in (0.01, 0.1, 1.0, 10.0):
        r = r_m / units.AU_M
        got = _si(units.disruption_threshold(r, rho))
        assert math.isclose(got, 608.0 * r_m ** -0.38, rel_tol=0.02)


def test_gravity_branch_dominates_for_a_hundred_kilometre_body():
    rho = units.from_gcm3(3.0)
    r = 1.0e5 / units.AU_M
    got = _si(units.disruption_threshold(r, rho))
    # ~3e5 J/kg is the measured value for a 100 km basalt target.
    assert 1.0e5 < got < 1.0e6


def test_the_curve_has_a_minimum_between_a_boulder_and_a_mountain():
    rho = units.from_gcm3(3.0)
    lo = units.disruption_threshold(1.0 / units.AU_M, rho)
    hi = units.disruption_threshold(1.0e5 / units.AU_M, rho)
    mid = units.disruption_threshold(110.0 / units.AU_M, rho)
    assert mid < lo and mid < hi          # the curve has its minimum in between


def test_toughness_scales_only_the_strength_branch():
    rho = units.from_gcm3(3.0)
    small = 10.0 / units.AU_M
    big = 1.0e5 / units.AU_M
    assert units.disruption_threshold(small, rho, 3.0) > \
           2.5 * units.disruption_threshold(small, rho, 1.0)
    assert math.isclose(units.disruption_threshold(big, rho, 3.0),
                        units.disruption_threshold(big, rho, 1.0), rel_tol=0.01)


def test_every_material_declares_toughness_and_vapour_point():
    for m in materials.MATERIALS:
        assert m.toughness > 0.0
        assert m.vapour_temperature_k > 0.0


# --- outcome classification -----------------------------------------------------

def _pair(m1, r1, m2, r2, sep_factor, v, offset=0.0):
    """Two bodies on a collision course, run through the real swept detector
    so ``b`` and ``v_impact`` are the genuine geometry, not hand-picked."""
    world = World(8)
    rock = materials.get("rock")
    world.add("A", rock, m1, r1, np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    world.add("B", rock, m2, r2, np.array([sep_factor * (r1 + r2), offset]),
              np.array([-v, 0.0]))
    prev = world.pos.copy()
    world.pos[1] = np.array([-sep_factor * (r1 + r2), offset])
    hits = collisions.find_collisions(world, prev)
    return world, hits[0]


def test_a_gentle_touch_accretes():
    m = units.EARTH_MASS
    world, c = _pair(m, units.EARTH_RADIUS, m * 1e-6, units.EARTH_RADIUS * 0.01,
                     2.0, 1e-6)
    assert collisions.classify(world, c) == collisions.ACCRETE


def test_a_fast_head_on_hit_disrupts():
    m = units.EARTH_MASS * 1e-6
    r = units.radius_from_mass_density(m, units.from_gcm3(3.0))
    world, c = _pair(m, r, m, r, 2.0, 0.5)
    assert collisions.classify(world, c) == collisions.DISRUPT


def test_equal_bodies_that_only_clip_each_other_hit_and_run():
    m = units.EARTH_MASS
    r = units.EARTH_RADIUS
    world, c = _pair(m, r, m, r, 4.0, 0.02, offset=1.9 * r)
    assert c.b > 0.9
    assert collisions.classify(world, c) == collisions.GRAZE


def test_an_asteroid_can_never_hit_and_run_off_the_sun():
    world = World(8)
    world.add("Sun", materials.get("hydrogen"), 1.0, units.SUN_RADIUS,
              np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    r = 50.0 / units.AU_KM
    m = units.FOUR_THIRDS_PI * materials.get("rock").density * r ** 3
    world.add("Rock", materials.get("rock"), m, r,
              np.array([units.SUN_RADIUS * 0.999, 0.0]), np.array([-0.05, 0.0]))
    prev = world.pos.copy()
    world.pos[1] = np.array([-units.SUN_RADIUS * 0.999, 0.0])
    c = collisions.find_collisions(world, prev)[0]
    assert collisions.classify(world, c) == collisions.ACCRETE
