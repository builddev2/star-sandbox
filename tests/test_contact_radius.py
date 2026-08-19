import numpy as np
import pytest

from src import collisions, materials, scenarios, units
from src.render import scale
from src.world import World

ROCK = materials.get("rock")


def test_contact_radius_defaults_to_the_true_radius():
    world = World(4)
    body = world.add("Rock", ROCK, units.EARTH_MASS, units.EARTH_RADIUS,
                     np.zeros(2), np.zeros(2))
    i = world.index_of(body.id)
    assert world.contact_radius[i] == pytest.approx(world.radius[i])


def test_contact_radii_au_never_reaches_a_neighbouring_orbit():
    """Every body of the real solar system, at every zoom the camera allows
    (render/camera.MIN_ZOOM .. MAX_ZOOM), must have a contact radius at most
    half its distance to whatever is nearest.

    render/scale.contact_radii_au's own MAX_AIM_FACTOR cap is not enough by
    itself -- Jupiter and Io sit close enough that even 8x Jupiter's true
    radius reaches past half that gap at the camera's widest zoom-out -- so
    this exercises the full pipeline: the render-only cap followed by
    World.clamp_contact_radii's geometry-aware one.
    """
    world = scenarios.solar_system()
    pos = world.positions
    n = world.n
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
    np.fill_diagonal(dist, np.inf)
    nearest = dist.min(axis=1)

    for zoom in (1e-2, 1.0, 10.0, 100.0, 1000.0, 1.0e6, 1e12):
        drawn = scale.contact_radii_au(world.radii, zoom, 360)
        contact = world.clamp_contact_radii(drawn)
        assert np.all(contact <= 0.5 * nearest + 1e-15), "zoom=%g" % zoom


def test_an_asteroid_aimed_at_the_drawn_sun_is_destroyed():
    """The reported bug: at a zoom where the Sun's true disc is invisible, a
    shot well inside the *drawn* disc but well outside the *true* one flew
    straight through untouched before contact radii existed.

    25% of the drawn radius is outside the true radius (~5% of drawn here)
    but inside the MAX_AIM_FACTOR-capped contact radius (~40% of drawn) --
    the shot this feature is actually for, not the full uncapped disc.
    """
    zoom = 64.0
    world = World(8)
    world.add("Sun", materials.get("hydrogen"), 1.0, units.SUN_RADIUS,
              np.zeros(2), np.zeros(2))
    world.contact_radius[:world.n] = scale.contact_radii_au(world.radii, zoom, 360)

    drawn_au = scale.display_radius_px(units.SUN_RADIUS, zoom, 360) / zoom
    b_par = 0.25 * drawn_au
    assert b_par > world.radius[0], "must be outside the true radius"
    assert b_par < world.contact_radius[0], "must be inside the capped contact radius"

    rock = materials.get("rock")
    r = 50.0 / units.AU_KM
    m = units.FOUR_THIRDS_PI * rock.density * r ** 3
    world.add("Rock", rock, m, r, np.array([0.5, b_par]), np.array([-0.06, 0.0]))
    world.contact_radius[world.n - 1] = r      # a thrown body's own contact
                                                # radius is refreshed the same
                                                # way; asteroids get no boost
                                                # here since none was applied

    hit = False
    prev = world.pos.copy()
    for _ in range(4000):
        prev[:] = world.pos[:]
        world.pos[1] += world.vel[1] * 0.05
        if collisions.find_collisions(world, prev):
            hit = True
            break
    assert hit, "an asteroid aimed at the drawn Sun must actually hit it"


def test_clamp_contact_radii_caps_at_half_the_gap_to_a_close_neighbour():
    world = World(4)
    world.add("A", ROCK, units.EARTH_MASS, units.EARTH_RADIUS,
              np.array([0.0, 0.0]), np.zeros(2))
    world.add("B", ROCK, units.EARTH_MASS, units.EARTH_RADIUS,
              np.array([0.01, 0.0]), np.zeros(2))
    # Ask for something absurdly oversized; the clamp must win.
    candidate = np.array([1.0, 1.0])
    clamped = world.clamp_contact_radii(candidate)
    assert np.all(clamped <= 0.005 + 1e-15)


def test_clamp_contact_radii_never_drops_below_the_true_radius():
    world = World(4)
    world.add("A", ROCK, units.EARTH_MASS, units.EARTH_RADIUS,
              np.array([0.0, 0.0]), np.zeros(2))
    world.add("B", ROCK, units.EARTH_MASS, units.EARTH_RADIUS,
              np.array([1e-9, 0.0]), np.zeros(2))    # pathologically close
    candidate = np.array([1.0, 1.0])
    clamped = world.clamp_contact_radii(candidate)
    assert np.all(clamped >= world.radii)


def test_real_size_mode_uses_true_radii_for_contact_too():
    zoom = 1000.0
    contact = scale.contact_radii_au([units.SUN_RADIUS], zoom, 360, real_size=True)
    assert contact[0] == pytest.approx(units.SUN_RADIUS, rel=1e-9)
