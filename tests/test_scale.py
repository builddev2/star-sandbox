import numpy as np
import pytest

from src import units
from src.render import scale

WIDE_ZOOM = 1728 / 70.0        # whole solar system across a 1728px buffer
H = 1084


def _px(r, zoom=WIDE_ZOOM, real=False):
    return scale.display_radius_px(r, zoom, H, real_size=real)


def test_size_order_is_preserved_when_zoomed_out():
    """The whole point: Jupiter must look bigger than Earth, which must look
    bigger than the Moon, even though all three are sub-pixel at true scale."""
    sun = _px(units.SUN_RADIUS)
    jupiter = _px(units.JUPITER_RADIUS)
    earth = _px(units.EARTH_RADIUS)
    moon = _px(1737.4 / units.AU_KM)
    assert sun > jupiter > earth > moon


def test_even_a_one_kilometre_rock_stays_visible():
    u = H / scale.REF_HEIGHT
    tiny = _px(1.0 / units.AU_KM)          # a 1 km rock
    assert tiny >= scale.FLOOR_MIN * u
    assert tiny >= 2.0, "debris must not vanish into a single pixel"


def test_small_bodies_are_still_told_apart():
    """The floor must not be so high that Mercury and the Moon flatten into the
    same dot -- distinguishing sizes is the whole job of this law."""
    mercury = _px(2439.7 / units.AU_KM)
    moon = _px(1737.4 / units.AU_KM)
    assert mercury > moon


def test_the_floor_is_capped_so_nothing_swallows_the_view():
    """A naive log exaggeration lets big bodies grow without bound until one disc
    covers the screen.  The floor is clamped, so it cannot."""
    u = H / scale.REF_HEIGHT
    # At a zoom where true size is negligible, the floor alone decides.
    assert _px(units.SUN_RADIUS * 1000.0, zoom=1e-6) == pytest.approx(scale.FLOOR_MAX * u)
    assert scale.FLOOR_MAX * u < 0.05 * H


def test_the_sun_stays_a_small_fraction_of_the_view():
    assert _px(units.SUN_RADIUS) < 0.05 * H


def test_the_sun_is_drawn_smaller_than_earths_orbit():
    """Otherwise Earth is drawn inside the star, which is both wrong and the
    exact failure the floor cap exists to prevent."""
    earth_orbit_px = 1.0 * WIDE_ZOOM
    assert _px(units.SUN_RADIUS) < earth_orbit_px


def test_the_sun_is_still_the_biggest_thing_on_screen():
    assert _px(units.SUN_RADIUS) > _px(units.JUPITER_RADIUS)


def test_true_scale_takes_over_continuously_when_zoomed_in():
    """No mode switch, no pop: as you zoom in, the true size overtakes the floor
    and from then on the drawing is honest."""
    r = units.EARTH_RADIUS
    previous = _px(r, zoom=1e2)
    for exponent in range(3, 12):
        current = _px(r, zoom=10.0 ** exponent)
        assert current >= previous
        previous = current
    # Far enough in, it is exactly true scale.
    huge = 1e11
    assert _px(r, zoom=huge) == pytest.approx(huge * r, rel=1e-12)


def test_real_size_mode_collapses_everything_to_a_dot():
    for r in (units.SUN_RADIUS, units.JUPITER_RADIUS, units.EARTH_RADIUS):
        assert _px(r, real=True) == pytest.approx(1.0)


def test_real_size_mode_is_still_honest_when_zoomed_in():
    r = units.EARTH_RADIUS
    assert _px(r, zoom=1e10, real=True) == pytest.approx(1e10 * r)


def test_floor_scales_with_resolution():
    """Constants tuned at 360px tall must not shrink to specks at 1084."""
    small = scale.display_radius_px(units.EARTH_RADIUS, WIDE_ZOOM, 360)
    large = scale.display_radius_px(units.EARTH_RADIUS, WIDE_ZOOM, 1084)
    assert large == pytest.approx(small * 1084 / 360, rel=1e-9)


def test_vectorised_matches_scalar():
    radii = np.array([units.SUN_RADIUS, units.EARTH_RADIUS, units.JUPITER_RADIUS])
    batch = scale.display_radii_px(radii, WIDE_ZOOM, H)
    for k, r in enumerate(radii):
        assert batch[k] == pytest.approx(_px(r))


def test_the_moon_separates_from_earth_when_you_zoom_in():
    """At system zoom the Moon is inside Earth's drawn disc, which is unavoidable.
    Zoom to the Earth-Moon system and they must be visibly distinct."""
    zoom = 1728 / 0.02            # 0.02 AU across the buffer
    earth = scale.display_radius_px(units.EARTH_RADIUS, zoom, H)
    separation = 0.00256955 * zoom
    assert separation > earth * 2, "Moon still buried inside Earth's disc"


def test_a_craft_is_drawn_from_a_stand_in_radius_not_its_true_hull():
    import numpy as np
    from src import materials, units
    from src.body import KIND_CRAFT
    from src.render import scale
    from src.world import World

    world = World()
    world.add("Shuttle", materials.SPACECRAFT, units.EARTH_MASS * 1e-18,
              units.CRAFT_RADIUS, np.array([1.0, 0.0]), np.zeros(2),
              kind=KIND_CRAFT)
    visual = scale.visual_radii(world)
    assert abs(float(visual[0]) - units.craft_apparent_radius()) < 1e-18
    # The physics array is untouched.
    assert abs(float(world.radius[0]) - units.CRAFT_RADIUS) < 1e-18
