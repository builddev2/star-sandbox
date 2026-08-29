"""Black hole rendering: visual radii, the sprite, and the swallow effect."""

import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    pygame.init()
    pygame.display.set_mode((320, 200))
    yield
    pygame.quit()


@pytest.fixture
def hole_world():
    """A 10 solar mass black hole and the Earth, for comparison."""
    from src import edits, materials, units
    from src.body import KIND_STAR
    from src.world import World
    w = World()
    w.add("Hole", materials.get("hydrogen"), 10.0, units.SUN_RADIUS,
          np.zeros(2), np.zeros(2), kind=KIND_STAR)
    edits.collapse(w, 0)
    w.add("Earth", materials.get("rock"), units.EARTH_MASS, units.EARTH_RADIUS,
          np.array([1.0, 0.0]), np.zeros(2))
    return w


# --- visual radii ---------------------------------------------------------------

def test_visual_radii_replaces_only_the_black_hole(hole_world):
    from src import units
    from src.render import scale
    visual = scale.visual_radii(hole_world)
    assert visual[0] == pytest.approx(units.apparent_radius(10.0))
    assert visual[1] == pytest.approx(units.EARTH_RADIUS)


def test_visual_radii_does_not_mutate_the_world(hole_world):
    from src import units
    from src.render import scale
    scale.visual_radii(hole_world)
    assert float(hole_world.radius[0]) == pytest.approx(
        units.schwarzschild_radius(10.0))


def test_without_the_floor_fix_a_black_hole_draws_as_one_pixel(hole_world):
    """The problem this task exists to solve, pinned so it cannot come back."""
    from src.render import scale
    naive = scale.display_radii_px(hole_world.radii, zoom=64.0,
                                   internal_height=360.0)
    assert naive[0] == pytest.approx(1.0)


def test_with_the_floor_fix_a_black_hole_draws_as_large_as_a_star(hole_world):
    from src.render import scale
    fixed = scale.display_radii_px(hole_world.radii, zoom=64.0,
                                   internal_height=360.0,
                                   floor_radii=scale.visual_radii(hole_world))
    assert fixed[0] == pytest.approx(scale.FLOOR_MAX)


def test_real_size_mode_collapses_a_black_hole_to_a_dot(hole_world):
    """Pressing R must still show the honest truth: the horizon is nothing."""
    from src.render import scale
    real = scale.display_radii_px(hole_world.radii, zoom=64.0,
                                  internal_height=360.0, real_size=True,
                                  floor_radii=scale.visual_radii(hole_world))
    assert real[0] == pytest.approx(1.0)


def test_the_floor_argument_leaves_ordinary_bodies_alone(hole_world):
    from src.render import scale
    plain = scale.display_radii_px(hole_world.radii, 64.0, 360.0)
    fixed = scale.display_radii_px(hole_world.radii, 64.0, 360.0,
                                   floor_radii=scale.visual_radii(hole_world))
    assert fixed[1] == pytest.approx(plain[1])


# --- aiming ---------------------------------------------------------------------

def test_without_the_aim_fix_a_black_hole_is_impossible_to_hit(hole_world):
    """8x the true horizon is 1.6e-6 AU: about 0.0001 pixels at system zoom."""
    from src.render import scale
    naive = scale.contact_radii_au(hole_world.radii, zoom=64.0,
                                   internal_height=360.0)
    assert naive[0] * 64.0 < 0.001          # sub-thousandth of a pixel


def test_with_the_aim_fix_a_black_hole_is_hittable(hole_world):
    from src.render import scale
    fixed = scale.contact_radii_au(hole_world.radii, zoom=64.0,
                                   internal_height=360.0,
                                   aim_radii=scale.visual_radii(hole_world))
    assert fixed[0] * 64.0 >= scale.FLOOR_MAX - 1e-9


def test_contact_radius_never_falls_below_the_true_radius(hole_world):
    from src.render import scale
    for zoom in (1e-3, 1.0, 64.0, 1e9):
        fixed = scale.contact_radii_au(hole_world.radii, zoom, 360.0,
                                       aim_radii=scale.visual_radii(hole_world))
        assert np.all(fixed >= hole_world.radii - 1e-30)


def test_the_aim_argument_leaves_ordinary_bodies_alone(hole_world):
    from src.render import scale
    plain = scale.contact_radii_au(hole_world.radii, 64.0, 360.0)
    fixed = scale.contact_radii_au(hole_world.radii, 64.0, 360.0,
                                   aim_radii=scale.visual_radii(hole_world))
    assert fixed[1] == pytest.approx(plain[1])
