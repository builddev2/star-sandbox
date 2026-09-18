"""The per-pixel sphere shader.  Every assertion is about a pixel, because the
whole point of the module is what the pixels look like."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import math

import numpy as np
import pygame
import pytest

from src.render import shade

TONES = ((0x4A, 0x33, 0x20), (0xB0, 0x80, 0x50), (0xE0, 0xB4, 0x84))
STYLES = ("cratered", "banded", "cracked", "turbulent", "terrestrial")


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    pygame.init()
    yield
    pygame.quit()


def _lum(colour):
    return 0.2126 * colour[0] + 0.7152 * colour[1] + 0.0722 * colour[2]


def _at(surface, dx, dy):
    """The pixel ``dx, dy`` from the centre of a sprite."""
    c = surface.get_width() // 2
    return surface.get_at((c + int(dx), c + int(dy)))


# --- noise ---------------------------------------------------------------------

def test_fbm_is_deterministic_and_in_range():
    xs, ys = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 1, 40))
    a = shade.fbm(xs, ys, seed=3)
    b = shade.fbm(xs, ys, seed=3)
    c = shade.fbm(xs, ys, seed=4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.min() >= 0.0 and a.max() <= 1.0
    assert a.std() > 0.02, "flat noise is no noise"


# --- the ladder ------------------------------------------------------------------

def test_small_sprites_are_shaded_at_their_own_size():
    for px in (1, 2, 7, shade.LADDER_START):
        assert shade.shade_radius(px) == px


def test_big_sprites_are_shaded_at_a_rung_and_never_above_the_cap():
    assert shade.shade_radius(shade.LADDER_START + 1) > shade.LADDER_START
    assert shade.shade_radius(700) == shade.SHADE_MAX_PX
    assert shade.shade_radius(1_000_000) == shade.SHADE_MAX_PX
    # A zoom notch is 18%; the ladder is 25%, so a notch re-uses a rung.
    assert shade.shade_radius(100) == shade.shade_radius(int(100 * 1.18))


# --- the rim ---------------------------------------------------------------------

def test_the_rim_is_antialiased():
    surf = shade.sphere(20, 0.0, TONES, "cratered", seed=1)
    assert surf.get_size() == (41, 41)
    assert _at(surf, 18, 0)[3] == 255
    assert 0 < _at(surf, 20, 0)[3] < 255
    assert 0 < _at(surf, 0, 20)[3] < 255
    # The corners, two pixels outside the disc diagonally, are nothing.
    assert surf.get_at((0, 0))[3] == 0
    assert surf.get_at((40, 40))[3] == 0


def test_a_one_pixel_body_is_still_a_sprite():
    for px in (1, 2, 3):
        surf = shade.sphere(px, 0.0, TONES, "cratered", seed=1)
        assert surf.get_size() == (2 * px + 1, 2 * px + 1)
        assert _at(surf, 0, 0)[3] == 255


# --- lighting --------------------------------------------------------------------

def test_the_lit_side_faces_the_light():
    surf = shade.sphere(30, 0.0, TONES, "cratered", seed=1)
    assert _lum(_at(surf, 18, 0)) > _lum(_at(surf, -18, 0)) * 1.5
    flipped = shade.sphere(30, math.pi, TONES, "cratered", seed=1)
    assert _lum(_at(flipped, -18, 0)) > _lum(_at(flipped, 18, 0)) * 1.5


def test_the_light_direction_is_screen_up_for_pi_over_two():
    """lit_angle is a world angle; +y in the world is *up* on screen."""
    surf = shade.sphere(30, math.pi / 2, TONES, "cratered", seed=1)
    assert _lum(_at(surf, 0, -18)) > _lum(_at(surf, 0, 18)) * 1.5


def test_the_night_side_is_never_black():
    surf = shade.sphere(30, 0.0, TONES, "cratered", seed=1)
    darkest = min(_lum(_at(surf, dx, dy))
                  for dx in range(-27, 28, 3) for dy in range(-27, 28, 3)
                  if dx * dx + dy * dy < 27 * 27)
    assert darkest > 0.0


def test_the_limb_is_darker_than_the_centre():
    surf = shade.sphere(40, 0.0, TONES, "banded", seed=2)
    # Straight above the centre, perpendicular to the light: same n.L on both,
    # so only limb darkening separates them.
    assert _lum(_at(surf, 0, 38)) < _lum(_at(surf, 0, 4))


# --- styles and sizes ---------------------------------------------------------------

@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("px", (3, 40, 600))
def test_every_style_renders_at_every_size(style, px):
    surf = shade.sphere(px, 1.2, TONES, style, seed=7)
    assert surf.get_size() == (2 * px + 1, 2 * px + 1)
    assert _at(surf, 0, 0)[3] == 255


def test_a_style_has_texture():
    """Two pixels at the same lighting must differ, or the albedo is flat."""
    surf = shade.sphere(60, 0.0, TONES, "cratered", seed=7)
    ring = [_at(surf, 0, dy)[:3] for dy in range(-40, 41, 5)]
    assert len(set(ring)) > 3


def test_the_same_seed_gives_the_same_picture_and_a_different_seed_does_not():
    a = shade.sphere(24, 0.3, TONES, "cracked", seed=5)
    b = shade.sphere(24, 0.3, TONES, "cracked", seed=5)
    c = shade.sphere(24, 0.3, TONES, "cracked", seed=6)
    assert pygame.image.tobytes(a, "RGBA") == pygame.image.tobytes(b, "RGBA")
    assert pygame.image.tobytes(a, "RGBA") != pygame.image.tobytes(c, "RGBA")


def test_an_unknown_style_is_a_plain_ball_not_an_error():
    surf = shade.sphere(10, 0.0, TONES, "nonsense", seed=1)
    assert _at(surf, 0, 0)[3] == 255


# --- atmosphere --------------------------------------------------------------------

def test_an_atmosphere_adds_a_halo_outside_the_rim():
    px = 30
    bare = shade.sphere(px, 0.0, TONES, "cratered", seed=1)
    hazy = shade.sphere(px, 0.0, TONES, "cratered", seed=1, atmosphere=(120, 170, 255))
    assert hazy.get_width() > bare.get_width()
    assert hazy.get_width() == shade.sphere_size(px, True)
    assert _at(hazy, px + 1, 0)[3] > 0
    assert _at(hazy, 0, 0)[3] == 255


def test_the_atmosphere_tints_the_limb():
    px = 40
    bare = shade.sphere(px, 0.0, TONES, "cratered", seed=1)
    hazy = shade.sphere(px, 0.0, TONES, "cratered", seed=1, atmosphere=(120, 170, 255))
    assert _at(hazy, 37, 0)[2] > _at(bare, 37, 0)[2] + 10, "the rim should go blue"
