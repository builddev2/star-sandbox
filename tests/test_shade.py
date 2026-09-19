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


def test_the_rim_pixel_takes_the_disc_edge_shading():
    """The antialiased rim pixel must not be black under its partial alpha:
    it should read close to the last fully-covered pixel's shading, not the
    near-zero limb value a naive clamp-to-edge geometry would give it."""
    px = 40
    surf = shade.sphere(px, 0.0, TONES, "cratered", seed=1)
    inner = _lum(_at(surf, px - 2, 0))
    rim = _lum(_at(surf, px, 0))
    assert rim >= inner * 0.5


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


# --- size-dependent ambient: a small dot has no room for a night side ------------

def test_a_tiny_dot_has_no_dark_side():
    """At px=5 there is no room for a terminator: the darkest pixel inside
    the disc must still read as a coloured dot, not a half-missing one.

    Shaded with the "nonsense" (flat-albedo) style rather than a textured
    one: this is a claim about the *lighting* model, and a textured style's
    own crater floors and rim highlights would swing brightest/darkest by
    more than the ambient term ever could, at any AMBIENT_SMALL -- confirmed
    by measurement, not assumed (see the task report)."""
    px = 5
    surf = shade.sphere(px, 0.0, TONES, "nonsense", seed=1)
    limit = 0.8 * px
    lums = [_lum(_at(surf, dx, dy))
            for dx in range(-px, px + 1) for dy in range(-px, px + 1)
            if dx * dx + dy * dy < limit * limit]
    darkest, brightest = min(lums), max(lums)
    assert darkest >= 0.30 * brightest


def test_a_big_body_still_keeps_a_real_night_side():
    """At px=40 the size-dependent ambient must have ramped back down to the
    ordinary AMBIENT: the dark side stays dark.  Same flat style as the tiny
    dot above, so the two tests measure the same thing at two sizes."""
    px = 40
    surf = shade.sphere(px, 0.0, TONES, "nonsense", seed=1)
    limit = 0.9 * px
    lums = [_lum(_at(surf, dx, dy))
            for dx in range(-px, px + 1) for dy in range(-px, px + 1)
            if dx * dx + dy * dy < limit * limit]
    darkest, brightest = min(lums), max(lums)
    assert darkest < 0.20 * brightest


# --- Jupiter's bands and Earth's oceans -------------------------------------------

def test_jupiter_bands_are_clearly_visible():
    """Measures the *albedo* directly, not a lit sphere: sampling a lit disc
    conflates band contrast with limb darkening, and at px=200 that alone
    cleared 1.35 (ratio 6.33) even on the pre-tuning code -- so the old
    lit-sphere version of this test guarded nothing.  Measured on the albedo
    alone: the pre-tuning bands (old tones, half the frequency, no contrast
    stretch, the 60/40 storm blend) give 1.259; the tuned code gives 2.245.
    1.8 is a floor the untuned code fails and the tuned code clears with 25%
    room -- see the task-12 fix report for both measurements."""
    from src.render.sprites import LOOKS
    tones, style, _ = LOOKS["jupiter"]
    px = 200
    r, u, v, z, alpha = shade._geometry(px, 0)
    albedo = shade._albedo(u, v, z, style, tones, seed=3)
    half = px
    x = int(-0.3 * px)
    lums = [_lum(albedo[half + y, half + x])
            for y in range(int(-0.8 * px), int(0.8 * px) + 1, 4)]
    assert min(lums) > 0
    assert max(lums) / min(lums) >= 1.8


def test_earth_is_mostly_ocean():
    from src.render.sprites import LOOKS
    tones, style, _ = LOOKS["terrestrial"]
    ocean, land, _ = tones
    px = 120
    r, u, v, z, _ = shade._geometry(px, 0)
    albedo = shade._albedo(u, v, z, style, tones, seed=3)
    ocean_arr, land_arr = np.asarray(ocean, dtype=np.float64), np.asarray(land, dtype=np.float64)
    dist_ocean = np.linalg.norm(albedo - ocean_arr, axis=-1)
    dist_land = np.linalg.norm(albedo - land_arr, axis=-1)
    disc = r < 0.9 * px
    fraction = float(((dist_ocean < dist_land) & disc).sum()) / float(disc.sum())
    assert 0.55 <= fraction <= 0.80


# --- styles and sizes ---------------------------------------------------------------

@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("px", (3, 40, 600))
def test_every_style_renders_at_every_size(style, px):
    surf = shade.sphere(px, 1.2, TONES, style, seed=7)
    assert surf.get_size() == (2 * px + 1, 2 * px + 1)
    assert _at(surf, 0, 0)[3] == 255


def test_a_style_has_texture():
    """The albedo itself -- before lighting is applied -- must vary across
    the disc for every real style, and be perfectly flat for an unknown
    one."""
    _, u, v, z, _ = shade._geometry(40, 0)
    for style in STYLES:
        albedo = shade._albedo(u, v, z, style, TONES, seed=7)
        assert albedo.std(axis=(0, 1)).max() > 5, style
    flat = shade._albedo(u, v, z, "nonsense", TONES, seed=7)
    assert flat.std(axis=(0, 1)).max() == 0


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


def test_the_atmosphere_rim_pixel_is_not_a_dark_ring():
    """The pixel straight at the disc edge must not be a near-black ring
    sitting between the lit disc and the halo."""
    px = 40
    bare = shade.sphere(px, 0.0, TONES, "cratered", seed=1)
    hazy = shade.sphere(px, 0.0, TONES, "cratered", seed=1, atmosphere=(120, 170, 255))
    rim = _at(hazy, px, 0)
    assert _lum(rim) > 20
    assert rim[2] > _at(bare, px, 0)[2]


# --- stars -----------------------------------------------------------------------

def test_a_star_is_opaque_everywhere_because_it_is_blitted_additively():
    surf = shade.star(20, (255, 244, 220))
    assert surf.get_size() == (shade.star_size(20), shade.star_size(20))
    corner = surf.get_at((0, 0))
    assert corner[3] == 255
    assert corner[:3] == (0, 0, 0), "outside the glow the colour is black, so adding it changes nothing"


def test_the_centre_is_brighter_than_the_limb_and_the_limb_brighter_than_the_corona():
    px = 30
    surf = shade.star(px, (255, 200, 120))
    centre = _lum(_at(surf, 0, 0))
    limb = _lum(_at(surf, px - 2, 0))
    corona = _lum(_at(surf, int(px * 1.5), 0))
    edge = _lum(_at(surf, int(px * shade.GLOW_SCALE), 0))
    assert centre > limb > corona > 0
    assert edge < corona


def test_a_cool_star_is_red_and_a_hot_one_is_blue():
    from src.render.palette import blackbody_rgb
    cool = shade.star(12, blackbody_rgb(3000.0))
    hot = shade.star(12, blackbody_rgb(20000.0))
    c = _at(cool, 9, 0)
    h = _at(hot, 9, 0)
    assert c[0] > c[2]
    assert h[2] > h[0]


def test_a_big_star_is_shaded_at_a_rung_and_scaled():
    surf = shade.star(600, (255, 244, 220))
    assert surf.get_size() == (shade.star_size(600), shade.star_size(600))
    assert _lum(_at(surf, 0, 0)) > 200


def test_the_limb_is_darker_than_the_disc_outside_the_core():
    """Sampled outside the white-hot core (r > 0.72 px), so the core cannot
    supply the contrast: this is limb darkening or nothing."""
    px = 60
    surf = shade.star(px, (255, 200, 120))
    assert _lum(_at(surf, int(0.80 * px), 0)) > _lum(_at(surf, int(0.97 * px), 0)) * 1.15


def test_granulation_textures_the_disc_and_follows_the_seed():
    px = 60
    a = shade.star(px, (200, 180, 120), seed=1)
    b = shade.star(px, (200, 180, 120), seed=2)
    ring = [_lum(_at(a, int(0.85 * px * math.cos(t)), int(0.85 * px * math.sin(t))))
            for t in np.linspace(0, 2 * math.pi, 48, endpoint=False)]
    assert np.std(ring) > 0.8, "a flat disc has no granulation"
    assert pygame.image.tobytes(a, "RGBA") != pygame.image.tobytes(b, "RGBA")


def test_streamers_make_the_corona_uneven_round_the_star():
    px = 40
    surf = shade.star(px, (255, 244, 220), seed=3)
    ring = [_lum(_at(surf, int(1.5 * px * math.cos(t)), int(1.5 * px * math.sin(t))))
            for t in np.linspace(0, 2 * math.pi, 48, endpoint=False)]
    assert max(ring) > min(ring) * 1.08


def test_the_corona_is_dimmer_than_the_limb_and_fades_to_nothing():
    px = 40
    surf = shade.star(px, (255, 244, 220), seed=3)
    angles = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    worst = max(_lum(_at(surf, int((px + 2) * math.cos(t)), int((px + 2) * math.sin(t))))
                / max(1.0, _lum(_at(surf, int((px - 1) * math.cos(t)), int((px - 1) * math.sin(t)))))
                for t in angles)
    assert worst < 1.0, "somewhere the corona outshines the limb it hugs"
    profile = [_lum(_at(surf, r, 0)) for r in range(px + 1, px + 8)]
    assert all(a > b for a, b in zip(profile, profile[1:])), "the corona must fall off, not sit on a shelf: %s" % profile
    edge = int(shade.GLOW_SCALE * px) - 1
    assert _lum(_at(surf, edge, 0)) < 8, "the glow must taper out, not stop at a circle"
