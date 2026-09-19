import math
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    pygame.init()
    pygame.display.set_mode((320, 200))
    yield
    pygame.quit()


@pytest.fixture
def cache():
    from src.render.sprites import SpriteCache
    return SpriteCache(limit=32)


def test_sprite_is_square_and_sized_to_the_radius(cache):
    from src import materials
    surf = cache.get(materials.get("rock"), seed=1, px=10, lit_angle=0.0, is_star=False)
    assert surf.get_width() == surf.get_height()
    assert surf.get_width() == 21          # 2*px + 1


def test_sprite_silhouette_is_circular(cache):
    """Detail drawing must never spill outside the disc."""
    from src import materials
    px = 12
    surf = cache.get(materials.get("gas"), seed=3, px=px, lit_angle=0.0, is_star=False)
    centre = px
    for x in range(surf.get_width()):
        for y in range(surf.get_height()):
            alpha = surf.get_at((x, y))[3]
            distance = math.hypot(x - centre, y - centre)
            if distance > px + 1.5:
                assert alpha == 0, "pixel at %d,%d outside the disc" % (x, y)


def test_sprite_centre_is_opaque(cache):
    from src import materials
    surf = cache.get(materials.get("ice"), seed=2, px=8, lit_angle=1.0, is_star=False)
    assert surf.get_at((8, 8))[3] == 255


def test_identical_requests_return_the_same_object(cache):
    from src import materials
    rock = materials.get("rock")
    a = cache.get(rock, seed=5, px=9, lit_angle=0.3, is_star=False)
    b = cache.get(rock, seed=5, px=9, lit_angle=0.3, is_star=False)
    assert a is b


def test_lit_angle_is_bucketed_so_the_cache_survives_camera_motion(cache):
    """A continuous lighting angle would miss the cache every single frame."""
    from src import materials
    from src.render.sprites import LIT_BUCKETS
    rock = materials.get("rock")
    step = 2 * math.pi / LIT_BUCKETS
    a = cache.get(rock, seed=5, px=9, lit_angle=0.0, is_star=False)
    b = cache.get(rock, seed=5, px=9, lit_angle=step * 0.2, is_star=False)
    c = cache.get(rock, seed=5, px=9, lit_angle=step * 1.5, is_star=False)
    assert a is b
    assert a is not c


def test_different_materials_look_different(cache):
    from src import materials
    a = cache.get(materials.get("iron"), seed=1, px=14, lit_angle=0.0, is_star=False)
    b = cache.get(materials.get("water"), seed=1, px=14, lit_angle=0.0, is_star=False)
    assert pygame.image.tostring(a, "RGBA") != pygame.image.tostring(b, "RGBA")


def test_lighting_makes_one_side_brighter(cache):
    from src import materials
    px = 30
    surf = cache.get(materials.get("rock"), seed=1, px=px, lit_angle=0.0, is_star=False)
    lit = surf.get_at((px + 18, px))
    dark = surf.get_at((px - 18, px))
    assert sum(lit[:3]) > sum(dark[:3]) * 1.5


def test_stars_are_larger_than_their_disc_because_of_the_glow(cache):
    from src import materials
    hydrogen = materials.get("hydrogen")
    plain = cache.get(hydrogen, seed=1, px=10, lit_angle=0.0, is_star=False)
    star = cache.get(hydrogen, seed=1, px=10, lit_angle=0.0, is_star=True)
    assert star.get_width() > plain.get_width()


def test_cache_evicts_and_stays_bounded(cache):
    from src import materials
    rock = materials.get("rock")
    for px in range(1, 60):
        cache.get(rock, seed=1, px=px, lit_angle=0.0, is_star=False)
    assert len(cache) <= 32


def test_tiny_bodies_still_produce_a_surface(cache):
    from src import materials
    for px in (0, 1, 2):
        surf = cache.get(materials.get("rock"), seed=1, px=px,
                         lit_angle=0.0, is_star=False)
        assert surf.get_width() == max(1, px) * 2 + 1
        assert surf.get_at((max(1, px), max(1, px)))[3] == 255


def test_every_material_renders_at_every_useful_size(cache):
    """Each style has its own detail routine; none may crash at any radius."""
    from src import materials
    for material in materials.MATERIALS:
        for px in (1, 3, 4, 12, 60):
            surf = cache.get(material, seed=7, px=px, lit_angle=1.2, is_star=False)
            assert surf.get_width() == px * 2 + 1


def test_rings_come_in_two_halves_that_bracket_the_planet():
    from src import materials
    from src.render.sprites import ring_size, ring_surfaces
    px = 20
    back, front = ring_surfaces(px, materials.get("gas").tones, seed=1, lit_angle=0.0)
    assert back.get_size() == front.get_size() == (ring_size(px), ring_size(px))
    assert back.get_width() > 2 * px + 1
    c = back.get_width() // 2
    # The back half is above the centre line and the front half below it.
    assert any(back.get_at((x, c - 4))[3] > 0 for x in range(back.get_width()))
    assert all(back.get_at((x, c + 4))[3] == 0 for x in range(back.get_width()))
    assert any(front.get_at((x, c + 4))[3] > 0 for x in range(front.get_width()))
    assert all(front.get_at((x, c - 4))[3] == 0 for x in range(front.get_width()))


def test_the_ring_has_a_gap():
    from src import materials
    from src.render.sprites import RING_GAP, ring_surfaces
    px = 40
    back, front = ring_surfaces(px, materials.get("gas").tones, seed=1, lit_angle=0.0)
    c = front.get_width() // 2
    # Along the horizontal axis the ring is unsquashed: the gap sits at RING_GAP radii.
    gap = front.get_at((c + int(px * RING_GAP), c + 1))[3]
    band = front.get_at((c + int(px * (RING_GAP + 0.15)), c + 1))[3]
    assert gap < band


def test_the_planet_shadows_the_far_half_of_the_ring():
    """With the light from +x, the anti-star direction is -x: the back half
    is darker there than where it is lit."""
    from src import materials
    from src.render.sprites import ring_surfaces
    px = 40
    back, _ = ring_surfaces(px, materials.get("gas").tones, seed=1, lit_angle=0.0)
    c = back.get_width() // 2
    y = c - int(px * 0.3 * 1.7)
    shadowed = sum(back.get_at((c - int(px * 0.3), y))[:3])
    lit = sum(back.get_at((c + int(px * 1.7), c - 2))[:3])
    assert shadowed < lit


def test_rings_are_cached(cache):
    from src import materials
    a = cache.get_rings(12, materials.get("gas").tones, seed=1, lit_angle=0.2)
    b = cache.get_rings(12, materials.get("gas").tones, seed=1, lit_angle=0.2)
    assert a is b


def test_rings_on_one_rung_share_the_shaded_base(cache):
    from src import materials
    from src.render import shade
    tones = materials.get("gas").tones
    cache.get_rings(300, tones, seed=1, lit_angle=0.0)
    assert len(cache._bases) == 1
    cache.get_rings(int(300 * 1.18), tones, seed=1, lit_angle=0.0)
    assert shade.shade_radius(300) == shade.shade_radius(int(300 * 1.18))
    assert len(cache._bases) == 1


def test_scaled_rings_are_the_size_asked_for(cache):
    from src import materials
    from src.render.sprites import ring_size
    back, front = cache.get_rings(300, materials.get("gas").tones, seed=1, lit_angle=0.0)
    assert back.get_size() == front.get_size() == (ring_size(300), ring_size(300))


# --- thermal glow -------------------------------------------------------------

def test_blackbody_colour_runs_red_through_white_up_the_planck_locus():
    from src.render import palette
    ember = palette.blackbody_rgb(1000.0)
    sunlike = palette.blackbody_rgb(5772.0)
    assert ember[0] > ember[1] > ember[2]          # red, almost no blue
    assert min(sunlike) > 200                       # warm white
    assert palette.blackbody_rgb(1500.0)[2] < palette.blackbody_rgb(3500.0)[2]


def test_the_sun_comes_out_warm_white_not_yellow():
    """5772 K on the Planck locus is (255, 242, 230) -- the actual colour of
    sunlight above the atmosphere."""
    from src.render import palette
    assert palette.blackbody_rgb(5772.0) == pytest.approx((255, 242, 230), abs=6)


def test_a_cold_body_has_no_glow_at_all():
    from src.render import sprites
    assert sprites.glow_fraction(300.0) == 0.0
    assert sprites.glow_key(300.0) == 0


def test_glow_climbs_from_dull_red_to_fully_blackbody():
    from src.render import sprites
    assert sprites.glow_fraction(sprites.GLOW_MIN_K) == 0.0
    assert 0.0 < sprites.glow_fraction(1800.0) < 1.0
    assert sprites.glow_fraction(sprites.GLOW_FULL_K) == pytest.approx(1.0)
    assert sprites.glow_fraction(20000.0) == pytest.approx(1.0)


def test_a_glowing_body_looks_different_from_a_cold_one(cache):
    from src import materials
    rock = materials.get("rock")
    cold = cache.get(rock, seed=1, px=14, lit_angle=0.0, is_star=False)
    hot = cache.get(rock, seed=1, px=14, lit_angle=0.0, is_star=False,
                    temperature=2500.0)
    assert pygame.image.tostring(cold, "RGBA") != pygame.image.tostring(hot, "RGBA")


def test_temperature_is_bucketed_so_a_cooling_body_does_not_rerender_every_frame(cache):
    """A body cooling by a fraction of a kelvin a frame would miss the cache on
    every one of them -- exactly what the cache exists to prevent."""
    from src import materials
    rock = materials.get("rock")
    a = cache.get(rock, seed=1, px=9, lit_angle=0.0, is_star=False, temperature=2000.0)
    b = cache.get(rock, seed=1, px=9, lit_angle=0.0, is_star=False, temperature=2003.0)
    assert a is b


def test_a_cold_body_renders_exactly_as_it_did_before_temperature_existed(cache):
    """The default has to be a no-op, or every ordinary world changes colour."""
    from src import materials, units
    rock = materials.get("rock")
    a = cache.get(rock, seed=4, px=12, lit_angle=0.0, is_star=False)
    b = cache.get(rock, seed=4, px=12, lit_angle=0.0, is_star=False,
                  temperature=units.CMB_TEMPERATURE_K)
    assert a is b


def test_a_craft_sprite_is_drawn_and_is_not_blank():
    from src.render.sprites import SpriteCache
    cache = SpriteCache()
    sprite = cache.get_craft(14, 0.0)
    assert sprite.get_width() > 8
    assert sprite.get_bounding_rect().width > 2


def test_craft_sprites_are_cached_within_a_heading_bucket():
    import math
    from src.render.sprites import SpriteCache
    cache = SpriteCache()
    first = cache.get_craft(14, 0.0)
    again = cache.get_craft(14, 0.001)      # same bucket
    assert again is first
    turned = cache.get_craft(14, math.pi)   # opposite bucket
    assert turned is not first


def test_a_tiny_craft_still_produces_a_visible_sprite():
    from src.render.sprites import SpriteCache
    cache = SpriteCache()
    sprite = cache.get_craft(2, 0.0)
    assert sprite.get_bounding_rect().width >= 1


def test_a_craft_sprite_is_a_shuttle_and_not_a_plain_dot():
    """The silhouette used to be untested: forcing the plain-dot branch --
    never drawing a single polygon -- passed the whole suite, because the
    other tests assert only that the surface is bigger than a few pixels,
    which a circle satisfies.  The thermal tile is the giveaway: the dot
    branch draws hull colour and nothing else."""
    from src.render.sprites import SpriteCache, _CRAFT_HULL, _CRAFT_TILE
    sprite = SpriteCache().get_craft(20, 0.0)
    seen = set()
    for x in range(sprite.get_width()):
        for y in range(sprite.get_height()):
            r, g, b, a = sprite.get_at((x, y))
            if a:
                seen.add((r, g, b))
    assert _CRAFT_HULL in seen, "no hull colour: nothing was drawn"
    assert _CRAFT_TILE in seen, "no thermal tile: this is a dot, not a shuttle"


def test_a_craft_sprite_really_is_rotated_to_its_heading():
    """Never rotating used to pass, because the cache test only compared
    object identity -- and the cache *key* differs by heading regardless of
    what the pixels do."""
    import math
    import pygame
    from src.render.sprites import SpriteCache
    cache = SpriteCache()
    nose_right = cache.get_craft(20, 0.0)
    nose_up = cache.get_craft(20, math.pi / 2.0)
    assert (pygame.image.tostring(nose_right, "RGBA")
            != pygame.image.tostring(nose_up, "RGBA"))


def test_the_heading_bucket_is_rounded_not_truncated():
    """int(x) instead of int(round(x)) shifts every craft half a bucket."""
    import math
    from src.render.sprites import SpriteCache, LIT_BUCKETS
    cache = SpriteCache()
    step = 2.0 * math.pi / LIT_BUCKETS
    # Just under half a step rounds back to bucket 0; just over rounds to 1.
    assert cache.get_craft(14, step * 0.49) is cache.get_craft(14, 0.0)
    assert cache.get_craft(14, step * 0.51) is not cache.get_craft(14, 0.0)


# --- looks -----------------------------------------------------------------------

def test_a_look_changes_the_picture_and_the_cache_key(cache):
    from src import materials
    rock = materials.get("rock")
    plain = cache.get(rock, seed=1, px=20, lit_angle=0.0, is_star=False)
    earth = cache.get(rock, seed=1, px=20, lit_angle=0.0, is_star=False, look="terrestrial")
    assert earth is not plain
    assert pygame.image.tobytes(earth, "RGBA") != pygame.image.tobytes(plain, "RGBA")
    # Earth's disc is wider than a bare rock's: the atmosphere pads it.
    assert earth.get_width() > plain.get_width()


def test_an_unknown_look_draws_from_the_material(cache):
    from src import materials
    rock = materials.get("rock")
    a = cache.get(rock, seed=1, px=20, lit_angle=0.0, is_star=False, look="nonsense")
    b = cache.get(rock, seed=1, px=20, lit_angle=0.0, is_star=False)
    assert pygame.image.tobytes(a, "RGBA") == pygame.image.tobytes(b, "RGBA")


def test_every_look_renders():
    from src import materials
    from src.render.sprites import LOOKS, SpriteCache
    cache = SpriteCache()
    for look in LOOKS:
        surf = cache.get(materials.get("rock"), seed=3, px=16, lit_angle=0.5,
                         is_star=False, look=look)
        assert surf.get_at((surf.get_width() // 2, surf.get_height() // 2))[3] == 255


def test_earth_is_blue(cache):
    from src import materials
    px = 40
    surf = cache.get(materials.get("rock"), seed=3, px=px, lit_angle=0.0,
                     is_star=False, look="terrestrial")
    c = surf.get_width() // 2
    blues = sum(1 for dx in range(-30, 31, 4) for dy in range(-30, 31, 4)
                if (lambda p: p[2] > p[0] + 20)(surf.get_at((c + dx, c + dy))))
    assert blues > 60, "most of a shaded Earth should be ocean"


# --- the ladder ------------------------------------------------------------------

def test_a_big_sprite_is_the_size_it_was_asked_for(cache):
    from src import materials
    surf = cache.get(materials.get("gas"), seed=1, px=500, lit_angle=0.0, is_star=False)
    assert surf.get_size() == (1001, 1001)


def test_two_sizes_on_one_rung_share_the_shaded_base(cache):
    """A zoom notch must not re-shade a 400 px planet."""
    from src import materials
    from src.render import shade
    gas = materials.get("gas")
    cache.get(gas, seed=1, px=300, lit_angle=0.0, is_star=False)
    bases = len(cache._bases)
    assert bases == 1, "the rung's base must be kept"
    cache.get(gas, seed=1, px=int(300 * 1.18), lit_angle=0.0, is_star=False)
    assert shade.shade_radius(300) == shade.shade_radius(int(300 * 1.18))
    assert len(cache._bases) == bases


def test_a_small_sprite_never_enters_the_ladder_table(cache):
    from src import materials
    cache.get(materials.get("rock"), seed=1, px=12, lit_angle=0.0, is_star=False)
    assert len(cache._bases) == 0


# --- supersampling ----------------------------------------------------------------

def test_the_black_hole_is_supersampled_and_lands_at_its_own_size(cache):
    from src.render import sprites
    px = 12
    big = sprites._make_black_hole(px * sprites.SUPERSAMPLE, 0.5, 0.8, 1,
                                   ss=sprites.SUPERSAMPLE)
    small = cache.get_black_hole(seed=1, px=px, spin=0.5, disk=0.8)
    assert small.get_width() < big.get_width()
    assert small.get_width() % 2 == 1, "an odd side keeps the centre on a pixel"
    c = small.get_width() // 2
    assert small.get_at((c, c))[:3] == (0, 0, 0)
    assert abs(small.get_width()
               - sprites._make_black_hole(px, 0.5, 0.8, 1).get_width()) <= 2


def test_a_tiny_hole_s_rim_scales_with_the_supersample(cache):
    """The non-detailed branch's filled-disc rim is a one-pixel stroke that
    must scale by ss like every other stroke width -- at px+1 from centre,
    on at least one axis pixel, the rim is still full brightness."""
    for px in (2, 3, 4):
        surf = cache.get_black_hole(seed=1, px=px, spin=0.0, disk=0.0)
        c = surf.get_width() // 2
        r = px + 1
        sums = [sum(surf.get_at((c + dx, c + dy))[:3])
                for dx, dy in ((r, 0), (-r, 0), (0, r), (0, -r))]
        assert max(sums) > 600, (px, sums)


def test_a_supersampled_rim_is_antialiased(cache):
    """Somewhere on the photon ring there is a pixel that is neither black nor
    the full ring colour -- the whole point of shading at 2x."""
    px = 14
    surf = cache.get_black_hole(seed=1, px=px, spin=0.0, disk=0.0)
    c = surf.get_width() // 2
    shades = {sum(surf.get_at((c + dx, c + dy))[:3]) for dx in range(-px - 2, px + 3)
              for dy in range(-px - 2, px + 3)}
    partial = [s for s in shades if 40 < s < 700]
    assert partial, "every rim pixel is either black or full: no antialiasing happened"


def test_the_craft_is_supersampled_but_keeps_its_hull_colours(cache):
    from src.render.sprites import _CRAFT_HULL, _CRAFT_TILE
    sprite = cache.get_craft(14, 0.0)
    seen = {sprite.get_at((x, y))[:3] for x in range(sprite.get_width())
            for y in range(sprite.get_height()) if sprite.get_at((x, y))[3] > 200}
    assert _CRAFT_HULL in seen
    assert _CRAFT_TILE in seen
