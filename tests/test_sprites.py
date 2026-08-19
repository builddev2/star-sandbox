import math
import os

import pygame
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


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
    """Bodies shade away from the star, so the lit limb must be brighter."""
    from src import materials
    px = 16
    surf = cache.get(materials.get("rock"), seed=1, px=px, lit_angle=0.0, is_star=False)
    # lit_angle 0 means the light comes from +x, so the right limb is lit.
    right = sum(surf.get_at((px + int(px * 0.85), px))[:3])
    left = sum(surf.get_at((px - int(px * 0.85), px))[:3])
    assert right > left


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
        assert surf.get_width() >= 1


def test_every_material_renders_at_every_useful_size(cache):
    """Each style has its own detail routine; none may crash at any radius."""
    from src import materials
    for material in materials.MATERIALS:
        for px in (1, 3, 4, 12, 60):
            surf = cache.get(material, seed=7, px=px, lit_angle=1.2, is_star=False)
            assert surf.get_width() == px * 2 + 1


def test_ring_surface_is_wider_than_the_planet():
    from src import materials
    from src.render.sprites import ring_surface
    rings = ring_surface(10, materials.get("gas").tones)
    assert rings.get_width() > 21
