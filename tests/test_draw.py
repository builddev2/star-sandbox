import math
import os

import numpy as np
import pygame
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def _ink(surface, background):
    """Count pixels that are not the background.

    average_color is useless here: a few hundred thin trail pixels across a
    640x400 surface move the mean by less than one unit and round away.
    """
    width, height = surface.get_size()
    return sum(1
               for x in range(width)
               for y in range(height)
               if surface.get_at((x, y))[:3] != background[:3])


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    pygame.init()
    pygame.display.set_mode((320, 200))
    yield
    pygame.quit()


@pytest.fixture
def parts():
    from src.render.camera import Camera
    from src.render.draw import ViewSettings, WorldRenderer
    from src.render.pixelfont import PixelFont
    from src.render.sprites import SpriteCache
    from src.sim import Simulation
    surface = pygame.Surface((640, 400))
    return {
        "surface": surface,
        "sim": Simulation(),
        "camera": Camera(640, 400),
        "renderer": WorldRenderer(SpriteCache(limit=256), PixelFont(14)),
        "view": ViewSettings(),
    }


def test_lighting_points_from_the_sun():
    """Bodies shade away from the star, so the angle must point sunward."""
    from src.render.draw import nearest_star_angle
    from src.sim import Simulation
    sim = Simulation()
    world = sim.world
    names = [b.name for b in world.bodies]
    sun, earth = names.index("Sun"), names.index("Earth")
    angle = nearest_star_angle(world, earth)
    rel = world.pos[sun] - world.pos[earth]
    assert angle == pytest.approx(math.atan2(rel[1], rel[0]))


def test_lighting_of_a_lone_body_does_not_crash():
    from src import materials, units
    from src.render.draw import nearest_star_angle
    from src.world import World
    world = World()
    world.add("lonely", materials.get("rock"), units.EARTH_MASS,
              units.EARTH_RADIUS, np.zeros(2), np.zeros(2))
    assert isinstance(nearest_star_angle(world, 0), float)


def test_drawing_the_solar_system_marks_the_surface(parts):
    from src.render import palette
    parts["surface"].fill(palette.SPACE)
    parts["renderer"].draw(parts["surface"], parts["sim"], parts["camera"],
                           parts["view"])
    assert pygame.transform.average_color(parts["surface"])[:3] != palette.SPACE[:3]


def test_drawing_an_empty_world_does_not_crash(parts):
    parts["sim"].world.clear()
    parts["renderer"].draw(parts["surface"], parts["sim"], parts["camera"],
                           parts["view"])


def test_bodies_far_off_screen_are_skipped(parts):
    """Culling is what keeps a 200-body world cheap when zoomed in."""
    parts["camera"].centre = np.array([1e6, 1e6])
    before = len(parts["renderer"].cache)
    parts["renderer"].draw(parts["surface"], parts["sim"], parts["camera"],
                           parts["view"])
    assert len(parts["renderer"].cache) == before, "off-screen bodies were rendered"


def test_trails_can_be_switched_off(parts):
    from src.render import palette
    for _ in range(30):
        parts["sim"].advance(5.0, budget_s=5.0)
        parts["sim"].update_trails()

    def ink_with(show):
        parts["view"].show_trails = show
        parts["surface"].fill(palette.SPACE)
        parts["renderer"].draw(parts["surface"], parts["sim"], parts["camera"],
                               parts["view"])
        return _ink(parts["surface"], palette.SPACE)

    assert ink_with(True) > ink_with(False)


def test_selection_ring_changes_the_picture(parts):
    from src.render import palette
    earth = next(b for b in parts["sim"].world.bodies if b.name == "Earth")
    parts["surface"].fill(palette.SPACE)
    parts["renderer"].draw(parts["surface"], parts["sim"], parts["camera"],
                           parts["view"])
    plain = pygame.image.tostring(parts["surface"], "RGB")

    parts["surface"].fill(palette.SPACE)
    parts["renderer"].draw(parts["surface"], parts["sim"], parts["camera"],
                           parts["view"], selected_id=earth.id)
    selected = pygame.image.tostring(parts["surface"], "RGB")

    assert plain != selected


def test_selecting_a_dead_body_id_is_harmless(parts):
    parts["renderer"].draw(parts["surface"], parts["sim"], parts["camera"],
                           parts["view"], selected_id=999999)


def test_real_size_mode_draws_far_less_ink(parts):
    """Collapsing everything to dots must visibly empty the screen."""
    from src.render import palette

    def ink():
        parts["surface"].fill(palette.SPACE)
        parts["renderer"].draw(parts["surface"], parts["sim"], parts["camera"],
                               parts["view"])
        return _ink(parts["surface"], palette.SPACE)

    parts["view"].show_trails = False
    parts["view"].show_labels = False
    full = ink()
    parts["view"].real_size = True
    dots = ink()
    assert dots < full


def test_starfield_draws_stars():
    from src.render import palette
    from src.render.camera import Camera
    from src.render.starfield import Starfield
    surface = pygame.Surface((320, 200))
    surface.fill(palette.SPACE)
    Starfield(320, 200, count=200).draw(surface, Camera(320, 200))
    # Every requested star must land on screen, give or take overlaps.
    assert _ink(surface, palette.SPACE) > 150


def test_starfield_wraps_seamlessly_when_panning():
    """The sky tiles modulo the screen, so panning any distance keeps the same
    number of stars visible rather than sliding them off the edge."""
    from src.render import palette
    from src.render.camera import Camera
    from src.render.starfield import Starfield
    field = Starfield(320, 200, count=200)
    counts = []
    cam = Camera(320, 200)
    for pan in (0, 500, 5000, 50000):
        cam.centre[0] = pan
        surface = pygame.Surface((320, 200))
        surface.fill(palette.SPACE)
        field.draw(surface, cam)
        counts.append(_ink(surface, palette.SPACE))
    assert min(counts) > 150, counts


def test_starfield_is_deterministic():
    from src.render.camera import Camera
    from src.render.starfield import Starfield
    a, b = pygame.Surface((320, 200)), pygame.Surface((320, 200))
    cam = Camera(320, 200)
    Starfield(320, 200, seed=42).draw(a, cam)
    Starfield(320, 200, seed=42).draw(b, cam)
    assert pygame.image.tostring(a, "RGB") == pygame.image.tostring(b, "RGB")
