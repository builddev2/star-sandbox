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


def test_font_reports_a_sensible_height():
    from src.render.pixelfont import PixelFont
    font = PixelFont(14)
    assert 8 <= font.height <= 26


def test_width_grows_with_text_length():
    from src.render.pixelfont import PixelFont
    font = PixelFont(14)
    assert font.width("XXXXXXXXXX") > font.width("X")
    assert font.width("") == 0


def test_draw_returns_the_advance_and_marks_pixels():
    from src.render import palette
    from src.render.pixelfont import PixelFont
    font = PixelFont(14)
    surface = pygame.Surface((200, 40))
    surface.fill(palette.SPACE)
    advance = font.draw(surface, 2, 2, "Neptune", palette.TEXT)
    assert advance == font.width("Neptune")
    assert pygame.transform.average_color(surface)[:3] != palette.SPACE[:3]


def test_text_is_antialiased():
    """Glyph edges blend into the background: at least one drawn pixel is
    neither the background nor exactly the text colour."""
    from src.render.pixelfont import PixelFont
    font = PixelFont(14)
    surface = pygame.Surface((160, 34))
    surface.fill((0, 0, 0))
    font.draw(surface, 1, 1, "AVW", (255, 255, 255))
    colours = {surface.get_at((x, y))[:3]
               for x in range(160) for y in range(34)}
    assert colours - {(0, 0, 0), (255, 255, 255)}, sorted(colours)


def test_the_bundled_face_is_used_and_sized_in_points():
    from src.render.pixelfont import FONT_PATH, PixelFont
    assert FONT_PATH.exists()
    assert 13 <= PixelFont(11).height <= 16
    assert PixelFont(22).height >= 2 * PixelFont(11).height - 2


def test_a_missing_font_file_falls_back_to_the_built_in_face_at_the_same_height(tmp_path):
    from src.render.pixelfont import PixelFont
    real = PixelFont(11).height
    fallback = PixelFont(11, path=tmp_path / "nope.ttf").height
    assert abs(fallback - real) <= 2


def test_repeated_renders_are_cached():
    from src.render.pixelfont import PixelFont
    font = PixelFont(14)
    a = font.render("Jupiter", (255, 255, 255))
    b = font.render("Jupiter", (255, 255, 255))
    assert a is b
