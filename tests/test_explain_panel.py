"""The panel: it says the right words, and it stays inside the window."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest

from src import edits, explain, scenarios
from src.render import widgets
from src.render.explain import ExplainPanel
from src.render.pixelfont import PixelFont
from src.render.widgets import Metrics


@pytest.fixture(scope="module", autouse=True)
def display():
    pygame.display.init()
    pygame.display.set_mode((64, 64))
    yield
    pygame.display.quit()


@pytest.fixture
def report():
    world = scenarios.solar_system(moons=False)
    before = explain.snapshot(world)
    i = [b.name for b in world.bodies].index("Sun")
    edits.set_mass(world, i, 2.0 * float(world.mass[i]))
    return explain.diff(before, explain.snapshot(world),
                        world.bodies[i].id, "mass")


@pytest.fixture
def panel():
    return ExplainPanel(PixelFont(8))


def test_a_period_row_is_formatted_as_a_period_not_as_a_bare_float(panel, report):
    earth = next(c for c in report.changes if c.name == "Earth")
    lap = next(r for r in earth.rows if r.label == "One lap")
    was, now = panel.format_row(lap)
    assert "d" in was or "yr" in was
    assert "138" in now


def test_the_panel_stays_inside_the_window(panel, report):
    surface = pygame.Surface((480, 270))
    m = Metrics(panel.font, 480, 270)
    x = widgets.overlay_x(m, show_body_list=True)
    rect = panel.draw(surface, report, m, x, show_maths=True)
    assert rect.right <= 480
    assert rect.top >= m.content_top
    assert rect.bottom <= m.content_bottom


def test_the_panel_starts_clear_of_the_body_list_drawer(panel):
    m = Metrics(panel.font, 480, 270)
    opened = widgets.overlay_x(m, show_body_list=True)
    closed = widgets.overlay_x(m, show_body_list=False)
    assert opened > closed
    assert opened >= widgets.body_list_width(m)


def test_showing_the_maths_makes_the_panel_taller(panel, report):
    m = Metrics(panel.font, 800, 600)
    x = widgets.overlay_x(m, show_body_list=False)
    assert (panel.rect(report, m, x, show_maths=True).height
            > panel.rect(report, m, x, show_maths=False).height)


def test_the_fly_readout_and_the_explain_panel_read_one_helper():
    """FlyTool.readout_x used to implement the 'clear of the drawer' rule
    itself.  Two copies of that rule is one of them eventually being wrong --
    the same argument scale.contact_radii_au makes about drawing and aiming."""
    import inspect
    from src.app.tools import FlyTool
    source = inspect.getsource(FlyTool.readout_x)
    assert "overlay_x" in source
    assert "body_list_width" not in source
