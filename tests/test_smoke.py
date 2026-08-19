"""Boot the real application headless and run frames through it.

Unit tests cover each piece; this is the one that catches the wiring -- a bad
argument order in main.py, a tool that crashes on its first overlay, a HUD that
divides by zero when the last body is destroyed.
"""

import os

import numpy as np
import pygame
import pytest

os.environ["SDL_VIDEODRIVER"] = "dummy"


@pytest.fixture
def app():
    from src.app.main import App
    instance = App(fullscreen=False)
    yield instance
    pygame.quit()


def test_a_hundred_frames_run_without_raising(app):
    for _ in range(100):
        app.step_once()
    assert app.state.sim.world.time > 0.0


def test_frames_run_with_every_tool_active(app):
    from src.app.tools import EditTool, LaunchTool, SelectTool
    for tool in (SelectTool, LaunchTool, EditTool):
        app.state.set_tool(tool)
        app.state.select(app.state.sim.world.bodies[3].id)
        for _ in range(10):
            app.step_once()


def test_frames_run_with_the_help_overlay_open(app):
    app.state.help_visible = True
    for _ in range(5):
        app.step_once()


def test_frames_run_in_real_size_mode(app):
    app.state.view.real_size = True
    for _ in range(5):
        app.step_once()


def test_frames_run_with_trails_and_labels_off(app):
    app.state.view.show_trails = False
    app.state.view.show_labels = False
    for _ in range(5):
        app.step_once()


def test_frames_run_after_everything_is_destroyed(app):
    app.state.sim.world.clear()
    for _ in range(5):
        app.step_once()


def test_frames_run_at_every_time_scale(app):
    from src.app.state import TIME_SCALES
    for index in range(len(TIME_SCALES)):
        app.state.time_index = index
        for _ in range(4):
            app.step_once()
    assert app.state.sim.world.time > 0


def test_frames_run_zoomed_all_the_way_in_and_out(app):
    for steps in (-200, 400):
        app.state.camera.zoom_by(steps)
        for _ in range(4):
            app.step_once()


def test_a_smashed_solar_system_keeps_rendering(app):
    """Throw a fast heavy body through the inner system and keep drawing."""
    from src import materials, units
    world = app.state.sim.world
    world.add("Bullet", materials.get("iron"), units.EARTH_MASS * 30,
              units.EARTH_RADIUS * 3,
              np.array([-1.4, 0.0]), np.array([0.35, 0.0]))
    app.state.sim.refresh()
    app.state.time_index = 3
    for _ in range(200):
        app.step_once()
    assert world.n > 0


def test_the_grid_can_be_changed_live(app):
    from src.app.config import GRID_CHOICES
    for grid in GRID_CHOICES:
        app.ctx.set_grid(grid)
        app.state.on_resize(app.ctx)
        app._build_render_stack()
        for _ in range(3):
            app.step_once()
        assert app.camera.width == app.ctx.width
