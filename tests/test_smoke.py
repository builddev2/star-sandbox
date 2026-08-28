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


# --- the interface, driven the way a mouse drives it ------------------------------

def _press(pos):
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1)


def _release(pos):
    return pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=1)


def test_the_buttons_are_laid_out_before_the_first_event(app):
    """A click in the opening frame has to land on something."""
    assert app.ui.controls
    assert app.ui.control_at(next(c for c in app.ui.controls
                                  if c.id == "time.pause").rect.center) is not None


def test_pressing_every_button_in_the_real_app_keeps_it_running(app):
    """Push each control through the whole frame loop, not just its action.

    This is the test that catches a button whose action leaves the app in a
    state the *renderer* cannot draw -- deleting the followed body, editing a
    world down to nothing -- which no unit test of the action alone would see.
    """
    from src.app.tools import EditTool, LaunchTool, SelectTool

    for tool in (SelectTool, LaunchTool, EditTool):
        app.state.set_tool(tool)
        if app.state.sim.world.n > 4:
            app.state.select(app.state.sim.world.bodies[4].id)
        app.step_once()

        for control in list(app.ui.controls):
            # Fullscreen has no meaning headless, and quit is what its own test
            # is for -- pressing it here would end the app mid-sweep.
            if not control.clickable or control.id in ("view.fullscreen", "app.quit"):
                continue
            pos = control.rect.center
            pygame.event.post(_press(pos))
            pygame.event.post(_release(pos))
            app.step_once()
            app.step_once()

    assert app.state.running


def test_a_click_on_empty_space_still_selects_a_body(app):
    """The interface must not have swallowed the world's own clicks."""
    for _ in range(3):
        app.step_once()
    world = app.state.sim.world
    index = next(i for i in range(world.n) if world.bodies[i].name == "Sun")
    sx, sy = app.state.camera.world_to_screen(world.pos[index])
    assert not app.ui.blocks((int(sx), int(sy)))

    pygame.event.post(_press((int(sx), int(sy))))
    pygame.event.post(_release((int(sx), int(sy))))
    app.step_once()
    assert app.state.selected_id == world.bodies[index].id


def test_the_quit_button_ends_the_program_on_the_second_press(app):
    """Driven through the real loop, because arming it changes the layout."""
    app.step_once()

    def quit_button():
        return next(c for c in app.ui.controls if c.id == "app.quit")

    pygame.event.post(_press(quit_button().rect.center))
    pygame.event.post(_release(quit_button().rect.center))
    app.step_once()
    assert app.state.quit_armed
    assert app.state.running

    # The button has grown into "Really quit?"; press wherever it is now.
    pygame.event.post(_press(quit_button().rect.center))
    pygame.event.post(_release(quit_button().rect.center))
    app.step_once()
    assert not app.state.running


def test_holding_the_mass_button_ignites_a_planet(app):
    """The whole point of hold-to-repeat, end to end through the real loop."""
    from src.app.tools import EditTool
    from src.body import KIND_STAR

    jupiter = next(b for b in app.state.sim.world.bodies if b.name == "Jupiter")
    app.state.select(jupiter.id)
    app.state.set_tool(EditTool)
    app.step_once()

    plus = next(c for c in app.ui.controls
                if c.id == "edit.step" and c.payload == ("mass", 3))
    pygame.event.post(_press(plus.rect.center))
    app.step_once()

    # Hold it for a simulated couple of seconds' worth of repeats -- long
    # enough to ignite (around k=20) but short of edits.MAX_MASS, which now
    # collapses the body into a black hole (around k=60) rather than capping
    # it as a star.
    now = pygame.time.get_ticks()
    for k in range(40):
        app.ui.update(app.state, app.ctx, now=now + 400 + k * 60)
        app.step_once()

    index = app.state.sim.world.index_of(jupiter.id)
    assert app.state.sim.world.bodies[index].kind == KIND_STAR


def test_the_app_survives_a_real_collision(app):
    """Drive two bodies into each other through the real frame loop and
    confirm the app does not raise, and that the effects layer actually
    receives and draws something for it."""
    from src import materials, units
    world = app.state.sim.world
    world.clear()
    rock = materials.get("rock")
    r = units.EARTH_RADIUS
    world.add("A", rock, units.EARTH_MASS, r, np.array([0.0, 0.0]),
             np.array([0.0, 0.0]))
    world.add("B", rock, units.EARTH_MASS, r, np.array([0.5 * r, 0.0]),
             np.array([0.0, 0.0]))
    app.state.sim.refresh()
    app.state.time_index = app.state.time_index_max

    hit = False
    for _ in range(60):
        app.step_once()
        if app.state.sim.impacts or app.state.effects.count > 0:
            hit = True
            break
    assert hit, "the collision must actually be detected and reach the effects layer"
    # The outcome itself (merge, or a cascade of further fragment-on-fragment
    # hits within the same wide-timescale frame) is not the point here -- the
    # frame loop surviving it, end to end, is.
    assert world.n >= 1


# --- resizing the real window ------------------------------------------------------

def _resize(size):
    return pygame.event.Event(pygame.VIDEORESIZE, {"size": size, "w": size[0],
                                                   "h": size[1]})


def test_resizing_the_window_carries_through_the_whole_frame_loop(app):
    """A drag on the window edge has to reach the camera, the starfield and the
    interface, not just the surface -- each of them is sized in pixels and each
    is rebuilt from a different place."""
    from src.app.config import clamp_window
    app.step_once()
    smaller = clamp_window((app.ctx.width - 160, app.ctx.height - 120))

    pygame.event.post(_resize(smaller))
    app.step_once()

    assert (app.ctx.width, app.ctx.height) == smaller
    assert app.ctx.surface.get_size() == smaller
    assert (app.camera.width, app.camera.height) == smaller
    assert (app.stars.width, app.stars.height) == smaller
    assert app.state.running

    # And the interface was rebuilt against the new size, so its buttons are
    # somewhere a mouse can still reach them.
    screen = pygame.Rect(0, 0, *smaller)
    assert app.ui.controls
    assert all(screen.contains(c.rect) for c in app.ui.controls)


def test_the_window_cannot_be_dragged_smaller_than_the_interface(app):
    from src.app.config import MIN_WINDOW
    pygame.event.post(_resize((80, 60)))
    app.step_once()
    assert (app.ctx.width, app.ctx.height) == MIN_WINDOW
    app.step_once()
    assert app.state.running


def test_the_window_size_button_actually_resizes_the_window(app):
    """The bug this replaces: the button was there, and pressing it did nothing."""
    app.step_once()
    button = next(c for c in app.ui.controls if c.id == "view.window")
    before = app.ctx.window

    pygame.event.post(_press(button.rect.center))
    pygame.event.post(_release(button.rect.center))
    app.step_once()
    app.step_once()

    assert app.ctx.window != before
    assert (app.camera.width, app.camera.height) == (app.ctx.width, app.ctx.height)
    assert app.state.running


def test_a_resize_does_not_lose_the_simulation(app):
    for _ in range(5):
        app.step_once()
    before = app.state.sim.world.time
    names = [b.name for b in app.state.sim.world.bodies[:app.state.sim.world.n]]

    pygame.event.post(_resize((720, 540)))
    for _ in range(5):
        app.step_once()

    world = app.state.sim.world
    assert world.time > before
    assert [b.name for b in world.bodies[:world.n]] == names
