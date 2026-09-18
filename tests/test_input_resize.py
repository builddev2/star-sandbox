"""A resize reaches the context and the camera through the event it actually
arrives as.  ``pygame.Window`` posts WINDOWSIZECHANGED; the old VIDEORESIZE
never fires for it, and a listener for the wrong one leaves the interface laid
out against a size the window no longer has."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest

from src.app.config import DisplayContext
from src.app.input import handle_event
from src.app.state import AppState
from src.render.camera import Camera
from src.render.draw import ViewSettings
from src.sim import Simulation


@pytest.fixture
def ctx():
    pygame.display.init()
    context = DisplayContext(size=(800, 600))
    yield context
    context.close()
    pygame.display.quit()


def _state(ctx):
    return AppState(Simulation(), Camera(ctx.width, ctx.height), ViewSettings())


def test_a_window_size_changed_event_resizes_the_context_and_the_camera(ctx):
    state = _state(ctx)
    event = pygame.event.Event(pygame.WINDOWSIZECHANGED, {"x": 640, "y": 480})
    handle_event(event, state, ctx)
    assert ctx.window == (640, 480)
    assert (state.camera.width, state.camera.height) == (640, 480)


def test_a_size_changed_event_is_consumed_before_the_interface(ctx):
    """Nothing else may see it: every control was laid out against the old size."""
    class Ui:
        def on_event(self, *_):
            raise AssertionError("the interface must not see a resize")
    event = pygame.event.Event(pygame.WINDOWSIZECHANGED, {"x": 700, "y": 500})
    handle_event(event, _state(ctx), ctx, Ui())
    assert ctx.window == (700, 500)


def test_the_old_videoresize_event_is_ignored(ctx):
    state = _state(ctx)
    event = pygame.event.Event(pygame.VIDEORESIZE, {"size": (640, 480), "w": 640, "h": 480})
    handle_event(event, state, ctx)
    assert ctx.window == (800, 600)
