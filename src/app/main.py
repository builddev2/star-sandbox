"""Entry point and frame loop."""

import sys

import pygame

from ..render import palette
from ..render.camera import Camera
from ..render.draw import ViewSettings, WorldRenderer
from ..render.hud import Hud
from ..render.pixelfont import PixelFont
from ..render.sprites import SpriteCache
from ..render.starfield import Starfield
from ..sim import Simulation
from .config import DEFAULT_S, open_display
from .input import handle_event
from .state import AppState
from .tools import SelectTool


def probe(fullscreen=False):
    """Report what SDL actually gives us.  Answers the Retina question."""
    pygame.init()
    ctx = open_display(fullscreen=fullscreen)
    info = {
        "desktop_size": pygame.display.get_desktop_sizes(),
        "window_size": pygame.display.get_window_size(),
        "driver": pygame.display.get_driver(),
        "default_S": DEFAULT_S,
        "buffer": (ctx.width, ctx.height),
        "upscale": ctx.upscale,
        "font_px": ctx.font_px,
    }
    for key, value in info.items():
        print("%-16s %s" % (key, value))
    pygame.quit()
    return info


class App:
    def __init__(self, fullscreen=True):
        pygame.init()
        self.ctx = open_display(fullscreen=fullscreen)
        self.clock = pygame.time.Clock()
        self.cache = SpriteCache()
        self.camera = Camera(self.ctx.width, self.ctx.height)
        self._build_render_stack()
        self.state = AppState(Simulation(), self.camera, ViewSettings())
        self.state.set_tool(SelectTool)

    def _build_render_stack(self):
        """(Re)create everything that is sized in pixels.  Called on grid change."""
        self.font = PixelFont(self.ctx.font_px)
        self.camera.width, self.camera.height = self.ctx.width, self.ctx.height
        self.renderer = WorldRenderer(self.cache, self.font)
        self.hud = Hud(self.font)
        self.stars = Starfield(self.ctx.width, self.ctx.height)
        self.cache.clear()

    def step_once(self):
        """One frame: events, physics, drawing.

        Extracted from the loop so the smoke test can drive the real application
        without owning the loop.
        """
        grid = self.ctx.s
        for event in pygame.event.get():
            handle_event(event, self.state, self.ctx)
        if self.ctx.s != grid or self.ctx.width != self.camera.width:
            self._build_render_stack()

        state = self.state
        frame_dt = self.clock.tick(60) / 1000.0
        state.sim.advance(state.days_per_second * frame_dt)
        state.sim.update_trails()
        state.camera.update_follow(state.sim.world)

        surface = self.ctx.surface
        surface.fill(palette.SPACE)
        self.stars.draw(surface, state.camera)
        self.renderer.draw(surface, state.sim, state.camera, state.view,
                           selected_id=state.selected_id)
        state.tool.draw_overlay(surface, state, self.font)
        self.hud.draw(surface, state)
        pygame.display.flip()

    def run(self):
        while self.state.running:
            self.step_once()
        pygame.quit()
        return 0


def main():
    return App().run()


if __name__ == "__main__":
    sys.exit(main())
