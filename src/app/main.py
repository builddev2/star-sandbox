"""Entry point and frame loop."""

import sys

import pygame

from ..render import palette, scale, widgets
from ..render.camera import Camera
from ..render.draw import ViewSettings, WorldRenderer
from ..render.explain import ExplainPanel
from ..render.hud import Hud
from ..render.pixelfont import PixelFont
from ..render.sprites import SpriteCache
from ..render.starfield import Starfield
from ..sim import Simulation
from .config import open_display
from .input import handle_event
from .saves import SlotStore, default_directory
from .state import AppState
from .tools import SelectTool
from .ui import UiLayer


def probe(fullscreen=False):
    """Report what SDL actually gives us.  Answers the Retina question."""
    # Display and font only.  There is no sound in the sandbox, and the
    # mixer's CoreAudio init can stall for fifteen seconds on a machine whose
    # audio device is in a bad state -- a fifteen-second boot for a subsystem
    # nothing here uses.
    pygame.display.init()
    pygame.font.init()
    ctx = open_display(fullscreen=fullscreen)
    info = {
        "desktop_size": pygame.display.get_desktop_sizes(),
        "window_size": ctx.window,
        "driver": pygame.display.get_driver(),
        "pixels_per_point": ctx.native,
        "s": ctx.s,
        "buffer": (ctx.width, ctx.height),
        "font_px": ctx.font_px,
        "resizable": ctx.resizable,
        "window_preset": ctx.window_size_label,
    }
    for key, value in info.items():
        print("%-16s %s" % (key, value))
    ctx.close()
    pygame.quit()
    return info


class App:
    def __init__(self, fullscreen=False, saves_dir=None):
        # Display and font only.  There is no sound in the sandbox, and the
        # mixer's CoreAudio init can stall for fifteen seconds on a machine whose
        # audio device is in a bad state -- a fifteen-second boot for a subsystem
        # nothing here uses.
        pygame.display.init()
        pygame.font.init()
        self.ctx = open_display(fullscreen=fullscreen)
        self.clock = pygame.time.Clock()
        self.cache = SpriteCache()
        self.camera = Camera(self.ctx.width, self.ctx.height)
        self._build_render_stack()
        self.store = SlotStore(default_directory() if saves_dir is None else saves_dir)
        self.state = AppState(Simulation(), self.camera, ViewSettings(), store=self.store)
        self.state.on_resize(self.ctx)
        self.state.set_tool(SelectTool)
        # Opens onto New / Load / Quit, with the solar system frozen behind it.
        self.state.open_start()
        # Lay the buttons out once before the first event, so a click landing in
        # the opening frame hits a button rather than falling through to space.
        self.ui.build(self.state, (self.ctx.width, self.ctx.height), self.ctx)

    def _build_render_stack(self):
        """(Re)create everything that is sized in pixels.  Called on grid change."""
        self.font = PixelFont(self.ctx.font_px)
        self.camera.width, self.camera.height = self.ctx.width, self.ctx.height
        self.renderer = WorldRenderer(self.cache, self.font)
        self.hud = Hud(self.font)
        self.ui = UiLayer(self.font, self.hud)
        self.explain_panel = ExplainPanel(self.font)
        self.stars = Starfield(self.ctx.width, self.ctx.height)
        self.cache.clear()

    def step_once(self):
        """One frame: events, physics, drawing.

        Extracted from the loop so the smoke test can drive the real application
        without owning the loop.
        """
        # Everything sized in pixels is rebuilt when the buffer changes shape --
        # a grid switch, a drag on the window edge, a size preset, fullscreen.
        # Compared after the events rather than inside them, so a window dragged
        # across several sizes in one frame costs one rebuild, not several.
        was = (self.ctx.width, self.ctx.height)
        for event in pygame.event.get():
            handle_event(self.ctx.map_event(event), self.state, self.ctx, self.ui)
        # Safety net for a resize that reached us through a path we do not
        # listen for -- a maximise, a macOS tile, a window moved by script.
        # One cheap SDL call a frame, and a no-op whenever nothing moved.
        if self.ctx.size_changed():
            self.ctx.resize()
            self.state.on_resize(self.ctx)
        if (self.ctx.width, self.ctx.height) != was:
            self._build_render_stack()

        state = self.state
        # Held steppers keep firing between clicks; this is where they fire.
        self.ui.update(state, self.ctx)

        # What a throw can actually hit is what the eye can see: refreshed
        # once a frame, from the same camera the aim overlay draws with, so a
        # click and a collision agree even though the two are computed
        # nowhere near each other.
        world = state.sim.world
        drawn = scale.contact_radii_au(world.radii, state.camera.zoom,
                                       state.camera.height,
                                       real_size=state.view.real_size,
                                       aim_radii=scale.visual_radii(world))
        world.contact_radius[:world.n] = world.clamp_contact_radii(drawn)

        frame_dt = self.clock.tick(60) / 1000.0
        state.sim.advance(state.days_per_second * frame_dt)
        state.sim.update_trails()
        # The orbit timer, on the same once-a-frame grain as the trails.
        state.sample_timer()
        state.explain.settle(state.sim.world, pygame.time.get_ticks())
        state.camera.update_follow(state.sim.world)
        for impact in state.sim.impacts:
            state.effects.spawn(impact)

        surface = self.ctx.surface
        surface.fill(palette.SPACE)
        self.stars.draw(surface, state.camera)
        self.renderer.draw(surface, state.sim, state.camera, state.view,
                           selected_id=state.selected_id)
        # Wall-clock, not simulated time: an impact must look the same at
        # any time-scale, and frame_dt is real seconds.
        state.effects.update(frame_dt)
        state.effects.draw(surface, state.camera)
        if state.dialog is None:
            report = state.explain.report if state.explain.enabled else None
            if report is not None:
                m = self.hud.metrics(surface.get_width(), surface.get_height())
                state.explain_rect = self.explain_panel.draw(
                    surface, report, m,
                    widgets.overlay_x(m, state.show_body_list), state.show_maths)
            else:
                state.explain_rect = None
            state.tool.draw_overlay(surface, state, self.font)
            self.hud.draw(surface, state)
        # Built at the end of the frame, from the state that is about to be
        # shown: next frame's clicks then hit exactly what the eye can see.
        self.ui.build(state, surface.get_size(), self.ctx)
        self.ui.draw(surface, state)
        self.ctx.present()

    def run(self):
        while self.state.running:
            self.step_once()
        self.ctx.close()
        pygame.quit()
        return 0


def main():
    return App().run()


if __name__ == "__main__":
    sys.exit(main())
