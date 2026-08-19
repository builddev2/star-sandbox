"""Entry point and frame loop."""

import sys

import pygame

from ..render import palette
from ..render.camera import Camera
from ..render.draw import ViewSettings, WorldRenderer
from ..render.pixelfont import PixelFont
from ..render.sprites import SpriteCache
from ..render.starfield import Starfield
from ..sim import Simulation
from .config import DEFAULT_S, open_display


def probe(fullscreen=False):
    """Report what SDL actually gives us.  Answers the Retina question."""
    pygame.init()
    ctx = open_display(fullscreen=fullscreen)
    info = {
        "desktop_size": pygame.display.get_desktop_sizes(),
        "internal_surface": ctx.surface.get_size(),
        "window_size": pygame.display.get_window_size(),
        "driver": pygame.display.get_driver(),
        "default_S": DEFAULT_S,
        "buffer": (ctx.width, ctx.height),
        "upscale": ctx.upscale,
        "font_px": ctx.font_px,
    }
    for key, value in info.items():
        print("%-18s %s" % (key, value))
    pygame.quit()
    return info


def main():
    pygame.init()
    ctx = open_display()
    clock = pygame.time.Clock()

    sim = Simulation()
    camera = Camera(ctx.width, ctx.height)
    font = PixelFont(ctx.font_px)
    renderer = WorldRenderer(SpriteCache(), font)
    stars = Starfield(ctx.width, ctx.height)
    view = ViewSettings()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                camera.zoom_by(event.y, mx, my)

        frame_dt = clock.tick(60) / 1000.0
        sim.advance(30.0 * frame_dt)
        sim.update_trails()
        camera.update_follow(sim.world)

        ctx.surface.fill(palette.SPACE)
        stars.draw(ctx.surface, camera)
        renderer.draw(ctx.surface, sim, camera, view)
        pygame.display.flip()

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
