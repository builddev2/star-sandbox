"""Entry point and frame loop."""

import sys

import pygame

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
    probe()
    return 0


if __name__ == "__main__":
    sys.exit(main())
