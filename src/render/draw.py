"""Rendering the world: sky, trails, bodies, selection.

Two performance rules shape this file.

**Cull first.**  A body whose disc cannot touch the viewport is skipped before
its sprite is ever requested, which is what keeps the sprite cache from
thrashing when the camera is zoomed into a moon.

**Trails are drawn in chunks, not segments.**  A trail holds up to 512 points;
fading each segment individually would mean ~500 draw calls per body and roughly
10ms a frame across the system.  Eight chunks, each a step dimmer, costs eight
calls and looks the same at this resolution.
"""

import math
from dataclasses import dataclass

import pygame

from ..body import KIND_BLACK_HOLE, KIND_STAR
from . import palette, scale
from .sprites import ring_surface

#: Trail fade steps.  See the module docstring for why this is not per-segment.
TRAIL_CHUNKS = 8
#: A body must be within this many pixels of the viewport to be drawn.
CULL_MARGIN = 64
#: Labels are suppressed below this drawn radius to stop the view turning to soup.
LABEL_MIN_PX = 3
#: Past this many screen-heights of radius a body is drawn as a plain filled
#: circle rather than a generated sprite.  Zoomed all the way in, the Sun's
#: drawn radius reaches 4.6e9 pixels -- allocating that sprite is an instant
#: crash, and holding the scroll wheel is the easiest thing in the world to do.
MAX_SPRITE_HEIGHTS = 1.5
#: Hard ceiling on any radius handed to pygame.draw, which wants a C int.
MAX_DRAW_PX = 1_000_000
#: Extra padding around a label when testing whether it collides with another.
LABEL_PAD = 3


@dataclass
class ViewSettings:
    show_trails: bool = True
    real_size: bool = False
    show_labels: bool = True


def nearest_star_angle(world, index):
    """Direction from ``index`` toward the star lighting it, in radians.

    Falls back to a fixed angle when there is no star, so a debris-only world
    still shades consistently instead of flickering.
    """
    best = None
    best_d2 = float("inf")
    for k in range(world.n):
        if k == index or world.bodies[k].kind != KIND_STAR:
            continue
        rel = world.pos[k] - world.pos[index]
        d2 = float(rel[0] * rel[0] + rel[1] * rel[1])
        if d2 < best_d2:
            best_d2, best = d2, rel
    if best is None:
        return 0.0
    return float(math.atan2(best[1], best[0]))


class WorldRenderer:
    def __init__(self, cache, font):
        self.cache = cache
        self.font = font

    def draw(self, surface, sim, camera, view, selected_id=None):
        world = sim.world
        width, height = surface.get_size()
        if world.n == 0:
            return

        if view.show_trails:
            self._draw_trails(surface, world, camera, width, height)

        radii = scale.display_radii_px(world.radii, camera.zoom, height,
                                       real_size=view.real_size,
                                       floor_radii=scale.visual_radii(world))
        screen = camera.world_to_screen_array(world.positions)

        # Biggest first, so that when labels collide the more important body is
        # the one that keeps its name.  At system zoom Jupiter's moons sit within
        # a fraction of a pixel of the planet, and without this their four names
        # overprint each other into mush.
        order = sorted(range(world.n), key=lambda k: -radii[k])
        placed = []

        for i in order:
            px = radii[i]
            x, y = screen[i]
            if (x < -px - CULL_MARGIN or x > width + px + CULL_MARGIN
                    or y < -px - CULL_MARGIN or y > height + px + CULL_MARGIN):
                continue
            body = world.bodies[i]
            is_star = body.kind == KIND_STAR
            is_hole = body.kind == KIND_BLACK_HOLE

            if px < 2.0:
                if 0 <= int(x) < width and 0 <= int(y) < height:
                    surface.set_at((int(x), int(y)), body.material.tones[2])
            elif px > MAX_SPRITE_HEIGHTS * height:
                # Bigger than the screen: no sprite could add detail, and
                # allocating one this large would fail outright.  A hole this
                # close fills the view with its own shadow, which is correct.
                pygame.draw.circle(surface,
                                   (0, 0, 0) if is_hole
                                   else body.material.tones[1],
                                   (int(x), int(y)),
                                   int(min(px, MAX_DRAW_PX)))
            elif is_hole:
                # Never BLEND_ADD: additive blending cannot draw black, and
                # the shadow is the whole point.
                sprite = self.cache.get_black_hole(body.seed, int(px),
                                                   body.spin, body.disk)
                surface.blit(sprite, sprite.get_rect(center=(int(x), int(y))))
            else:
                if body.rings and px >= 4:
                    rings = ring_surface(int(px), body.material.tones)
                    surface.blit(rings, rings.get_rect(center=(int(x), int(y))))
                sprite = self.cache.get(body.material, body.seed, int(px),
                                        nearest_star_angle(world, i), is_star,
                                        float(world.temperature[i]))
                rect = sprite.get_rect(center=(int(x), int(y)))
                surface.blit(sprite, rect,
                             special_flags=pygame.BLEND_ADD if is_star else 0)

            if selected_id is not None and body.id == selected_id:
                pygame.draw.circle(surface, palette.SELECT, (int(x), int(y)),
                                   int(min(max(px, 3) + 5, MAX_DRAW_PX)), 1)

            if view.show_labels and px >= LABEL_MIN_PX:
                self._label(surface, body.name, x + px + 4,
                            y - self.font.height // 2, placed)

    def _label(self, surface, text, x, y, placed):
        """Draw a name unless it would overprint one already drawn."""
        box = pygame.Rect(int(x) - LABEL_PAD, int(y) - LABEL_PAD,
                          self.font.width(text) + LABEL_PAD * 2,
                          self.font.height + LABEL_PAD * 2)
        if box.collidelist(placed) != -1:
            return
        placed.append(box)
        self.font.draw(surface, int(x), int(y), text, palette.TEXT_DIM)

    def _draw_trails(self, surface, world, camera, width, height):
        for i in range(world.n):
            points = world.bodies[i].trail.ordered()
            if len(points) < 4:
                continue
            projected = camera.world_to_screen_array(points)
            # Cheap reject: a trail entirely off-screen costs nothing.
            if (projected[:, 0].max() < 0 or projected[:, 0].min() > width
                    or projected[:, 1].max() < 0 or projected[:, 1].min() > height):
                continue
            base = world.bodies[i].material.tones[1]
            per = max(2, len(projected) // TRAIL_CHUNKS)
            for chunk in range(TRAIL_CHUNKS):
                start = chunk * per
                stop = min(len(projected), start + per + 1)
                if stop - start < 2:
                    continue
                fade = 0.25 + 0.75 * (chunk + 1) / TRAIL_CHUNKS
                colour = tuple(int(palette.TRAIL_BASE[k] * (1 - fade) + base[k] * fade)
                               for k in range(3))
                pygame.draw.lines(surface, colour, False,
                                  projected[start:stop].astype(int).tolist(), 1)
