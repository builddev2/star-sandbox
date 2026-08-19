"""Interaction tools.

Each tool owns its own event handling and its own overlay drawing, so adding one
is a new class rather than another branch in a growing event handler.  Exactly
one tool is active; the HUD's hint line is whatever it says it is.
"""

import math
from dataclasses import dataclass

import numpy as np
import pygame

from .. import materials, orbits, units
from ..body import KIND_PLANET, KIND_STAR
from ..render import palette

#: A press-and-release within this many pixels counts as a click, not a drag.
CLICK_SLOP = 4
#: Two clicks within this many milliseconds on the same body zoom to it.
DOUBLE_CLICK_MS = 400


class Tool:
    name = "Tool"
    hint = ""

    def on_event(self, event, state):
        """Return True if the event was consumed."""
        return False

    def draw_overlay(self, surface, state, font):
        pass

    def on_exit(self, state):
        pass


class SelectTool(Tool):
    name = "Look"
    hint = ("Click a body to inspect it  -  drag to pan  -  wheel to zoom  -  "
            "double-click to fly to it  -  ? for controls")

    def __init__(self):
        self._press = None
        self._dragged = False
        self._last_click = (None, 0)

    def on_event(self, event, state):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._press = event.pos
            self._dragged = False
            return True

        if event.type == pygame.MOUSEMOTION and self._press is not None:
            if event.buttons[0]:
                if (abs(event.pos[0] - self._press[0]) > CLICK_SLOP
                        or abs(event.pos[1] - self._press[1]) > CLICK_SLOP):
                    self._dragged = True
                if self._dragged:
                    state.camera.pan_pixels(event.rel[0], event.rel[1])
                return True

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            pressed, self._press = self._press, None
            if pressed is None or self._dragged:
                return True
            self._click(event.pos, state)
            return True

        return False

    def _click(self, pos, state):
        index = state.pick_body_at(pos[0], pos[1])
        if index is None:
            state.selected_id = None
            self._last_click = (None, 0)
            return

        body_id = state.sim.world.bodies[index].id
        now = pygame.time.get_ticks()
        previous_id, previous_time = self._last_click
        if previous_id == body_id and now - previous_time < DOUBLE_CLICK_MS:
            state.camera.frame_body(state.sim.world, index)
            state.camera.follow_id = body_id
        state.selected_id = body_id
        self._last_click = (body_id, now)


@dataclass(frozen=True)
class LaunchPreset:
    key: int                 # 1..5, the number key that selects it
    label: str
    material_key: str
    radius_km: float
    #: Solar masses, or None to derive from the material's real density.
    mass: float = None


PRESETS = (
    LaunchPreset(1, "Asteroid", "rock", 50.0),
    LaunchPreset(2, "Rocky planet", "rock", 6371.0),
    LaunchPreset(3, "Ice planet", "ice", 2500.0),
    LaunchPreset(4, "Gas giant", "gas", 69911.0),
    LaunchPreset(5, "Star", "hydrogen", 0.3 * 695700.0, mass=0.3),
)

#: A drag this fraction of the window height throws at exactly the speed of a
#: circular orbit at that point.  Escape is 1.41x that, so the whole interesting
#: range -- limp lob to gone forever -- fits inside a comfortable quarter-screen
#: gesture, at *any* zoom.  A fixed pixels-per-AU/day scale cannot do that: it
#: means something different at every zoom level, and at solar-system zoom you
#: would have to drag most of the screen just to reach escape velocity.
DRAG_FRACTION = 0.18
#: Fallback speed scale when there is nothing to orbit, in AU/day.
LONE_DRAG_SPEED = 0.02


def preset_body(preset):
    """(mass, radius) in simulation units for a launch preset."""
    radius = preset.radius_km / units.AU_KM
    if preset.mass is not None:
        return preset.mass, radius
    density = materials.get(preset.material_key).density
    return units.FOUR_THIRDS_PI * density * radius ** 3, radius


class LaunchTool(Tool):
    name = "Throw"
    hint = ("1-5 pick what to throw  -  press and drag to aim, release to launch  -  "
            "hold Shift for a perfect circular orbit")

    def __init__(self):
        self.preset_index = 0
        self.shift = False
        self._origin = None          # world position, set on press
        self._cursor = None          # screen position while dragging

    # --- events -----------------------------------------------------------------

    def on_event(self, event, state):
        if event.type == pygame.KEYDOWN:
            # K_1 is left alone so that pressing 1 returns to the Look tool.
            if pygame.K_2 <= event.key <= pygame.K_5:
                self.preset_index = event.key - pygame.K_1
                return True
            if event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                self.shift = True
                return True

        if event.type == pygame.KEYUP and event.key in (pygame.K_LSHIFT,
                                                        pygame.K_RSHIFT):
            self.shift = False
            return True

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._origin = state.camera.screen_to_world(*event.pos)
            self._cursor = event.pos
            return True

        if event.type == pygame.MOUSEMOTION and self._origin is not None:
            self._cursor = event.pos
            return True

        if (event.type == pygame.MOUSEBUTTONUP and event.button == 1
                and self._origin is not None):
            self._cursor = event.pos
            self._launch(state)
            self._origin = None
            self._cursor = None
            return True

        return False

    # --- physics ----------------------------------------------------------------

    def _reference_speed(self, state, pos):
        """Circular-orbit speed at ``pos``: the natural unit for a throw here."""
        world = state.sim.world
        index = self._nearest_attractor(world, pos)
        if index is None:
            return LONE_DRAG_SPEED
        rel = pos - world.pos[index]
        r = float(np.hypot(rel[0], rel[1]))
        if r <= 0.0:
            return LONE_DRAG_SPEED
        return math.sqrt(units.G * float(world.mass[index]) / r)

    def _drag_velocity(self, state):
        """Velocity implied by the current drag, in AU/day.

        Magnitude is measured against the local circular-orbit speed rather than
        against pixels, so the gesture means the same thing whether you are
        looking at the whole solar system or at one moon.
        """
        if self._origin is None or self._cursor is None:
            return np.zeros(2)
        start = state.camera.world_to_screen(self._origin)
        dx = self._cursor[0] - start[0]
        dy = -(self._cursor[1] - start[1])
        length = math.hypot(dx, dy)
        if length <= 0.0:
            return np.zeros(2)
        span = max(1.0, DRAG_FRACTION * state.camera.height)
        speed = self._reference_speed(state, self._origin) * length / span
        return np.array([dx, dy]) / length * speed

    def _tentative(self, state):
        """(position, velocity) the body would have if released right now."""
        pos = self._origin
        vel = self._drag_velocity(state)
        if self.shift:
            world = state.sim.world
            index = self._nearest_attractor(world, pos)
            if index is not None:
                rel = pos - world.pos[index]
                r = float(np.hypot(rel[0], rel[1]))
                if r > 0.0:
                    mu = units.G * world.mass[index]
                    tangent = np.array([-rel[1], rel[0]]) / r
                    # Match the drag's sense of rotation.
                    if float(np.dot(tangent, vel)) < 0.0:
                        tangent = -tangent
                    vel = world.vel[index] + tangent * math.sqrt(mu / r)
        return pos, vel

    @staticmethod
    def _nearest_attractor(world, pos):
        """Index of whatever pulls hardest at a point that is not yet a body.

        The full sphere-of-influence rule in orbits.py needs a body already in
        the world; this is the cheap stand-in for a point being aimed at.
        """
        if world.n == 0:
            return None
        d = world.positions - pos
        r2 = np.einsum("ij,ij->i", d, d)
        r2[r2 <= 0.0] = np.inf
        return int(np.argmax(world.masses / r2))

    def _launch(self, state):
        world = state.sim.world
        if world.n >= world.capacity:
            return
        preset = PRESETS[self.preset_index]
        mass, radius = preset_body(preset)
        pos, vel = self._tentative(state)

        kind = KIND_STAR if mass >= units.STAR_MASS_THRESHOLD else KIND_PLANET
        body = world.add(preset.label, materials.get(preset.material_key),
                         mass, radius, np.array(pos, dtype=float), vel,
                         kind=kind,
                         seed=(int(world.time * 1000) + world.n * 7919) % (1 << 30))
        if body is None:
            return
        state.selected_id = body.id
        state.sim.refresh()

    # --- overlay ----------------------------------------------------------------

    def draw_overlay(self, surface, state, font):
        preset = PRESETS[self.preset_index]
        if self._origin is None or self._cursor is None:
            self._draw_palette(surface, state, font)
            return

        camera = state.camera
        world = state.sim.world
        pos, vel = self._tentative(state)

        start = camera.world_to_screen(pos)
        pygame.draw.line(surface, palette.TEXT, start, self._cursor, 1)
        pygame.draw.circle(surface, palette.SELECT,
                           (int(start[0]), int(start[1])), 4, 1)

        index = self._nearest_attractor(world, pos)
        if index is not None:
            mu = units.G * float(world.mass[index])
            points, bound = orbits.conic_points(
                mu, pos - world.pos[index], vel - world.vel[index], samples=160)
            if len(points) >= 2:
                projected = camera.world_to_screen_array(points + world.pos[index])
                # Clamp before casting: a near-parabolic arc can reach 1e12 px,
                # which overflows the C int pygame.draw.lines wants.
                np.clip(projected, -32000, 32000, out=projected)
                colour = palette.AIM_GOOD if bound else palette.AIM_ESCAPE
                pygame.draw.lines(surface, colour, bool(bound),
                                  projected.astype(int).tolist(), 1)

        speed = float(np.hypot(vel[0], vel[1]))
        label = "%.4g km/s" % units.to_kms(speed)
        if self.shift:
            label += "   circular"
        font.draw(surface, self._cursor[0] + 8, self._cursor[1] + 8,
                  label, palette.TEXT)
        font.draw(surface, self._cursor[0] + 8, self._cursor[1] + 8 + font.height,
                  preset.label, palette.TEXT_DIM)

    def _draw_palette(self, surface, state, font):
        """The row of things you can throw, along the bottom."""
        line = font.height + 2
        y = surface.get_height() - line * 2 - 10
        x = 8
        for k, preset in enumerate(PRESETS):
            active = k == self.preset_index
            text = "%d %s" % (preset.key, preset.label)
            colour = palette.SELECT if active else palette.TEXT_DIM
            x += font.draw(surface, x, y, text, colour) + 18


TOOLS = {
    pygame.K_1: SelectTool,
    pygame.K_2: LaunchTool,
}
