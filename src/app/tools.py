"""Interaction tools.

Each tool owns its own event handling, its own overlay drawing, and its own row
of buttons, so adding one is a new class rather than another branch in a growing
event handler.  Exactly one tool is active; the HUD's hint line and the panel
under the tool palette are whatever it says they are.

A tool never draws its own buttons or reads the mouse position to find them.  It
describes them (``controls``) and says what they mean (``perform``); ``ui.py``
lays them out, draws them and dispatches the clicks.  That is what keeps a
tool's mouse-drag handling and its buttons from fighting over the same press.
"""

import math
from dataclasses import dataclass

import numpy as np
import pygame

from .. import edits, materials, orbits, units
from ..body import KIND_PLANET, KIND_STAR
from ..render import palette
from ..render.widgets import Control

#: A press-and-release within this many pixels counts as a click, not a drag.
CLICK_SLOP = 4
#: Two clicks within this many milliseconds on the same body zoom to it.
DOUBLE_CLICK_MS = 400


class Tool:
    name = "Tool"
    hint = ""
    #: Heading for this tool's button panel; no panel is drawn without one.
    context_title = ""

    def on_event(self, event, state):
        """Return True if the event was consumed."""
        return False

    def controls(self, state, metrics):
        """Rows of :class:`Control` for the panel under the tool palette.

        A list of lists: each inner list is one row, laid out left to right.
        """
        return []

    def perform(self, control, state):
        """Act on one of this tool's controls.  Return True if handled."""
        return False

    def draw_overlay(self, surface, state, font):
        pass

    def on_exit(self, state):
        pass


class SelectTool(Tool):
    name = "Look"
    hint = ("Click a body to see what it is  -  or pick one from the list  -  "
            "drag to pan, and the + and - buttons zoom  -  double-click to fly to it")

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
    hint = ("Pick what to throw on the left  -  then press and drag in space to aim, "
            "and let go to launch  -  green comes back, red never does  -  "
            "X, then click a body, aims a guaranteed hit on it")

    context_title = "Throw"

    def __init__(self):
        self.preset_index = 0
        self.shift = False
        #: The circular-orbit button latches what holding Shift does, because a
        #: child cannot hold Shift and drag a mouse across a trackpad at once.
        self.lock_circular = False
        #: True for the one click after pressing X: the next thing clicked
        #: becomes the target rather than starting a throw.
        self.picking_target = False
        #: Body id of the locked target, or None.  Persists across throws so
        #: several shots at the same body do not need re-aiming each time.
        self.target_id = None
        #: Whether the most recent _tentative() found a real intercept for the
        #: locked target -- cached so the overlay's status label does not pay
        #: for a second angle-sweep the same frame.
        self._last_target_hit = None
        self._origin = None          # world position, set on press
        self._cursor = None          # screen position while dragging

    @property
    def circular(self):
        """Aim for a circular orbit -- held Shift or the latched button."""
        return self.shift or self.lock_circular

    # --- buttons ----------------------------------------------------------------

    def controls(self, state, metrics):
        rows = [[Control("throw.preset", label=preset.label, payload=k,
                         icon="star" if preset.mass is not None else "circle",
                         hotkey=str(preset.key),
                         tooltip="Throw %s" % preset.label.lower(),
                         active=k == self.preset_index)]
                for k, preset in enumerate(PRESETS)]
        rows.append([Control("throw.circular", label="Circular orbit", icon="circle",
                             hotkey="Shift", active=self.lock_circular,
                             tooltip="Drop it straight into a perfect round orbit")])
        rows.append([Control("throw.target", label=self._target_label(state),
                             icon="target", hotkey="X",
                             active=self.picking_target or self.target_id is not None,
                             tooltip="Click a body to aim a guaranteed hit on it")])
        return rows

    def _target_label(self, state):
        if self.picking_target:
            return "Click a body..."
        index = self._target_index(state.sim.world)
        if index is not None:
            return "Target: %s" % state.sim.world.bodies[index].name
        return "Target lock"

    def perform(self, control, state):
        if control.id == "throw.preset":
            self.preset_index = int(control.payload)
            return True
        if control.id == "throw.circular":
            self.lock_circular = not self.lock_circular
            return True
        if control.id == "throw.target":
            self.picking_target = not self.picking_target
            return True
        return False

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
            if event.key == pygame.K_x:
                self.picking_target = not self.picking_target
                return True

        if event.type == pygame.KEYUP and event.key in (pygame.K_LSHIFT,
                                                        pygame.K_RSHIFT):
            self.shift = False
            return True

        if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self.picking_target):
            index = state.pick_body_at(event.pos[0], event.pos[1])
            self.target_id = state.sim.world.bodies[index].id if index is not None else None
            self.picking_target = False
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
        speed = float(np.hypot(vel[0], vel[1]))
        self._last_target_hit = None
        if self.target_id is not None and speed > 0.0:
            solved = self._solve_target(state, pos, speed)
            if solved is not None:
                self._last_target_hit = solved["hit"]
                if solved["hit"]:
                    attractor_vel = state.sim.world.vel[solved["attractor"]]
                    return pos, attractor_vel + solved["velocity"]
        if self.circular:
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

    def _target_index(self, world):
        """Row of the locked target, clearing the lock if it no longer exists."""
        if self.target_id is None:
            return None
        index = world.index_of(self.target_id)
        if index is None:
            self.target_id = None
        return index

    def _solve_target(self, state, pos, speed):
        """Intercept solution for the locked target at this speed, or None.

        Both bodies are expressed relative to whatever the launch point's own
        gravity treats as dominant -- see :func:`orbits.solve_intercept` for
        why that is exact for the ordinary case and an estimate otherwise.
        """
        world = state.sim.world
        target = self._target_index(world)
        if target is None:
            return None
        attractor = self._nearest_attractor(world, pos)
        if attractor is None or attractor == target:
            return None
        rel = pos - world.pos[attractor]
        r = float(np.hypot(rel[0], rel[1]))
        if r <= 0.0:
            return None
        mu = units.G * float(world.mass[attractor])
        target_rel = world.pos[target] - world.pos[attractor]
        target_vel_rel = world.vel[target] - world.vel[attractor]
        result = orbits.solve_intercept(mu, rel, speed, target_rel, target_vel_rel)
        result["attractor"] = attractor
        return result

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
        camera = state.camera
        world = state.sim.world

        target = self._target_index(world)
        if target is not None:
            tx, ty = camera.world_to_screen(world.pos[target])
            tx, ty = int(tx), int(ty)
            pygame.draw.circle(surface, palette.SELECT, (tx, ty), 6, 1)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                pygame.draw.line(surface, palette.SELECT,
                                 (tx + dx * 8, ty + dy * 8),
                                 (tx + dx * 11, ty + dy * 11), 1)

        if self._origin is None or self._cursor is None:
            # What can be thrown is on the buttons now, not in a legend here.
            return

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
        if self.circular:
            label += "   circular"
        if target is not None:
            label += "   locked on %s" % world.bodies[target].name \
                     if self._last_target_hit else \
                     "   %s out of range" % world.bodies[target].name
        font.draw(surface, self._cursor[0] + 8, self._cursor[1] + 8,
                  label, palette.TEXT)
        font.draw(surface, self._cursor[0] + 8, self._cursor[1] + 8 + font.height,
                  preset.label, palette.TEXT_DIM)


EDIT_FIELDS = edits.EDIT_FIELDS

#: Screen pixels of horizontal drag per scrub notch.
DRAG_PIXELS_PER_NOTCH = 6.0


class EditTool(Tool):
    name = "Edit"
    hint = ("Click a body, then use the -- and + buttons  -  or Up/Down to pick a "
            "row and Left/Right to change it  -  hold a button to go faster")
    context_title = "Edit"

    #: Notches one press of a stepper applies.  Mass moves three at a time
    #: because its scrub is a *ratio*: at 1.06 per notch a single one is
    #: invisible, and igniting Jupiter is eighty of them.
    CLICK_NOTCHES = {"mass": 3, "density": 1, "material": 1}

    def __init__(self):
        self.field_index = 0
        self._drag_x = None

    # --- buttons ----------------------------------------------------------------

    def field_values(self, world, index):
        """The three editable numbers, formatted the way the panel shows them."""
        return {
            "mass": _mass_label(float(world.mass[index])),
            "density": "%.3g g/cm3" % units.to_gcm3(edits.body_density(world, index)),
            "material": world.bodies[index].material.name,
        }

    def status_label(self, world, index):
        if world.bodies[index].kind == KIND_STAR:
            return "burning"
        return "%.0fx heavier to ignite" % edits.ignition_ratio(world, index)

    def controls(self, state, metrics):
        index = state.selected_index
        if index is None:
            return [[Control("edit.none", label="Click a body first", readout=True)]]

        world = state.sim.world
        values = self.field_values(world, index)
        rows = []
        for k, field in enumerate(EDIT_FIELDS):
            step = self.CLICK_NOTCHES[field]
            # Material is a list of substances, not a quantity, so it gets
            # arrows rather than a minus and a plus, and it does not repeat.
            icons = ("left", "right") if field == "material" else ("minus", "plus")
            rows.append([
                Control("edit.step", icon=icons[0], payload=(field, -step),
                        compact=True, repeat=field != "material",
                        tooltip="Less %s" % field),
                Control("edit.field", label="%s  %s" % (field.title(), values[field]),
                        payload=field, active=k == self.field_index,
                        tooltip="Or pick this row with Up/Down and change it "
                                "with Left/Right"),
                Control("edit.step", icon=icons[1], payload=(field, step),
                        compact=True, repeat=field != "material",
                        tooltip="More %s" % field),
            ])
        rows.append([Control("edit.status", label=self.status_label(world, index),
                             readout=True)])
        return rows

    def perform(self, control, state):
        if control.id == "edit.field":
            self.field_index = EDIT_FIELDS.index(control.payload)
            return True
        if control.id == "edit.step":
            field, notches = control.payload
            self.field_index = EDIT_FIELDS.index(field)
            self._scrub(state, notches)
            return True
        return False

    def on_event(self, event, state):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DOWN:
                self.field_index = min(len(EDIT_FIELDS) - 1, self.field_index + 1)
                return True
            if event.key == pygame.K_UP:
                self.field_index = max(0, self.field_index - 1)
                return True
            if event.key == pygame.K_RIGHT:
                self._scrub(state, 1)
                return True
            if event.key == pygame.K_LEFT:
                self._scrub(state, -1)
                return True

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            index = state.pick_body_at(*event.pos)
            if index is not None:
                state.selected_id = state.sim.world.bodies[index].id
            self._drag_x = event.pos[0]
            return True

        if event.type == pygame.MOUSEMOTION and self._drag_x is not None:
            travelled = event.pos[0] - self._drag_x
            notches = int(travelled / DRAG_PIXELS_PER_NOTCH)
            if notches:
                self._scrub(state, notches)
                self._drag_x += notches * DRAG_PIXELS_PER_NOTCH
            return True

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._drag_x = None
            return True

        return False

    def _scrub(self, state, notches):
        index = state.selected_index
        if index is None:
            return
        world = state.sim.world
        field = EDIT_FIELDS[self.field_index]

        if field == "mass":
            edits.set_mass(world, index,
                           edits.scrub_mass(float(world.mass[index]), notches))
        elif field == "density":
            current = units.to_gcm3(edits.body_density(world, index))
            edits.set_density(
                world, index,
                units.from_gcm3(edits.scrub_density_gcm3(current, notches)))
        else:
            edits.cycle_material(world, index, 1 if notches > 0 else -1)

        state.sim.refresh()

    # This tool draws no overlay: its panel carries every value it used to
    # print over the world, and the inspector already names the body.  Two
    # copies of the same number is one of them eventually being wrong.


def _mass_label(mass):
    if mass >= 0.02:
        return "%.3g Suns" % mass
    if mass >= 0.3 * units.JUPITER_MASS:
        return "%.3g Jupiters" % units.to_jupiter_masses(mass)
    return "%.3g Earths" % units.to_earth_masses(mass)


TOOLS = {
    pygame.K_1: SelectTool,
    pygame.K_2: LaunchTool,
    pygame.K_3: EditTool,
}

#: The same three tools addressed by name, for the buttons.  Keyed off the tool
#: rather than duplicated, so the two routes cannot disagree about what "Throw"
#: means.
TOOL_BY_NAME = {cls.name.lower(): cls for cls in TOOLS.values()}
