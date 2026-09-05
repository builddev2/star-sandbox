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

from .. import edits, maneuver, materials, orbits, units
from ..body import (DEFAULT_BLACK_HOLE_DISK, DEFAULT_BLACK_HOLE_SPIN,
                    KIND_BLACK_HOLE, KIND_CRAFT, KIND_PLANET, KIND_STAR)
from ..render import hud, palette, widgets
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


class DragPan:
    """A left button that both pans the camera and clicks what is under it.

    Look and Fly both do this, and getting it wrong either way is felt: a pan
    that also selects makes the panel flicker through every body it crosses,
    and a click that has to be perfectly still is beyond a child's hand.  So a
    press that travels more than ``CLICK_SLOP`` pans and never selects, and one
    that does not is a click, delivered to ``_on_click`` at the position the
    button came *up* -- where the finger ended, not where it started.
    """

    def __init__(self):
        #: Press position and whether it has become a drag, so a click can
        #: select and a drag can pan without the two fighting.
        self._press = None
        self._dragged = False

    def _pan_event(self, event, state):
        """Handle one press/drag/release.  Return True if it was consumed."""
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
            if pressed is not None and not self._dragged:
                self._on_click(event.pos, state)
            return True

        return False

    def _on_click(self, pos, state):
        """What a press that did not travel means for this tool."""


class TargetLock:
    """A body the tool is aimed at, picked by pressing X and clicking it.

    Throw and Fly both keep one.  Both hold the id rather than the row, because
    ``World.remove`` is a swap-remove and a row silently becomes another body;
    both drop the lock when the body is gone; and both label the button the
    same way apart from what it reads with nothing locked.
    """

    #: The button's label with no target locked.
    no_target_label = "Target lock"

    def __init__(self):
        #: True for the one click after pressing X: the next thing clicked
        #: becomes the target rather than doing whatever a click usually does.
        self.picking_target = False
        #: Body id of the locked target, or None.  Persists, so several shots
        #: at the same body do not need re-aiming each time.
        self.target_id = None

    def _target_index(self, world):
        """Row of the locked target, clearing the lock if it is gone."""
        if self.target_id is None:
            return None
        index = world.index_of(self.target_id)
        if index is None:
            self.target_id = None
        return index

    def _target_label(self, state):
        if self.picking_target:
            return "Click a body..."
        index = self._target_index(state.sim.world)
        if index is not None:
            return "Target: %s" % state.sim.world.bodies[index].name
        return self.no_target_label

    def _pick_target(self, event, state):
        """Consume the click that picks the target.  True if it was that click."""
        if not self.picking_target:
            return False
        index = state.pick_body_at(event.pos[0], event.pos[1])
        self.target_id = (state.sim.world.bodies[index].id
                          if index is not None else None)
        self.picking_target = False
        return True


class SelectTool(DragPan, Tool):
    name = "Look"
    hint = ("Click a body to see what it is  -  or pick one from the list  -  "
            "drag to pan, and the + and - buttons zoom  -  double-click to fly to it")

    def __init__(self):
        DragPan.__init__(self)
        self._last_click = (None, 0)

    def on_event(self, event, state):
        return self._pan_event(event, state)

    def _on_click(self, pos, state):
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
        state.select(body_id, now=now)
        self._last_click = (body_id, now)


@dataclass(frozen=True)
class LaunchPreset:
    key: int                 # 1..6, the number key that selects it
    label: str
    material_key: str
    radius_km: float
    #: Solar masses, or None to derive from the material's real density.
    mass: float = None
    #: Overrides the mass-derived kind.  A black hole is not a heavy star, and
    #: no threshold can tell them apart.
    kind: str = KIND_PLANET


PRESETS = (
    LaunchPreset(1, "Asteroid", "rock", 50.0),
    LaunchPreset(2, "Rocky planet", "rock", 6371.0),
    LaunchPreset(3, "Ice planet", "ice", 2500.0),
    LaunchPreset(4, "Gas giant", "gas", 69911.0),
    LaunchPreset(5, "Star", "hydrogen", 0.3 * 695700.0, mass=0.3),
    LaunchPreset(6, "Black hole", "singularity", 0.0, mass=10.0,
                 kind=KIND_BLACK_HOLE),
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
    if preset.kind == KIND_BLACK_HOLE:
        # radius_km means nothing here: a horizon is 2GM/c^2 and nothing else.
        return preset.mass, units.schwarzschild_radius(preset.mass)
    radius = preset.radius_km / units.AU_KM
    if preset.mass is not None:
        return preset.mass, radius
    density = materials.get(preset.material_key).density
    return units.FOUR_THIRDS_PI * density * radius ** 3, radius


class LaunchTool(TargetLock, Tool):
    name = "Throw"
    hint = ("Pick what to throw on the left  -  then press and drag in space to aim, "
            "and let go to launch  -  green comes back, red never does  -  "
            "X, then click a body, aims a guaranteed hit on it")

    context_title = "Throw"

    def __init__(self):
        TargetLock.__init__(self)
        self.preset_index = 0
        self.shift = False
        #: The circular-orbit button latches what holding Shift does, because a
        #: child cannot hold Shift and drag a mouse across a trackpad at once.
        self.lock_circular = False
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
        rows = []
        for k, preset in enumerate(PRESETS):
            if preset.kind == KIND_BLACK_HOLE:
                icon = "blackhole"
            elif preset.mass is not None:
                icon = "star"
            else:
                icon = "circle"
            rows.append([Control("throw.preset", label=preset.label, payload=k,
                                 icon=icon, hotkey=str(preset.key),
                                 tooltip="Throw %s" % preset.label.lower(),
                                 active=k == self.preset_index)])
        rows.append([Control("throw.circular", label="Circular orbit", icon="circle",
                             hotkey="Shift", active=self.lock_circular,
                             tooltip="Drop it straight into a perfect round orbit")])
        rows.append([Control("throw.target", label=self._target_label(state),
                             icon="target", hotkey="X",
                             active=self.picking_target or self.target_id is not None,
                             tooltip="Click a body to aim a guaranteed hit on it")])
        return rows

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
            if pygame.K_2 <= event.key <= pygame.K_6:
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
                and self._pick_target(event, state)):
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
                # The drag still decides which way round it goes.
                vel = orbits.circular_velocity(world, index,
                                               pos - world.pos[index], vel)
        return pos, vel

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

        kind = preset.kind
        if kind == KIND_PLANET and mass >= units.STAR_MASS_THRESHOLD:
            kind = KIND_STAR
        body = world.add(preset.label, materials.get(preset.material_key),
                         mass, radius, np.array(pos, dtype=float), vel,
                         kind=kind,
                         seed=(int(world.time * 1000) + world.n * 7919) % (1 << 30))
        if body is None:
            return
        if kind == KIND_BLACK_HOLE:
            body.spin = DEFAULT_BLACK_HOLE_SPIN
            body.disk = DEFAULT_BLACK_HOLE_DISK
        state.select(body.id)
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


#: Retained for callers that predate black holes and only ever meant the
#: ordinary case.
EDIT_FIELDS = edits.PLANET_EDIT_FIELDS

#: Screen pixels of horizontal drag per scrub notch.
DRAG_PIXELS_PER_NOTCH = 6.0


class EditTool(Tool):
    name = "Edit"
    hint = ("Click a body, then use the -- and + buttons  -  hold one to go faster "
            " -  Up/Down picks a row and Left/Right changes it  -  K locks a row "
            "so nothing can move it")
    context_title = "Edit"

    #: Notches one press of a stepper applies.  Mass moves three at a time
    #: because its scrub is a *ratio*: at 1.06 per notch a single one is
    #: invisible, and igniting Jupiter is eighty of them.
    CLICK_NOTCHES = {"mass": 3, "density": 1, "material": 1,
                     "temp": 3, "spin": 1, "disk": 1}

    def __init__(self):
        self.field_index = 0
        self._drag_x = None

    # --- buttons ----------------------------------------------------------------

    def _fields(self, world, index):
        return edits.edit_fields(world.bodies[index])

    def _field_count(self, state):
        """How many rows the *selected* body offers.

        A hotkey never passes through ``UiLayer.perform``, so it cannot lean on
        anything the panel knows -- it has to ask the same question the panel
        asked.  Bounding Up/Down with the module-level ``EDIT_FIELDS`` instead
        was harmless while every kind had three fields; a planet now has four
        and a star three, and the two must not be confused.
        """
        index = state.selected_index
        if index is None:
            return 1
        # max(1, ...): a craft's fields tuple is empty (edits.edit_fields), and
        # the Up/Down modulo below divides by this -- an empty selection must
        # not mean a crash.
        return max(1, len(self._fields(state.sim.world, index)))

    def _pick_row(self, state, field):
        """Point ``field_index`` at ``field``'s row.  False if there is no such row.

        The panel is built at the end of a frame from the body as it was then,
        and the press it answers arrives at the start of the next one -- or,
        for a held stepper, many frames later, because ``UiLayer`` keeps the
        pressed ``Control`` object while ``ui.build`` replaces the panel around
        it.  In between the body can change kind, and ``edit_fields`` then
        returns a different tuple: a planet that merges past the ignition
        threshold loses its Temp row, and ``fields.index("temp")`` on a star
        raises.  A destroyed body is the same story with ``selected_index``
        gone.

        Doing nothing is the whole fix.  Guessing a row by position would pin
        or scrub whichever field happened to land at that index instead, and
        the panel is rebuilt from the new kind before the next press anyway.
        """
        index = state.selected_index
        if index is None:
            return False
        fields = self._fields(state.sim.world, index)
        if field not in fields:
            return False
        self.field_index = fields.index(field)
        return True

    def field_values(self, world, index):
        """The editable values, formatted the way the panel shows them.

        A black hole has no density to hold constant and nothing it is made
        of, so it is offered its spin and how hard it is feeding instead.
        """
        body = world.bodies[index]
        if body.kind == KIND_BLACK_HOLE:
            return {
                "mass": hud.format_mass(float(world.mass[index])),
                "spin": "%.2f" % body.spin,
                "disk": "off" if body.disk <= 0.0 else "%.0f%%" % (body.disk * 100),
            }
        return {
            "mass": hud.format_mass(float(world.mass[index])),
            "density": "%.3g g/cm3" % units.to_gcm3(edits.body_density(world, index)),
            "temp": hud.format_temperature(float(world.temperature[index])),
            "material": world.bodies[index].material.name,
        }

    def status_label(self, world, index):
        body = world.bodies[index]
        if body.kind == KIND_BLACK_HOLE:
            return "nothing escapes"
        if body.kind == KIND_STAR:
            return "burning"
        return "%.0fx heavier to ignite" % edits.ignition_ratio(world, index)

    def controls(self, state, metrics):
        index = state.selected_index
        if index is None:
            return [[Control("edit.none", label="Click a body first", readout=True)]]

        world = state.sim.world
        body = world.bodies[index]
        fields = self._fields(world, index)
        self.field_index = min(self.field_index, len(fields) - 1)
        values = self.field_values(world, index)
        rows = []
        for k, field in enumerate(fields):
            step = self.CLICK_NOTCHES[field]
            # Material is a list of substances, not a quantity, so it gets
            # arrows rather than a minus and a plus, and it does not repeat.
            icons = ("left", "right") if field == "material" else ("minus", "plus")
            # Greying the steppers out is the *second* line of defence -- the
            # first is the guard in edits.py, which is what stops the drag and
            # the arrow keys, neither of which comes through a button.
            locked = edits.is_locked(body, field)
            rows.append([
                Control("edit.step", icon=icons[0], payload=(field, -step),
                        compact=True, repeat=field != "material",
                        enabled=not locked, tooltip="Less %s" % field),
                Control("edit.field", label="%s  %s" % (field.title(), values[field]),
                        payload=field, active=k == self.field_index,
                        tooltip="Or pick this row with Up/Down and change it "
                                "with Left/Right"),
                Control("edit.step", icon=icons[1], payload=(field, step),
                        compact=True, repeat=field != "material",
                        enabled=not locked, tooltip="More %s" % field),
                # The hotkey is appended by Control.hint(), and K acts on
                # whichever row is *selected* -- so the text must not read as a
                # promise about this row, which may not be the selected one.
                Control("edit.lock", icon="lock" if locked else "unlock",
                        payload=field, compact=True, hotkey="K", active=locked,
                        tooltip="%s %s -- K locks or frees the row that is picked"
                                % ("Unlock" if locked else "Lock", field)),
            ])
        # Unlock-all shares the Collapse row rather than taking one of its own:
        # at the interface floor a seventh row reaches past the transport bar.
        # A black hole has no Collapse to share with, so it gets the row.
        unlock_all = Control(
            "edit.unlock_all", label="Unlock all", icon="unlock", hotkey="U",
            enabled=bool(body.locked_fields),
            tooltip="Free every locked field on this body")
        if body.kind != KIND_BLACK_HOLE:
            rows.append([
                Control("edit.collapse", label="Collapse", icon="blackhole",
                        hotkey="C", enabled=body.kind == KIND_STAR,
                        tooltip="Crush this star into a black hole -- "
                                "there is no way back"),
                unlock_all,
            ])
        else:
            rows.append([unlock_all])
        rows.append([Control("edit.status", label=self.status_label(world, index),
                             readout=True)])
        return rows

    def perform(self, control, state):
        if control.id == "edit.field":
            return self._pick_row(state, control.payload)
        if control.id == "edit.step":
            field, notches = control.payload
            if not self._pick_row(state, field):
                return False
            self._scrub(state, notches)
            return True
        if control.id == "edit.lock":
            return self._toggle_lock(state, control.payload)
        if control.id == "edit.unlock_all":
            return self._unlock_all(state)
        if control.id == "edit.collapse":
            return self._collapse(state)
        return False

    def _collapse(self, state):
        """Crush the selected star into a black hole.

        Guarded on kind rather than left to ``edits.collapse``: the button is
        drawn disabled for anything that is not a star, and the hotkey must
        enforce the same rule -- keyboard events never pass through
        ``UiLayer.perform``, so the control's ``enabled`` flag cannot gate them.
        """
        index = state.selected_index
        if index is None:
            return False
        if state.sim.world.bodies[index].kind != KIND_STAR:
            return False
        edits.collapse(state.sim.world, index)
        state.sim.refresh()
        return True

    def _toggle_lock(self, state, field=None):
        """Lock or free one field on the selected body.

        ``field`` is the row whose padlock was pressed; the key path passes
        None and means "whichever row is selected", the way Left and Right
        already do.  A click also picks the row, so the arrows carry on from
        where the mouse left off.

        No ``sim.refresh()``: a lock changes no mass, no radius and no
        membership, so nothing the integrator caches has gone stale.
        """
        index = state.selected_index
        if index is None:
            return False
        world = state.sim.world
        if field is None:
            fields = self._fields(world, index)
            field = fields[min(self.field_index, len(fields) - 1)]
        elif not self._pick_row(state, field):
            return False
        edits.toggle_lock(world.bodies[index], field)
        return True

    def _unlock_all(self, state):
        """Free every field on the selected body at once.

        Re-checks that there is something to free rather than leaning on the
        button being drawn disabled: a keyboard event never passes through
        ``UiLayer.perform``, so the control's ``enabled`` flag cannot gate it --
        the same rule ``_collapse`` follows for ``C``.
        """
        index = state.selected_index
        if index is None:
            return False
        body = state.sim.world.bodies[index]
        if not body.locked_fields:
            return False
        edits.unlock_all(body)
        return True

    def on_event(self, event, state):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                return self._collapse(state)
            if event.key == pygame.K_k:
                return self._toggle_lock(state)
            if event.key == pygame.K_u:
                return self._unlock_all(state)
            if event.key == pygame.K_DOWN:
                self.field_index = min(self._field_count(state) - 1,
                                       self.field_index + 1)
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
                state.select(state.sim.world.bodies[index].id)
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
        fields = self._fields(world, index)
        field = fields[min(self.field_index, len(fields) - 1)]
        body = world.bodies[index]

        if field == "mass":
            edits.set_mass(world, index,
                           edits.scrub_mass(float(world.mass[index]), notches))
        elif field == "density":
            current = units.to_gcm3(edits.body_density(world, index))
            edits.set_density(
                world, index,
                units.from_gcm3(edits.scrub_density_gcm3(current, notches)))
        elif field == "temp":
            edits.set_temperature(
                world, index,
                edits.scrub_temperature(float(world.temperature[index]), notches))
        elif field == "spin":
            edits.set_spin(world, index, edits.scrub_spin(body.spin, notches))
        elif field == "disk":
            edits.set_disk(world, index, edits.scrub_disk(body.disk, notches))
        else:
            edits.cycle_material(world, index, 1 if notches > 0 else -1)

        state.sim.refresh()

    # This tool draws no overlay: its panel carries every value it used to
    # print over the world, and the inspector already names the body.  Two
    # copies of the same number is one of them eventually being wrong.


#: How far ahead of the body it launches from a craft is placed, as a
#: fraction of that body's own orbital radius.  Far enough that the two are
#: visibly separate at system zoom, near enough that a transfer from here is
#: the same transfer the body itself would fly.
LAUNCH_OFFSET = 0.02


class FlyTool(TargetLock, DragPan, Tool):
    name = "Fly"
    hint = ("Click a planet, then Launch craft  -  X locks somewhere to fly to, "
            "M works out the two burns  -  the new orbit is drawn before you "
            "commit, and nothing moves until the burn falls due")
    context_title = "Fly"

    no_target_label = "Pick somewhere to go"

    #: The rows the panel scrubs, in order.  Up/Down picks one.
    FIELDS = ("when", "prograde", "radial")
    #: Notches one press of a stepper applies.  All three move three at a
    #: time: a single 10 m/s notch is invisible against a 3,000 m/s burn.
    CLICK_NOTCHES = {"when": 3, "prograde": 3, "radial": 3}

    def __init__(self):
        TargetLock.__init__(self)
        DragPan.__init__(self)
        self.field_index = 0
        #: The craft this tool is flying, remembered so that looking at
        #: another body does not empty the whole panel under the cursor.
        self.craft_id = None
        #: The closest approach to the locked target, so the overlay can draw
        #: it without recomputing it.  Established here rather than left to
        #: spring into existence on first read.
        self._encounter_cache = None

    # --- what we are flying -----------------------------------------------------

    def _craft(self, state):
        """The craft this tool is flying.

        The selection first, when it *is* a craft -- that is how the rest of
        the interface says what it acts on.  Otherwise the craft last flown,
        for as long as it still exists.

        That fallback is not a nicety.  Locking a target means clicking a
        planet, and so does simply looking at one; without it, either action
        emptied every row of the panel and greyed out the buttons the player
        was in the middle of using.
        """
        body = state.selected_body
        if body is not None and body.kind == KIND_CRAFT:
            self.craft_id = body.id
            return body
        if self.craft_id is not None:
            remembered = state.sim.world.body_by_id(self.craft_id)
            if remembered is None:
                self.craft_id = None          # destroyed; stop pointing at it
            return remembered
        return None

    def _craft_index(self, state):
        """The craft's current row, or None.

        **Never ``state.selected_index``.**  The tool goes on flying a craft
        while the player looks at something else, so the selection and the
        craft are routinely different rows -- and when the player clicks empty
        space the selection is no row at all.  Passing that ``None`` into the
        physics did not raise where it happened: ``pos[None]`` quietly adds a
        numpy axis, and the failure surfaced several frames later inside an
        einsum.  Worse, when the selection *was* a valid row, a transfer got
        worked out from that body's orbit and written onto the craft.
        """
        craft = self._craft(state)
        if craft is None:
            return None
        return state.sim.world.index_of(craft.id)


    def _node(self, state):
        craft = self._craft(state)
        if craft is None or not craft.plan:
            return None
        return craft.plan[0]

    # --- launching --------------------------------------------------------------

    def _launch(self, state):
        """Put a craft in orbit beside the selected body.

        **Around whatever that body orbits, not around the body itself.**  A
        parking orbit 300 km above Earth has a ninety-minute period, and
        ``Integrator.dynamical_time`` takes the tightest pair in the *entire*
        system: one craft in low orbit would drive the block level to its
        ceiling and step the whole solar system in units of seconds.  The six
        existing moons already cost a factor of 64 in step size; this would be
        far worse.  Starting co-orbital costs nothing and is the right frame
        for an interplanetary transfer anyway.
        """
        world = state.sim.world
        index = state.selected_index
        if index is None or world.n >= world.capacity:
            return None

        attractor = orbits.dominant_attractor(world, index)
        if attractor is None:
            return None

        rel = world.pos[index] - world.pos[attractor]
        radius = float(np.hypot(rel[0], rel[1]))
        if radius <= 0.0:
            return None

        # A small rotation ahead of the parent, on the same circle.
        angle = LAUNCH_OFFSET
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        offset = np.array([cos_a * rel[0] - sin_a * rel[1],
                           sin_a * rel[0] + cos_a * rel[1]])

        # The parent's own motion decides which way round the craft goes.
        vel = orbits.circular_velocity(world, attractor, offset,
                                       world.vel[index] - world.vel[attractor])

        body = world.add("Shuttle", materials.SPACECRAFT,
                         units.EARTH_MASS * 1e-18, units.CRAFT_RADIUS,
                         world.pos[attractor] + offset, vel,
                         kind=KIND_CRAFT,
                         temperature=float(world.temperature[index]))
        if body is None:
            return None
        body.fuel_dv = units.from_ms(units.DEFAULT_FUEL_MS)
        # Recorded here rather than left for _craft to notice: launching is
        # what makes this the craft being flown, and the player may click
        # away to another body before _craft is ever asked.
        self.craft_id = body.id
        state.select(body.id)
        state.sim.refresh()
        return body

    # --- buttons ----------------------------------------------------------------

    def _default_lead(self, state):
        """Where a fresh node lands: a quarter of the way round the orbit.

        Far enough ahead to be worth planning, near enough to arrive while
        anyone is still watching.
        """
        index = self._craft_index(state)
        if index is None:
            return 30.0
        attractor = orbits.dominant_attractor(state.sim.world, index)
        if attractor is None:
            return 30.0
        world = state.sim.world
        mu = units.G * (float(world.mass[attractor]) + float(world.mass[index]))
        el = orbits.elements_from_state(
            mu, world.pos[index] - world.pos[attractor],
            world.vel[index] - world.vel[attractor])
        period = el["period"]
        return 30.0 if period is None else max(0.5, period * 0.25)

    def _ensure_node(self, state):
        """The head node, created at a sensible time if there is none."""
        craft = self._craft(state)
        if craft is None:
            return None
        if not craft.plan:
            due = state.sim.world.time + self._default_lead(state)
            craft.plan = (maneuver.Maneuver(due_time=due),)
            state.sim.refresh()
        return craft.plan[0]

    def _replace_node(self, state, node):
        craft = self._craft(state)
        if craft is None:
            return
        craft.plan = (node,) + tuple(craft.plan[1:])
        state.sim.refresh()

    def controls(self, state, metrics):
        craft = self._craft(state)
        has_craft = craft is not None
        node = self._node(state)
        world = state.sim.world
        target = self._target_index(world)

        when = ("--" if node is None
                else _lead_label(node.due_time - world.time))
        prograde = ("0 m/s" if node is None
                    else "{:+,.0f} m/s".format(units.to_ms(node.prograde)))
        radial = ("0 m/s" if node is None
                  else "{:+,.0f} m/s".format(units.to_ms(node.radial)))

        def stepper(cid, label, row):
            return [
                Control(cid, icon="minus", payload=-1, compact=True, repeat=True,
                        enabled=has_craft, active=self.field_index == row,
                        tooltip="Less"),
                Control(cid + ".readout", label=label, readout=True),
                Control(cid, icon="plus", payload=1, compact=True, repeat=True,
                        enabled=has_craft, active=self.field_index == row,
                        tooltip="More"),
            ]

        return [
            [Control("fly.launch",
                     # A greyed-out button with no reason on it is a dead end:
                     # it names what is missing instead.
                     label=("Launch craft" if state.selected_index is not None
                            else "Pick a body first"),
                     icon="fly",
                     enabled=state.selected_index is not None,
                     tooltip=("Put a shuttle in orbit beside what you picked"
                              if state.selected_index is not None
                              else "Click a planet first, then press this")),
             Control("fly.refuel",
                     label="Refuel" if has_craft else "No craft yet",
                     icon="reset", enabled=has_craft,
                     tooltip=("Fill the tank back up" if has_craft
                              else "Launch a craft first"))],
            [Control("fly.target", label=self._target_label(state), icon="target",
                     hotkey="X", active=self.picking_target or target is not None,
                     tooltip="Click a body to measure your approach to it")],
            stepper("fly.when", "When  %s" % when, 0),
            stepper("fly.prograde", "Along  %s" % prograde, 1),
            stepper("fly.radial", "Out    %s" % radial, 2),
            [Control("fly.hohmann",
                     label=("Work out the transfer" if has_craft and target is not None
                            else "Launch, then lock a target" if not has_craft
                            else "Lock a target first"),
                     icon="flyto",
                     hotkey="M", enabled=has_craft and target is not None,
                     tooltip=("Plan the two burns that get you there"
                              if has_craft and target is not None
                              else "Press X, then click where you want to go")),
             Control("fly.clear", icon="close", hotkey="N", compact=True,
                     enabled=has_craft and node is not None,
                     tooltip="Throw the plan away")],
        ]

    def perform(self, control, state):
        cid = control.id
        if cid == "fly.launch":
            self._launch(state)
            return True
        if cid == "fly.refuel":
            craft = self._craft(state)
            if craft is not None:
                craft.fuel_dv = units.from_ms(units.DEFAULT_FUEL_MS)
            return True
        if cid == "fly.target":
            self.picking_target = not self.picking_target
            return True
        if cid == "fly.clear":
            craft = self._craft(state)
            if craft is not None:
                craft.plan = ()
                state.sim.refresh()
            return True
        if cid == "fly.hohmann":
            self._plan_transfer(state)
            return True
        if cid in ("fly.when", "fly.prograde", "fly.radial"):
            field = cid.split(".", 1)[1]
            self.field_index = self.FIELDS.index(field)
            self._scrub(state, field,
                        int(control.payload) * self.CLICK_NOTCHES[field])
            return True
        return False

    def _scrub(self, state, field, notches):
        node = self._ensure_node(state)
        if node is None:
            return
        world = state.sim.world
        if field == "when":
            lead = maneuver.scrub_lead(node.due_time - world.time, notches)
            node = maneuver.Maneuver(world.time + lead, node.prograde, node.radial)
        elif field == "prograde":
            node = maneuver.Maneuver(node.due_time,
                                     maneuver.scrub_dv(node.prograde, notches),
                                     node.radial)
        else:
            node = maneuver.Maneuver(node.due_time, node.prograde,
                                     maneuver.scrub_dv(node.radial, notches))
        self._replace_node(state, node)

    def _plan_transfer(self, state):
        """Fill the plan with the two burns of a Hohmann transfer."""
        craft = self._craft(state)
        world = state.sim.world
        index = self._craft_index(state)
        target = self._target_index(world)
        if craft is None or index is None or target is None:
            return

        state_now = maneuver.craft_state(world, index)
        if state_now is None:
            return
        attractor, mu, rel_pos, rel_vel = state_now
        if target == attractor:
            return

        target_rel = world.pos[target] - world.pos[attractor]
        target_vel = world.vel[target] - world.vel[attractor]

        # **Semi-major axes, not present distances.**  A Hohmann transfer is
        # derived for two circular orbits, and the one radius that stands in
        # for a real elliptical one is its semi-major axis.  Mars swings
        # between 1.38 and 1.67 AU: aiming at wherever it is standing today
        # sends the craft to a radius Mars has already left by the time it
        # arrives, and the phase-angle wait -- which assumes a circular mean
        # motion -- would be computed from a period that does not exist.
        # Measured, aiming at the instantaneous radius missed Mars by 1.9 AU.
        r1 = _mean_radius(mu, rel_pos, rel_vel)
        r2 = _mean_radius(mu, target_rel, target_vel)
        if r1 <= 0.0 or r2 <= 0.0:
            return

        solved = maneuver.hohmann(
            mu, r1, r2,
            math.atan2(float(rel_pos[1]), float(rel_pos[0])),
            math.atan2(float(target_rel[1]), float(target_rel[0])))
        if solved is None:
            # A degenerate orbit -- a craft sitting on its attractor's centre.
            # There is no transfer to describe, so leave the plan alone rather
            # than writing burns derived from a division that did not happen.
            return

        # The circular-orbit window is the first guess; this slides it until
        # the target really is at the arrival point, which matters because
        # Mars's true anomaly does not advance at its mean motion.
        wait = maneuver.refine_window(mu, rel_pos, rel_vel,
                                      target_rel, target_vel,
                                      solved["wait_days"],
                                      solved["transfer_days"])
        depart = world.time + wait
        craft.plan = (
            maneuver.Maneuver(depart, solved["depart_dv"], 0.0),
            maneuver.Maneuver(depart + solved["transfer_days"],
                              solved["arrive_dv"], 0.0),
        )
        state.sim.refresh()

    # --- events -----------------------------------------------------------------

    def on_event(self, event, state):
        """Keys.  Every one re-checks its own precondition, because a key
        never passes through ``UiLayer.perform``'s ``clickable`` gate -- a
        greyed-out button's hotkey would otherwise still work.
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_x:
                self.picking_target = not self.picking_target
                return True
            if event.key == pygame.K_n:
                craft = self._craft(state)
                if craft is not None and craft.plan:
                    craft.plan = ()
                    state.sim.refresh()
                return True
            if event.key == pygame.K_m:
                if (self._craft(state) is not None
                        and self._target_index(state.sim.world) is not None):
                    self._plan_transfer(state)
                return True
            if event.key in (pygame.K_UP, pygame.K_DOWN):
                step = -1 if event.key == pygame.K_UP else 1
                self.field_index = (self.field_index + step) % len(self.FIELDS)
                return True
            if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                if self._craft(state) is None:
                    return True
                field = self.FIELDS[self.field_index]
                step = -1 if event.key == pygame.K_LEFT else 1
                self._scrub(state, field, step * self.CLICK_NOTCHES[field])
                return True

        if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self._pick_target(event, state)):
            return True

        return self._pan_event(event, state)

    def _on_click(self, pos, state):
        # Every other tool that acts on a selection lets you click one -- Edit
        # does -- and Fly not doing so is exactly what made "press 4, click a
        # planet" appear to do nothing at all.
        index = state.pick_body_at(pos[0], pos[1])
        if index is not None:
            state.select(state.sim.world.bodies[index].id)
        else:
            state.selected_id = None

    # --- overlay ----------------------------------------------------------------

    #: How long an encounter result stays good, milliseconds.  The search is a
    #: few hundred Kepler propagations; recomputing it for every frame would be
    #: felt.  Any change to the plan, the craft or the target invalidates it
    #: immediately -- this only bounds how stale a result can get while the
    #: world moves underneath an unchanged plan.
    ENCOUNTER_TTL_MS = 200

    def _encounter_key(self, state):
        craft = self._craft(state)
        if craft is None:
            return None
        return (craft.id, self.target_id, craft.plan)

    def _encounter(self, state):
        """Closest approach to the locked target under the current plan, cached.

        The craft is flown forward through its whole plan first -- propagate to
        each node, apply that burn, carry on -- so what is measured is where
        the plan actually puts it, not where it is drifting now.
        """
        key = self._encounter_key(state)
        if key is None or self.target_id is None:
            self._encounter_cache = None
            return None

        now = pygame.time.get_ticks()
        cached = self._encounter_cache
        if (cached is not None and cached[0] == key
                and now - cached[1] < self.ENCOUNTER_TTL_MS):
            return cached[2]

        world = state.sim.world
        index = self._craft_index(state)
        target = self._target_index(world)
        found = maneuver.craft_state(world, index) if index is not None else None
        if found is None or target is None:
            self._encounter_cache = (key, now, None)
            return None

        attractor, mu, rel_pos, rel_vel = found
        if target == attractor:
            self._encounter_cache = (key, now, None)
            return None

        clock = float(world.time)
        for node in self._craft(state).plan:
            dt = float(node.due_time) - clock
            if dt > 0.0:
                rel_pos, rel_vel = orbits.propagate(mu, rel_pos, rel_vel, dt)
                clock = float(node.due_time)
            rel_vel = maneuver.burn_velocity(rel_pos, rel_vel, node)

        target_pos = world.pos[target] - world.pos[attractor]
        target_vel = world.vel[target] - world.vel[attractor]
        lead = max(0.0, clock - float(world.time))
        if lead > 0.0:
            target_pos, target_vel = orbits.propagate(mu, target_pos,
                                                      target_vel, lead)

        el = orbits.elements_from_state(mu, rel_pos, rel_vel)
        horizon = el["period"] if el["period"] is not None else 2000.0
        result = maneuver.closest_approach(mu, rel_pos, rel_vel,
                                           target_pos, target_vel,
                                           horizon_days=min(horizon, 4000.0),
                                           samples=200)
        result["attractor"] = attractor
        result["mu"] = mu
        self._encounter_cache = (key, now, result)
        return result

    def draw_overlay(self, surface, state, font):
        world = state.sim.world
        camera = state.camera

        target = self._target_index(world)
        if target is not None:
            tx, ty = camera.world_to_screen(world.pos[target])
            pygame.draw.circle(surface, palette.SELECT, (int(tx), int(ty)), 6, 1)

        craft = self._craft(state)
        if craft is None:
            return

        index = self._craft_index(state)
        if index is None:
            return
        attractor = self._draw_paths(surface, state, craft, index)
        self._draw_node(surface, state, craft, index, attractor)
        self._draw_readout(surface, state, font, craft)
        self._draw_assist(surface, state, font)

    def _draw_paths(self, surface, state, craft, index):
        """The orbit the craft is on now, and the one each burn puts it on."""
        world = state.sim.world
        camera = state.camera
        attractor, conics = maneuver.plan_conics(world, index, craft.plan,
                                                 samples=160)
        if attractor is None:
            return None
        origin = world.pos[attractor]

        current = maneuver.craft_state(world, index)
        if current is not None:
            points, bound = orbits.conic_points(current[1], current[2],
                                                current[3], samples=160)
            self._stroke(surface, camera, points, origin, bound,
                         palette.TRAIL_BASE)

        for k, (points, bound) in enumerate(conics):
            # The first burn's result is the one being aimed; anything after it
            # is context, so it is drawn a tier back.
            colour = palette.ACCENT if k == 0 else palette.AIM_GOOD
            self._stroke(surface, camera, points, origin, bound, colour)
        return attractor

    @staticmethod
    def _stroke(surface, camera, points, origin, bound, colour):
        if len(points) < 2:
            return
        projected = camera.world_to_screen_array(points + origin)
        # Clamp before casting: a near-parabolic arc can reach 1e12 px, which
        # overflows the C int pygame.draw.lines wants.
        np.clip(projected, -32000, 32000, out=projected)
        pygame.draw.lines(surface, colour, bool(bound),
                          projected.astype(int).tolist(), 1)

    def _draw_node(self, surface, state, craft, index, attractor):
        """A handle where the burn happens, on the orbit it happens from."""
        node = self._node(state)
        if node is None or attractor is None:
            return
        found = maneuver.node_state(state.sim.world, index, node)
        if found is None:
            return
        _mu, node_pos, _node_vel = found
        nx, ny = state.camera.world_to_screen(
            node_pos + state.sim.world.pos[attractor])
        nx, ny = int(np.clip(nx, -32000, 32000)), int(np.clip(ny, -32000, 32000))
        pygame.draw.circle(surface, palette.SELECT, (nx, ny), 4, 1)
        pygame.draw.line(surface, palette.SELECT, (nx - 7, ny), (nx + 7, ny), 1)
        pygame.draw.line(surface, palette.SELECT, (nx, ny - 7), (nx, ny + 7), 1)

    def shortfall_ms(self, state):
        """How much delta-v the whole plan needs beyond the tank, in m/s.

        The whole plan, not the next burn: two burns of 3,000 each fit
        individually in a 5,000 tank and do not fit together, and it is the
        pair that decides whether the trip happens.

        A finite tank only teaches anything if the interface admits when the
        plan costs more than it holds.  Silently scaling the burn down --
        which is what ``maneuver.apply_burn`` correctly does when the moment
        arrives -- leaves a child waiting four hundred days for a burn that
        fires at a fifteenth of its size and arrives nowhere, with nothing on
        screen having said why.
        """
        craft = self._craft(state)
        if craft is None or not craft.plan:
            return 0.0
        wanted = sum(node.magnitude for node in craft.plan)
        return max(0.0, units.to_ms(wanted - craft.fuel_dv))

    def readout_x(self, state, metrics):
        """Left edge for the overlay readout, in pixels.

        The body list is open by default and sits in exactly this corner, so
        the readout starts clear of it when it is showing -- otherwise the
        fuel figure is drawn underneath the drawer and is simply invisible
        the first time the tool is opened.
        """
        if state.show_body_list:
            return metrics.pad + widgets.body_list_width(metrics) + metrics.gap
        return 8

    def _draw_readout(self, surface, state, font, craft):
        """Fuel, the countdown, and where the plan puts you.

        In the overlay rather than in the panel because the panel is full: six
        rows is the floor-size limit ``tests/test_layout.py`` enforces.
        """
        world = state.sim.world
        x = self.readout_x(state, widgets.Metrics(font, *surface.get_size()))
        y = 34
        line = font.height + 2

        fuel_ms = units.to_ms(craft.fuel_dv)
        font.draw(surface, x, y, "Fuel   {:,.0f} m/s".format(fuel_ms),
                  palette.WARN if fuel_ms < 500.0 else palette.TEXT)
        bar = pygame.Rect(x, y + line, font.width("X") * 16, 3)
        widgets.draw_bar(surface, bar,
                         fuel_ms / max(units.DEFAULT_FUEL_MS, 1.0),
                         palette.ACCENT)
        y += line * 2

        node = self._node(state)
        if node is not None:
            font.draw(surface, x, y, "Burn in %s"
                      % _lead_label(node.due_time - world.time), palette.TEXT)
            y += line
            found = maneuver.node_state(world, self._craft_index(state), node)
            if found is not None:
                mu, pos, vel = found
                el = orbits.elements_from_state(
                    mu, pos, maneuver.burn_velocity(pos, vel, node))
                font.draw(surface, x, y, "Closest  %s"
                          % hud.format_length(el["periapsis"]), palette.TEXT_DIM)
                y += line
                farthest = ("never returns" if el["apoapsis"] is None
                            else hud.format_length(el["apoapsis"]))
                font.draw(surface, x, y, "Farthest %s" % farthest,
                          palette.TEXT_DIM)
                y += line

        short = self.shortfall_ms(state)
        if short > 0.0:
            font.draw(surface, x, y, "Short by {:,.0f} m/s".format(short),
                      palette.DANGER)
            y += line
            font.draw(surface, x, y, "the burn will cut out early",
                      palette.WARN)
            y += line

        encounter = self._encounter(state)
        target = self._target_index(world)
        if encounter is not None and target is not None:
            font.draw(surface, x, y, "Passes %s at %s"
                      % (world.bodies[target].name,
                         hud.format_length(encounter["distance"])),
                      palette.AIM_GOOD)

    def _draw_assist(self, surface, state, font):
        """The velocity triangle at a flyby.

        |v_inf| is the same going out as coming in; all the planet does is
        rotate it.  Drawing the craft's velocity, the planet's, and their
        difference is what makes that visible, and the two speeds beside them
        are what make it a lesson rather than a decoration.
        """
        encounter = self._encounter(state)
        if encounter is None:
            return
        world = state.sim.world
        target = self._target_index(world)
        if target is None or encounter["distance"] <= 0.0:
            return

        primary = float(world.mass[encounter["attractor"]])
        distance = float(np.hypot(*encounter["target_pos"]))
        hill = units.hill_radius(float(world.mass[target]), primary, distance)
        if encounter["distance"] > hill:
            return

        v_craft = encounter["craft_vel"]
        v_planet = encounter["target_vel"]
        v_inf = v_craft - v_planet

        # Which way the planet bends it is set by which side the craft passes,
        # and that is exactly what decides boost from brake: the sense is the
        # sign of the flyby's own angular momentum.
        offset = encounter["craft_pos"] - encounter["target_pos"]
        side = math.copysign(
            1.0, float(offset[0] * v_inf[1] - offset[1] * v_inf[0]) or 1.0)
        turned = maneuver.flyby(units.G * float(world.mass[target]), v_inf,
                                encounter["distance"], side=side)
        v_out = v_planet + turned["v_out"]

        origin = state.camera.world_to_screen(
            encounter["craft_pos"] + world.pos[encounter["attractor"]])
        if not (-32000 < origin[0] < 32000 and -32000 < origin[1] < 32000):
            return

        # One pixel scale for all three arrows, so the triangle actually closes.
        span = max(float(np.hypot(*v_craft)), 1e-12)
        pixels = 46.0 / span

        def arrow(vec, colour):
            end = (origin[0] + float(vec[0]) * pixels,
                   origin[1] - float(vec[1]) * pixels)
            pygame.draw.line(surface, colour,
                             (int(origin[0]), int(origin[1])),
                             (int(np.clip(end[0], -32000, 32000)),
                              int(np.clip(end[1], -32000, 32000))), 1)
            return end

        arrow(v_planet, palette.TEXT_DIM)
        tip = arrow(v_craft, palette.ACCENT)
        arrow(v_out, palette.AIM_GOOD)

        font.draw(surface, int(tip[0]) + 4, int(tip[1]),
                  "in  %s" % hud.format_speed(float(np.hypot(*v_craft))),
                  palette.ACCENT)
        font.draw(surface, int(tip[0]) + 4, int(tip[1]) + font.height + 1,
                  "out %s" % hud.format_speed(float(np.hypot(*v_out))),
                  palette.AIM_GOOD)


def _mean_radius(mu, rel_pos, rel_vel):
    """The radius a body's orbit stands in for, in AU: its semi-major axis.

    Falls back to the present distance for an unbound path, which has no
    semi-major axis worth the name, and to 0.0 for a body sitting exactly on
    its attractor -- ``elements_from_state`` divides by the separation, so it
    cannot be asked about a body that has none.
    """
    distance = float(np.hypot(rel_pos[0], rel_pos[1]))
    if distance <= 0.0:
        return 0.0
    el = orbits.elements_from_state(mu, rel_pos, rel_vel)
    a = el["a"]
    if el["bound"] and math.isfinite(a) and a > 0.0:
        return float(a)
    return distance


def _lead_label(days):
    """A countdown in whatever unit keeps the number readable."""
    if days < 0.0:
        return "now"
    if days < 1.0:
        return "%.1f hr" % (days * 24.0)
    if days < 2.0 * units.DAYS_PER_YEAR:
        return "{:,.0f} d".format(days)
    return "%.1f yr" % units.to_years(days)


TOOLS = {
    pygame.K_1: SelectTool,
    pygame.K_2: LaunchTool,
    pygame.K_3: EditTool,
    pygame.K_4: FlyTool,
}

#: The same four tools addressed by name, for the buttons.  Keyed off the tool
#: rather than duplicated, so the two routes cannot disagree about what "Throw"
#: means.
TOOL_BY_NAME = {cls.name.lower(): cls for cls in TOOLS.values()}
