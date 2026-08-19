"""Numbers on screen.

The audience is an eight-year-old sitting next to an adult, so this shows real
figures in real units rather than icons: Earth masses, kilometres, g's, km/s.
Every number is derived from the simulation state at the moment it is drawn --
nothing is cached or precomputed, so a body edited mid-drag updates live.
"""

import numpy as np
import pygame

from .. import orbits, units
from ..body import KIND_STAR
from . import palette
from .widgets import Metrics

PAD = 6
PANEL_WIDTH_CHARS = 34

#: Opening line of the help panel.  Nothing in this list is the *only* way to do
#: anything, and saying so is what stops the panel reading like a requirement.
HELP_INTRO = "Everything here is also a button on screen -- these are the shortcuts."

KEY_HELP = (
    ("Space", "pause and resume"),
    ("[  ]", "slower / faster"),
    ("Wheel", "zoom at the cursor"),
    ("+  -", "zoom in and out"),
    ("H", "back to the whole solar system"),
    ("Drag", "pan the view"),
    ("Click", "select a body"),
    ("Double-click", "fly to a body"),
    ("Tab", "cycle through bodies"),
    ("B", "the list of bodies"),
    ("F", "follow the selected body"),
    ("1 2 3", "Look / Throw / Edit"),
    ("0", "stop the selected body dead"),
    ("T", "orbit trails on and off"),
    ("R", "real sizes (everything is a dot)"),
    ("L", "name labels"),
    ("G", "pixel grid size"),
    ("Del", "delete the selected body"),
    ("F5", "start the solar system over"),
    ("Cmd+F", "fullscreen"),
    ("?", "this help"),
    ("Esc", "deselect, or close this help"),
)


# --- formatting -------------------------------------------------------------------

def format_mass(mass):
    """Mass in whichever familiar unit keeps the number readable."""
    if mass >= 0.02:
        return "%.3g Suns" % mass
    if mass >= 0.3 * units.JUPITER_MASS:
        return "%.3g Jupiters" % units.to_jupiter_masses(mass)
    if mass >= 1e-4 * units.EARTH_MASS:
        return "%.3g Earths" % units.to_earth_masses(mass)
    return "%.3g kg" % (mass * units.MSUN_KG)


def format_length(length):
    """Distances in AU, sizes in kilometres -- and never in exponent notation.

    "5.823e+04 km" is unreadable to the audience.  Grouped digits are not.
    """
    if length >= 0.02:
        return "%.4g AU" % length
    km = units.to_km(length)
    if km >= 10.0:
        return "{:,.0f} km".format(km)
    if km >= 0.01:
        return "%.2f km" % km
    return "%.3g m" % (km * 1000.0)


def format_speed(speed):
    return "%.4g km/s" % units.to_kms(speed)


def orbit_facts(world, index):
    """Osculating orbit of ``index`` about whatever it is bound to."""
    attractor = orbits.dominant_attractor(world, index)
    if attractor is None:
        return None
    mu = units.G * (world.mass[attractor] + world.mass[index])
    rel_pos = world.pos[index] - world.pos[attractor]
    rel_vel = world.vel[index] - world.vel[attractor]
    el = orbits.elements_from_state(mu, rel_pos, rel_vel)

    if not el["bound"]:
        period_label = "escaping"
    elif el["period"] >= 2.0 * units.DAYS_PER_YEAR:
        period_label = "%.1f yr" % units.to_years(el["period"])
    else:
        period_label = "%.1f days" % el["period"]

    return {
        "attractor": world.bodies[attractor].name,
        "a": el["a"],
        "e": el["e"],
        "period_label": period_label,
        "periapsis": el["periapsis"],
        "apoapsis": el["apoapsis"],
        "speed": float(np.hypot(rel_vel[0], rel_vel[1])),
    }


# --- drawing ----------------------------------------------------------------------

class Hud:
    def __init__(self, font):
        self.font = font
        self.line = font.height + 2

    def _panel(self, surface, rect):
        box = pygame.Surface(rect.size, pygame.SRCALPHA)
        box.fill(tuple(palette.PANEL) + (232,))
        surface.blit(box, rect.topleft)
        pygame.draw.rect(surface, palette.PANEL_EDGE, rect, 1)

    def metrics(self, width, height):
        return Metrics(self.font, width, height)

    def draw(self, surface, state):
        """The bars and the readout.

        The help panel is deliberately *not* drawn here: it has to sit over the
        buttons, and the buttons are drawn after this.  ``ui.py`` puts it up.
        """
        self._draw_top_bar(surface, state)
        self._draw_inspector(surface, state)
        self._draw_hint(surface, state)

    def _draw_top_bar(self, surface, state):
        width = surface.get_width()
        m = self.metrics(width, surface.get_height())
        rect = pygame.Rect(0, 0, width, m.top_h)
        self._panel(surface, rect)

        x = PAD
        y = (m.top_h - self.font.height) // 2
        x += self.font.draw(surface, x, y, state.elapsed_label, palette.TEXT) + 18
        x += self.font.draw(surface, x, y,
                            "PAUSED" if state.sim.paused else state.time_label,
                            palette.WARN if state.sim.paused else palette.ACCENT) + 18

        if state.sim.last_ratio < 0.95 and not state.sim.paused:
            x += self.font.draw(surface, x, y,
                                "running at %d%%" % int(state.sim.last_ratio * 100),
                                palette.WARN) + 18

        x += self.font.draw(surface, x, y,
                            "%d bodies" % state.sim.world.n, palette.TEXT_DIM) + 18
        self.font.draw(surface, x, y, state.tool_name, palette.SELECT)

    def _rows(self, state, index):
        world = state.sim.world
        body = world.bodies[index]
        mass = float(world.mass[index])
        radius = float(world.radius[index])
        density = units.density_from_mass_radius(mass, radius)

        rows = [
            (body.name, palette.SELECT),
            ("%s  -  %s" % (body.kind, body.material.name), palette.TEXT_DIM),
            ("", palette.TEXT),
            ("Mass       %s" % format_mass(mass), palette.TEXT),
            ("Radius     %s" % format_length(radius), palette.TEXT),
            ("           %.3g Earths" % units.to_earth_radii(radius), palette.TEXT_DIM),
            ("Density    %.3g g/cm3" % units.to_gcm3(density), palette.TEXT),
            ("Gravity    %.3g g" % units.surface_gravity_g(mass, radius), palette.TEXT),
            ("Escape     %s" % format_speed(units.escape_velocity(mass, radius)),
             palette.TEXT),
        ]

        facts = orbit_facts(world, index)
        if facts is not None:
            rows += [
                ("", palette.TEXT),
                ("Orbits     %s" % facts["attractor"], palette.ACCENT),
                ("Distance   %s" % format_length(facts["a"]), palette.TEXT),
                ("Stretch    %.3f" % facts["e"], palette.TEXT),
                ("One lap    %s" % facts["period_label"], palette.TEXT),
                ("Closest    %s" % format_length(facts["periapsis"]), palette.TEXT),
            ]
            if facts["apoapsis"] is not None:
                rows.append(("Farthest   %s" % format_length(facts["apoapsis"]),
                             palette.TEXT))
            rows.append(("Speed      %s" % format_speed(facts["speed"]), palette.TEXT))

        if body.kind != KIND_STAR and mass > 0.0:
            rows += [
                ("", palette.TEXT),
                ("%.0fx heavier to ignite" % (units.STAR_MASS_THRESHOLD / mass),
                 palette.WARN),
            ]

        rows += [("", palette.TEXT), (body.material.blurb, palette.TEXT_DIM)]
        return rows

    def visible_rows(self, state, index, height):
        """The inspector rows that fit above the buttons underneath it.

        On a short window the readout is trimmed rather than allowed to grow
        under the action buttons: a number you cannot see is worth less than a
        button you cannot press.
        """
        rows = self._rows(state, index)
        m = self.metrics(0, height)
        # Two button rows plus the gaps around them is the worst case the
        # interface can place below this panel.
        reserved = 2 * (m.btn_h + m.gap) + m.gap
        room = m.content_bottom - m.content_top - reserved - PAD * 2
        return rows[:max(1, room // self.line)]

    def inspector_rect(self, state, width, height):
        """Where the selected-body readout sits, or None when nothing is selected.

        ``ui.py`` anchors the Follow/Fly to/Stop/Delete buttons to this, which
        is why it is computed here rather than inside the drawing code: one
        rectangle, two readers, no chance of them disagreeing.
        """
        index = state.selected_index
        if index is None:
            return None
        m = self.metrics(width, height)
        rows = self.visible_rows(state, index, height)
        panel_w = self.font.width("X") * PANEL_WIDTH_CHARS + PAD * 2
        panel_h = len(rows) * self.line + PAD * 2
        return pygame.Rect(width - panel_w - PAD, m.content_top, panel_w, panel_h)

    def _draw_inspector(self, surface, state):
        index = state.selected_index
        if index is None:
            return
        width, height = surface.get_size()
        rect = self.inspector_rect(state, width, height)
        rows = self.visible_rows(state, index, height)
        self._panel(surface, rect)
        for k, (text, colour) in enumerate(rows):
            self.font.draw(surface, rect.x + PAD, rect.y + PAD + k * self.line,
                           text, colour)

    def _draw_hint(self, surface, state):
        width, height = surface.get_size()
        rect = pygame.Rect(0, height - self.line - PAD, width, self.line + PAD)
        self._panel(surface, rect)
        self.font.draw(surface, PAD, height - self.line - PAD // 2,
                       state.hint, palette.TEXT_DIM)

    def help_rect(self, width, height):
        """Where the help panel goes.  ``ui.py`` hangs its close button off this."""
        char = self.font.width("X")
        panel_w = max(char * 46, self.font.width(HELP_INTRO)) + PAD * 4
        panel_h = (len(KEY_HELP) + 4) * self.line + PAD * 2
        return pygame.Rect((width - panel_w) // 2, (height - panel_h) // 2,
                           panel_w, panel_h)

    def draw_help(self, surface):
        char = self.font.width("X")
        rect = self.help_rect(surface.get_width(), surface.get_height())
        self._panel(surface, rect)
        self.font.draw(surface, rect.x + PAD, rect.y + PAD, "Controls", palette.SELECT)
        self.font.draw(surface, rect.x + PAD, rect.y + PAD + self.line,
                       HELP_INTRO, palette.TEXT_DIM)
        for k, (keys, description) in enumerate(KEY_HELP):
            y = rect.y + PAD + (k + 3) * self.line
            self.font.draw(surface, rect.x + PAD, y, keys, palette.ACCENT)
            self.font.draw(surface, rect.x + PAD + char * 13, y,
                           description, palette.TEXT)
