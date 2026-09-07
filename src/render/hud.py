"""Numbers on screen.

The audience is an eight-year-old sitting next to an adult, so this shows real
figures in real units rather than icons: Earth masses, kilometres, g's, km/s.
Every number is derived from the simulation state at the moment it is drawn --
nothing is cached or precomputed, so a body edited mid-drag updates live.
"""

import numpy as np
import pygame

from .. import orbits, units
from ..body import KIND_BLACK_HOLE, KIND_STAR
from . import palette, widgets
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
    ("1 2 3 4", "Look / Throw / Edit / Fly"),
    ("M", "work out the transfer to the target (Fly)"),
    ("N", "throw the flight plan away (Fly)"),
    ("C", "collapse a star into a black hole (Edit)"),
    ("K", "lock or unlock the selected field (Edit)"),
    ("U", "unlock every field on the selected body (Edit)"),
    ("X", "lock a target (Throw, Fly)"),
    ("0", "stop the selected body dead"),
    ("T", "orbit trails on and off"),
    ("R", "real sizes (everything is a dot)"),
    ("L", "name labels"),
    ("Y", "explanations on and off"),
    ("G", "pixel grid size"),
    ("Del", "delete the selected body"),
    ("F5", "start the solar system over"),
    ("W", "window size -- two thirds, half, a third"),
    ("Cmd+F", "fullscreen"),
    ("?", "this help"),
    ("Esc", "deselect, or close this help"),
    ("Cmd+Q", "quit"),
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
    """Speed in km/s, and never in exponent notation -- see format_length.

    Only a horizon reaches the grouped-digit branch: Earth orbits at 29.8 km/s,
    Mercury at 47.9, and escaping the Sun's surface takes 617.
    """
    kms = units.to_kms(speed)
    if kms >= 1.0e4:
        return "{:,.0f} km/s".format(kms)
    return "%.4g km/s" % kms


def format_temperature(kelvin):
    """Kelvin, without exponent notation anywhere a black hole can reach.

    A hole's own heat runs from 0.4 billionths of a kelvin for the heaviest
    down to hundreds of millions for one scrubbed absurdly small, so no single
    unit reads well across the whole span.
    """
    if kelvin >= 1.0:
        return "{:,.0f} K".format(kelvin)
    if kelvin >= 0.001:
        return "%.4f K" % kelvin
    nano = kelvin * 1.0e9
    if nano >= 100.0:
        return "{:,.0f} billionths of a K".format(nano)
    return "%.3g billionths of a K" % nano


def format_body_temperature(kelvin):
    """Kelvin with Celsius beside it -- "254 K  (-19 C)".

    Two units rather than one because the two readers need different ones: the
    physics is in kelvin and every threshold in the sandbox is quoted there,
    while an eight-year-old knows what -19 C feels like and does not yet know
    what 254 K means.  Grouped digits, never an exponent, for the same reason
    ``format_length`` refuses them.
    """
    return "{:,.0f} K  ({:,.0f} C)".format(kelvin, units.to_celsius(kelvin))


def orbit_period_label(days):
    """A period in whichever unit reads better.  Mercury is 88 days; Neptune
    is 165 years, and "60,190 days" tells a child nothing."""
    if days <= 0.0:
        return "escaping"
    if days >= 2.0 * units.DAYS_PER_YEAR:
        return "%.1f yr" % units.to_years(days)
    return "%.1f days" % days


def orbit_facts(world, index):
    """Osculating orbit of ``index`` about whatever it is bound to.

    ``distance_label``, ``closest_label`` and ``farthest_label`` are formatted
    here rather than in the inspector because an unbound path needs *words*
    where a closed orbit has a number.  Semi-major axis is negative on a
    hyperbola and apoapsis does not exist at all, so a body thrown past escape
    velocity read "Distance   -1.44e+11 m" and then simply omitted its farthest
    distance -- a hole in the readout, where what is true is that it never
    comes back.
    """
    attractor = orbits.dominant_attractor(world, index)
    if attractor is None:
        return None
    mu = units.G * (world.mass[attractor] + world.mass[index])
    rel_pos = world.pos[index] - world.pos[attractor]
    rel_vel = world.vel[index] - world.vel[attractor]
    el = orbits.elements_from_state(mu, rel_pos, rel_vel)
    bound = el["bound"]

    period_label = "escaping" if not bound else orbit_period_label(el["period"])

    # An open path has no average distance to quote, so quote where it is now.
    separation = float(np.hypot(rel_pos[0], rel_pos[1]))
    distance_label = format_length(el["a"] if bound else separation)
    farthest_label = (format_length(el["apoapsis"]) if el["apoapsis"] is not None
                      else "never returns")

    return {
        "attractor": world.bodies[attractor].name,
        "a": el["a"],
        "e": el["e"],
        "bound": bound,
        "distance": separation,
        "period_label": period_label,
        "distance_label": distance_label,
        "periapsis": el["periapsis"],
        "apoapsis": el["apoapsis"],
        "closest_label": format_length(el["periapsis"]),
        "farthest_label": farthest_label,
        "speed": float(np.hypot(rel_vel[0], rel_vel[1])),
    }


# --- drawing ----------------------------------------------------------------------

class Hud:
    #: Follow/Fly-to/Stop/Delete now sit above the stats, always as a fixed
    #: 2x2 grid -- see the design spec's "Right context column" section for
    #: why fixed beats the old one-row-if-it-fits branch.
    ACTIONS_ROWS = 2

    def __init__(self, font):
        self.font = font
        self.line = font.height + 2

    def actions_band_height(self, m):
        """Pixels the fixed action grid reserves above the stats."""
        return self.ACTIONS_ROWS * m.btn_h + (self.ACTIONS_ROWS - 1) * m.gap

    def right_margin(self, m):
        """Clearance the right column keeps from the window's right edge.

        Wider than a bare ``PAD`` because the zoom cluster
        (``ui._build_zoom_cluster``) is anchored independently in the same
        bottom-right corner, at ``m.width - m.pad - m.icon_btn_w``.  Without
        this, a tall Edit panel at the window floor grows down and sideways
        into the zoom slider -- same bug class ``contact_radius`` exists to
        prevent, a control drawn where it cannot be read as what it is.
        """
        return PAD + m.icon_btn_w + m.gap

    def headline_stats(self, world, index):
        """The one or two numbers worth seeing before the chevron is clicked.

        Kind-specific, the same split `_rows` already makes: a star has
        nothing more headline-worthy than its mass, a planet's temperature is
        the number a child asks about second, and a black hole's horizon is
        the number that makes it a black hole.
        """
        body = world.bodies[index]
        mass = float(world.mass[index])
        if body.kind == KIND_BLACK_HOLE:
            return [("Mass", format_mass(mass)),
                    ("Horizon", format_length(float(world.radius[index])))]
        if body.kind == KIND_STAR:
            return [("Mass", format_mass(mass))]
        return [("Mass", format_mass(mass)),
                ("Temp", format_body_temperature(float(world.temperature[index])))]

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

        if body.kind == KIND_BLACK_HOLE:
            spin = body.spin
            rows = [
                (body.name, palette.SELECT),
                ("black hole", palette.TEXT_DIM),
                ("", palette.TEXT),
                ("Mass       %s" % format_mass(mass), palette.TEXT),
                ("Horizon    %s" % format_length(radius), palette.TEXT),
                ("Shadow     %s" % format_length(units.shadow_radius(mass)),
                 palette.TEXT),
                ("Orbit edge %s" % format_length(units.isco_radius(mass, spin)),
                 palette.TEXT_DIM),
                ("Spin       %.3f" % spin, palette.TEXT),
                ("Feeding    %s" % ("nothing" if body.disk <= 0.0
                                    else "%.0f%%" % (body.disk * 100.0)),
                 palette.TEXT),
                ("Converts   %.1f%% to light"
                 % (units.accretion_efficiency(spin) * 100.0), palette.ACCENT),
                ("Density    %.3g g/cm3" % units.to_gcm3(density), palette.TEXT),
                ("Escape     %s"
                 % format_speed(units.escape_velocity(mass, radius)),
                 palette.WARN),
                ("Own heat   %s" % format_temperature(
                    units.hawking_temperature_k(mass)), palette.TEXT_DIM),
            ]
            if mass < units.TOV_LIMIT_MASS:
                rows.append(("Too light to be real -- under %g Suns"
                             % units.TOV_LIMIT_MASS, palette.WARN))
                rows.append(("nature makes a neutron star.", palette.WARN))
        else:
            rows = [
                (body.name, palette.SELECT),
                ("%s  -  %s" % (body.kind, body.material.name), palette.TEXT_DIM),
                ("", palette.TEXT),
                ("Mass       %s" % format_mass(mass), palette.TEXT),
                ("Radius     %s" % format_length(radius), palette.TEXT),
                ("           %.3g Earths" % units.to_earth_radii(radius),
                 palette.TEXT_DIM),
                ("Temp       %s" % format_body_temperature(
                    float(world.temperature[index])), palette.TEXT),
                ("Density    %.3g g/cm3" % units.to_gcm3(density), palette.TEXT),
                ("Gravity    %.3g g" % units.surface_gravity_g(mass, radius),
                 palette.TEXT),
                ("Escape     %s"
                 % format_speed(units.escape_velocity(mass, radius)),
                 palette.TEXT),
            ]

        facts = orbit_facts(world, index)
        if facts is not None:
            rows += [
                ("", palette.TEXT),
                ("Orbits     %s" % facts["attractor"], palette.ACCENT),
                ("Distance   %s" % facts["distance_label"], palette.TEXT),
                ("Stretch    %.3f" % facts["e"], palette.TEXT),
                ("One lap    %s" % facts["period_label"], palette.TEXT),
                # Closest and farthest are a pair: whatever the body is doing,
                # both rows are always there, so neither reads as a number the
                # readout failed to work out.
                ("Closest    %s" % facts["closest_label"], palette.TEXT),
                ("Farthest   %s" % facts["farthest_label"], palette.TEXT),
                ("Speed      %s" % format_speed(facts["speed"]), palette.TEXT),
            ]

        if body.kind not in (KIND_STAR, KIND_BLACK_HOLE) and mass > 0.0:
            rows += [
                ("", palette.TEXT),
                ("%.0fx heavier to ignite" % (units.STAR_MASS_THRESHOLD / mass),
                 palette.WARN),
            ]

        rows += [("", palette.TEXT), (body.material.blurb, palette.TEXT_FAINT)]
        return rows

    def bar_specs(self, world, index, rows):
        """(row_index, fraction, colour) for every row `_rows` gave a bar to.

        Matched by the same text prefix `_rows` already writes, so the two
        cannot drift apart without a row's own label changing too.
        """
        body = world.bodies[index]
        specs = []
        for i, (text, _colour) in enumerate(rows):
            if text.startswith("Temp") and body.kind not in (KIND_STAR, KIND_BLACK_HOLE):
                vapour = body.material.vapour_temperature_k
                frac = float(world.temperature[index]) / vapour if vapour > 0 else 0.0
                specs.append((i, frac, palette.WARN))
            elif "heavier to ignite" in text:
                mass = float(world.mass[index])
                frac = mass / units.STAR_MASS_THRESHOLD if mass > 0 else 0.0
                specs.append((i, frac, palette.ACCENT))
        return specs

    def visible_rows(self, state, index, height):
        """The inspector rows that fit above the buttons underneath it.

        Collapsed, this is identity plus `headline_stats` -- a glance.
        Expanded, it is the old trimmed-to-fit full readout.  Either way the
        panel that grows under a short window is the *expanded* one, since
        collapsed is already short enough to never need trimming in practice.
        """
        world = state.sim.world
        body = world.bodies[index]
        if not state.inspector_expanded:
            # Name plus the headline numbers only -- kind and material still
            # show, one click away, in the expanded `_rows` groups.  Keeping
            # the glance card to this few rows is what lets it fit at the
            # floor window size without trimming the numbers themselves.
            rows = [(body.name, palette.SELECT)]
            rows += [("%-10s %s" % (label, value), palette.TEXT)
                     for label, value in self.headline_stats(world, index)]
            return rows
        rows = self._rows(state, index)
        m = self.metrics(0, height)
        reserved = self.actions_band_height(m) + m.gap
        room = m.content_bottom - m.content_top - reserved - PAD * 2
        return rows[:max(1, room // self.line)]

    def inspector_rect(self, state, width, height):
        """Where the selected-body panel sits, or None with no selection.

        Starts a fixed `actions_band_height` below the top of the column --
        actions sit above this rect, not below it, so this method no longer
        needs to know anything about the action buttons themselves, only how
        tall their fixed band is.  ``ui.py`` reads this rect to anchor the
        actions, the expand chevron and the live tool's own controls, which is
        why it is computed here rather than inside the drawing code: one
        rectangle, several readers, no chance of them disagreeing.
        """
        index = state.selected_index
        if index is None:
            return None
        m = self.metrics(width, height)
        rows = self.visible_rows(state, index, height)
        panel_w = self.font.width("X") * PANEL_WIDTH_CHARS + PAD * 2
        panel_h = len(rows) * self.line + PAD * 2
        top = m.content_top + self.actions_band_height(m)
        return pygame.Rect(width - panel_w - self.right_margin(m), top,
                           panel_w, panel_h)

    def _draw_inspector(self, surface, state):
        index = state.selected_index
        if index is None:
            return
        width, height = surface.get_size()
        rect = self.inspector_rect(state, width, height)
        rows = self.visible_rows(state, index, height)
        self._panel(surface, rect)
        if not state.inspector_expanded:
            # The collapsed card is its own, slightly raised elevation tier --
            # one step lighter than the panel it sits in, the same ladder
            # `palette.py` defines for buttons, applied to a group of rows
            # instead of a control.
            pygame.draw.rect(surface, palette.PANEL_ROW, rect.inflate(-4, -4))
        bars = dict((i, (frac, colour)) for i, frac, colour
                   in self.bar_specs(state.sim.world, index, rows)) \
               if state.inspector_expanded else {}
        for k, (text, colour) in enumerate(rows):
            y = rect.y + PAD + k * self.line
            self.font.draw(surface, rect.x + PAD, y, text, colour)
            if k in bars:
                frac, bar_colour = bars[k]
                bar_w = self.font.width("X") * 10
                bar_rect = pygame.Rect(rect.right - PAD - bar_w,
                                       y + (self.line - 4) // 2, bar_w, 4)
                widgets.draw_bar(surface, bar_rect, frac, bar_colour)

    def _draw_hint(self, surface, state):
        width, height = surface.get_size()
        rect = pygame.Rect(0, height - self.line - PAD, width, self.line + PAD)
        self._panel(surface, rect)
        self.font.draw(surface, PAD, height - self.line - PAD // 2,
                       state.hint, palette.TEXT_DIM)

    def help_rect(self, width, height):
        """Where the help panel goes.  ``ui.py`` hangs its close button off this.

        Never wider or taller than what it is being drawn into.  This is the
        one panel whose size comes from its contents rather than from the
        window, so it is the one that can outgrow a small window, and a panel
        hanging over the edge takes its own close button with it.
        """
        char = self.font.width("X")
        panel_w = min(max(char * 46, self.font.width(HELP_INTRO)) + PAD * 4,
                      max(1, width - 2 * PAD))
        panel_h = min((len(KEY_HELP) + 4) * self.line + PAD * 2,
                      max(1, height - 2 * PAD))
        return pygame.Rect((width - panel_w) // 2, (height - panel_h) // 2,
                           panel_w, panel_h)

    def draw_help(self, surface):
        char = self.font.width("X")
        rect = self.help_rect(surface.get_width(), surface.get_height())
        self._panel(surface, rect)
        # Clipped to the panel: a description is as long as it needs to be to
        # read well, and text spilling across the world behind it looks like a
        # crash rather than like a window that is a little too small.
        previous = surface.get_clip()
        surface.set_clip(rect.clip(previous) if previous else rect)
        try:
            self.font.draw(surface, rect.x + PAD, rect.y + PAD, "Controls",
                           palette.SELECT)
            self.font.draw(surface, rect.x + PAD, rect.y + PAD + self.line,
                           HELP_INTRO, palette.TEXT_DIM)
            for k, (keys, description) in enumerate(KEY_HELP):
                y = rect.y + PAD + (k + 3) * self.line
                self.font.draw(surface, rect.x + PAD, y, keys, palette.ACCENT)
                self.font.draw(surface, rect.x + PAD + char * 13, y,
                               description, palette.TEXT)
        finally:
            surface.set_clip(previous)
