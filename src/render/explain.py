"""The "what changed" panel.

Floating over the world rather than in a column, because both columns are
already full: the Edit panel is at the six-row floor ``tests/test_layout.py``
enforces, and the right column has the inspector, the action grid and the zoom
cluster colliding at small sizes.  A floating panel owes nothing to that layout
and so cannot push any of it off screen.

Every number arrives as a ``Row`` carrying a *format tag*, never as text.
``src/explain.py`` may not import pygame, so it cannot reach the ``hud.format_*``
functions; resolving the tag is this module's job, which keeps exactly one
place in the program knowing what a mass looks like written down.
"""

import pygame

from .. import units
from . import hud, palette

#: Characters wide.  The inspector's width (``hud.PANEL_WIDTH_CHARS``), so the
#: interface has one text-panel width rather than two.
PANEL_CHARS = hud.PANEL_WIDTH_CHARS

TITLE = "WHAT CHANGED"

#: Tag -> the function that turns a raw simulation number into words.  The one
#: table that crosses the physics/render line.
FORMATTERS = {
    "mass": hud.format_mass,
    "length": hud.format_length,
    "temperature": hud.format_body_temperature,
    "period": hud.orbit_period_label,
    "density": lambda density: "%.3g g/cm3" % units.to_gcm3(density),
    "plain": lambda value: "%.3f" % value,
}


class ExplainPanel:
    def __init__(self, font):
        self.font = font
        self.line = font.height + 2

    def format_row(self, row):
        """(before, after) as text, through the formatter the row names."""
        formatter = FORMATTERS.get(row.format, FORMATTERS["plain"])
        return formatter(row.before), formatter(row.after)

    def lines(self, report, width_chars, show_maths):
        """Every line the panel will draw, as (text, colour).

        Built once and used for both sizing and drawing, so the rect can never
        be a line out of step with what goes in it.
        """
        out = [(TITLE, palette.SELECT), ("", palette.TEXT)]
        out.append((report.action, palette.TEXT))
        out.append(("", palette.TEXT))
        for change in report.changes:
            out.append((change.name,
                        palette.SELECT if change.is_edited else palette.ACCENT))
            for row in change.rows:
                was, now = self.format_row(row)
                out.append(("  %-9s %s -> %s" % (row.label.lower(), was, now),
                            palette.TEXT))
        out.append(("", palette.TEXT))
        for sentence in report.why:
            out += [(part, palette.TEXT_DIM)
                    for part in _wrap(sentence, width_chars)]
        if show_maths and report.maths:
            out.append(("", palette.TEXT))
            for formula in report.maths:
                out += [(part, palette.ACCENT)
                        for part in _wrap(formula.general, width_chars)]
                out += [(part, palette.TEXT_DIM)
                        for part in _wrap(formula.substituted, width_chars)]
        return out

    def _chars(self, metrics, x):
        """Characters that fit between ``x`` and the right edge.

        The panel wants ``PANEL_CHARS``, and takes fewer when there is not
        room: at ``config.min_internal(3)`` -- a 213x196 buffer -- a
        34-character panel starting clear of the body-list drawer reaches 217
        pixels against a 213-pixel window, and carries its own close button
        off the edge.  Wrapping to a width the panel does not have is the same
        bug one layer down, so the text follows this number too.
        """
        room = metrics.width - x - metrics.pad
        usable = (room - metrics.pad * 2) // max(1, self.font.width("X"))
        return max(8, min(PANEL_CHARS, usable))

    def rect(self, report, metrics, x, show_maths):
        """Where the panel sits.  Clipped to the content band, never past it."""
        chars = self._chars(metrics, x)
        lines = self.lines(report, chars, show_maths)
        width = self.font.width("X") * chars + metrics.pad * 2
        height = len(lines) * self.line + metrics.pad * 2
        top = metrics.content_top
        room = metrics.content_bottom - top
        return pygame.Rect(x, top, width, min(height, max(self.line, room)))

    def draw(self, surface, report, metrics, x, show_maths):
        """Paint it, trimming rows from the end if the window is too short."""
        rect = self.rect(report, metrics, x, show_maths)
        box = pygame.Surface(rect.size, pygame.SRCALPHA)
        box.fill(tuple(palette.PANEL) + (232,))
        surface.blit(box, rect.topleft)
        pygame.draw.rect(surface, palette.PANEL_EDGE, rect, 1)

        room = (rect.height - metrics.pad * 2) // self.line
        for k, (text, colour) in enumerate(
                self.lines(report, self._chars(metrics, x), show_maths)[:max(1, room)]):
            self.font.draw(surface, rect.x + metrics.pad,
                           rect.y + metrics.pad + k * self.line, text, colour)
        return rect


def _wrap(text, chars):
    """Greedy word wrap.  Never returns an empty list, so a blank line stays
    a blank line rather than vanishing out of the layout."""
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = (current + " " + word).strip()
        if len(candidate) > chars and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    lines.append(current)
    return lines
