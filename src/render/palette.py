"""Theme colours.

Body colours are *not* here -- they live on the Material, because they are a
property of the substance rather than of the theme.  This module owns only the
chrome: space, panels, text, and the overlays the tools draw.
"""

import math

SPACE = (6, 8, 16)
PANEL = (14, 18, 30)
PANEL_EDGE = (44, 54, 78)
TEXT = (226, 234, 246)
TEXT_DIM = (128, 142, 168)
ACCENT = (108, 196, 232)
WARN = (236, 152, 72)
SELECT = (250, 226, 108)
TRAIL_BASE = (74, 96, 132)
STAR_GLOW = (255, 238, 190)
#: Aiming preview: a bound orbit vs one that leaves forever.
AIM_GOOD = (124, 226, 160)
AIM_ESCAPE = (236, 112, 112)

# --- interface chrome ---------------------------------------------------------
# Every button is drawn from these fills plus a border, so a new control cannot
# invent its own colour and drift away from the rest of the interface.  The
# four fills are one per interaction state: resting, under the cursor, held
# down, and switched on.
BUTTON = (26, 33, 52)
BUTTON_HOVER = (42, 54, 82)
BUTTON_DOWN = (56, 72, 108)
BUTTON_ACTIVE = (28, 62, 94)
BUTTON_EDGE = (58, 72, 104)
#: An active control is outlined in the accent colour, which is the one visual
#: cue that survives being read at a glance from across the room.
BUTTON_EDGE_ACTIVE = ACCENT
#: Label colour for a toggle that is switched off, dimmer than body text but
#: still clearly readable -- an off switch has to stay legible to be found.
BUTTON_TEXT_OFF = (152, 166, 192)
DISABLED = (74, 84, 104)
DANGER = (214, 92, 92)
TOOLTIP = (8, 11, 20)


# --- thermal emission ---------------------------------------------------------

def blackbody_rgb(kelvin):
    """The colour of a blackbody at ``kelvin``, as 0..255 RGB.

    A piecewise fit to the Planck locus -- the same curve a star's colour
    actually follows -- accurate to a few percent over 1000..40000 K, which is
    far tighter than a pixel-art sprite can show.  Below 1000 K a blackbody is
    too dim to have a useful colour at all, so it is clamped there and the
    caller's glow weight is what fades it in.

    This is one of the few colours in the sandbox that is *not* a theme choice:
    an ember is orange because of physics, not because the palette says so.
    """
    t = min(max(float(kelvin), 1000.0), 40000.0) / 100.0
    if t <= 66.0:
        red = 255.0
        green = 99.4708025861 * math.log(t) - 161.1195681661
    else:
        red = 329.698727446 * (t - 60.0) ** -0.1332047592
        green = 288.1221695283 * (t - 60.0) ** -0.0755148492
    if t >= 66.0:
        blue = 255.0
    elif t <= 19.0:
        blue = 0.0
    else:
        blue = 138.5177312231 * math.log(t - 10.0) - 305.0447927307
    return tuple(int(min(max(v, 0.0), 255.0)) for v in (red, green, blue))
