"""How big to draw a body.

At true scale a solar-system view makes Earth about 0.001 pixels across, so some
cheat is compulsory.  This one is a floor rather than a multiplier:

    draw_px = max(zoom * r_true, floor_px(r_true))

The floor rises with real radius to the 0.35 power, so size *ordering* survives
-- the Sun reads bigger than Jupiter, which reads bigger than Earth -- while the
range compresses enormously.  Crucially the floor stays small in absolute terms,
so no body inflates far enough to swallow the orbits around it.  That is the
failure mode of a naive log exaggeration, which would draw Earth's disc a
hundred times wider than the Moon's entire orbit.

Because it is a floor, zooming in is continuous: at some point ``zoom * r_true``
overtakes it and the drawing simply becomes honest.  There is no mode switch and
nothing pops.

Real-size mode drops the floor to a single pixel.  The whole solar system
collapses to dots at once, which is the lesson, and it costs one branch.
"""

import numpy as np

from .. import units

#: The resolution the constants below were tuned at.  Everything scales from it.
REF_HEIGHT = 360.0

FLOOR_BASE = 2.0
FLOOR_EXP = 0.35
FLOOR_MIN = 1.0
#: The ceiling is what keeps the Sun's drawn disc smaller than Earth's orbit at
#: full-system zoom.  Uncapped, the 0.35 power puts the Sun at 31px against an
#: orbit only 24.7px across, and Earth is drawn *inside* the star.
FLOOR_MAX = 6.0


def _floor(r_true, unit):
    raw = FLOOR_BASE * unit * (r_true / units.EARTH_RADIUS) ** FLOOR_EXP
    return np.clip(raw, FLOOR_MIN * unit, FLOOR_MAX * unit)


def display_radius_px(r_true, zoom, internal_height, real_size=False):
    """Radius to draw, in internal-buffer pixels."""
    true_px = zoom * r_true
    if real_size:
        return max(true_px, 1.0)
    return float(max(true_px, _floor(r_true, internal_height / REF_HEIGHT)))


def display_radii_px(radii, zoom, internal_height, real_size=False):
    """Vectorised form for drawing every body in one pass."""
    radii = np.asarray(radii, dtype=np.float64)
    true_px = zoom * radii
    if real_size:
        return np.maximum(true_px, 1.0)
    return np.maximum(true_px, _floor(radii, internal_height / REF_HEIGHT))


#: How far past its true radius a body's contact radius may be stretched so
#: that aiming at what is drawn actually works.  Uncapped, the Sun's drawn
#: disc is 0.6 AU across at 10 px/AU -- wider than Mercury's own orbit -- and
#: would swallow it the moment the player zoomed out.  At 8x the Sun's
#: contact radius is 0.037 AU, comfortably inside Mercury's 0.31 AU
#: perihelion, and Earth's is inside the Moon's orbit by about the same
#: margin.
MAX_AIM_FACTOR = 8.0


def contact_radii_au(radii, zoom, internal_height, real_size=False):
    """Radii to collide against, in AU: what is drawn, capped so nothing
    stretches far enough to eat a neighbour's orbit.

    Collision *detection* is the only thing that ever reads this -- every
    outcome the physics computes (energy, fragment count, disruption
    threshold, ...) still uses the true radius. This is deliberately a
    render-layer computation handed to the physics as a plain array: the
    physics never learns what a camera is, and the render layer never learns
    what a collision is.
    """
    radii = np.asarray(radii, dtype=np.float64)
    drawn_px = display_radii_px(radii, zoom, internal_height, real_size)
    drawn_au = drawn_px / max(zoom, 1e-30)
    return np.clip(drawn_au, radii, radii * MAX_AIM_FACTOR)
