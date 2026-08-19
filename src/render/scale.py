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
FLOOR_MIN = 1.5
FLOOR_MAX = 14.0


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
