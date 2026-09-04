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
from ..body import KIND_BLACK_HOLE, KIND_CRAFT

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


def visual_radii(world):
    """Per-body radius the *eye* should be served by, in AU.

    Equal to the true radius for everything except a black hole, whose horizon
    is seven orders of magnitude smaller than anything else in the simulation.
    Two things break because of that, and both are fixed here:

    **It would draw as one pixel.** A 10 solar mass hole's horizon is 0.0046
    Earth radii, which puts ``_floor`` at 0.31 px and so clamps up to
    FLOOR_MIN. The Sun draws at 6. The most dangerous object in the sandbox
    would be the least visible thing on screen.

    **It would be impossible to hit.** ``contact_radii_au`` caps the aim radius
    at MAX_AIM_FACTOR times the true radius -- 1.6e-6 AU for the same hole,
    about 0.0001 pixels across at system zoom.

    ``units.apparent_radius`` stands in: the main-sequence mass-radius
    relation, on the grounds that the accretion disk is roughly as big as the
    star that made the hole. **Physics never reads this.** Impact energy, tidal
    stress, disruption thresholds and merged radii all use the true horizon;
    only drawing, picking and the yes/no contact test see this array, which is
    exactly the split ``radius`` and ``contact_radius`` already make.

    A craft shares this branch for the same reason, at the opposite end of the
    scale: see the comment inline below.

    Recomputed per frame from at most 200 bodies -- nothing may cache a size
    across frames, and this is negligible beside the O(n^2) force kernel.
    """
    out = np.array(world.radii, dtype=np.float64)
    for i in range(world.n):
        kind = world.bodies[i].kind
        if kind == KIND_BLACK_HOLE:
            out[i] = units.apparent_radius(float(world.mass[i]))
        elif kind == KIND_CRAFT:
            # A 37 m hull is 1.2e-10 AU: the black hole's problem in
            # miniature.  It would draw as one pixel and, worse,
            # contact_radii_au would cap its aim radius far below a pixel,
            # making it impossible to hit.  Modest on purpose -- this also
            # caps the craft's contact radius, and a shuttle should be hard
            # to hit rather than easy.  (Clicking is unaffected either way --
            # PICK_MARGIN already covers a sub-pixel target; see
            # ``craft_floor_px`` for what actually keeps a craft clickable.)
            out[i] = units.craft_apparent_radius()
    return out


#: However far out the camera is, a craft is drawn at least this big, so its
#: silhouette stays readable.  Shared by ``render.draw`` (drawing) and
#: ``app.state.pick_body_at`` (picking) through ``craft_floor_px`` below --
#: they used to disagree, because only the draw side scaled with the window:
#: at h=360 that was a rounding error (drawn 9px, picked ~10px), but at
#: h=1440 the drawn floor had grown to 36px against a pick radius still
#: pinned near 12px, leaving two-thirds of the visible shuttle unclickable.
#: A render-only floor regardless: it must never reach ``contact_radius``,
#: so a bigger picture must not buy a bigger target to *hit*.
CRAFT_MIN_PX = 9


def craft_floor_px(internal_height):
    """The craft render/pick floor, in pixels, scaled like every other pixel
    constant by ``internal_height / REF_HEIGHT`` -- see the module docstring.
    """
    return CRAFT_MIN_PX * internal_height / REF_HEIGHT


def display_radius_px(r_true, zoom, internal_height, real_size=False):
    """Radius to draw, in internal-buffer pixels."""
    true_px = zoom * r_true
    if real_size:
        return max(true_px, 1.0)
    return float(max(true_px, _floor(r_true, internal_height / REF_HEIGHT)))


def display_radii_px(radii, zoom, internal_height, real_size=False,
                     floor_radii=None):
    """Vectorised form for drawing every body in one pass.

    ``floor_radii`` is what the *floor* is computed from, defaulting to the
    true radii. Only a black hole passes anything different -- see
    :func:`visual_radii`. The honest ``zoom * radius`` term always uses the
    true radii, so zooming far enough in still shows the real horizon, and
    real-size mode drops the floor entirely.
    """
    radii = np.asarray(radii, dtype=np.float64)
    true_px = zoom * radii
    if real_size:
        return np.maximum(true_px, 1.0)
    basis = radii if floor_radii is None else np.asarray(floor_radii,
                                                         dtype=np.float64)
    return np.maximum(true_px, _floor(basis, internal_height / REF_HEIGHT))


#: How far past its true radius a body's contact radius may be stretched so
#: that aiming at what is drawn actually works.  Uncapped, the Sun's drawn
#: disc is 0.6 AU across at 10 px/AU -- wider than Mercury's own orbit -- and
#: would swallow it the moment the player zoomed out.  At 8x the Sun's
#: contact radius is 0.037 AU, comfortably inside Mercury's 0.31 AU
#: perihelion, and Earth's is inside the Moon's orbit by about the same
#: margin.
MAX_AIM_FACTOR = 8.0


def contact_radii_au(radii, zoom, internal_height, real_size=False,
                     aim_radii=None):
    """Radii to collide against, in AU: what is drawn, capped so nothing
    stretches far enough to eat a neighbour's orbit.

    Collision *detection* is the only thing that ever reads this -- every
    outcome the physics computes (energy, fragment count, disruption
    threshold, ...) still uses the true radius. This is deliberately a
    render-layer computation handed to the physics as a plain array: the
    physics never learns what a camera is, and the render layer never learns
    what a collision is.

    ``aim_radii`` replaces the true radius **in the upper clip only**, so a
    black hole can be aimed at where its disc appears rather than at a target
    a ten-thousandth of a pixel wide. The lower bound stays the true radius:
    a contact radius may never fall below the real one.
    """
    radii = np.asarray(radii, dtype=np.float64)
    basis = radii if aim_radii is None else np.asarray(aim_radii,
                                                       dtype=np.float64)
    drawn_px = display_radii_px(radii, zoom, internal_height, real_size,
                                floor_radii=basis)
    drawn_au = drawn_px / max(zoom, 1e-30)
    return np.clip(drawn_au, radii, basis * MAX_AIM_FACTOR)
