"""Per-body metadata and its orbit trail.

The numeric state (position, velocity, mass, radius) lives in the World's numpy arrays;
this holds everything that is not arithmetic.  One Body per live array row, kept
index-parallel by the World.
"""

import numpy as np

KIND_PLANET = "planet"
KIND_STAR = "star"
KIND_DEBRIS = "debris"
KIND_BLACK_HOLE = "black hole"
KIND_CRAFT = "craft"

#: How many past positions a trail remembers.  Sampling is by angle swept (see Trail),
#: so this is about 1.7 orbits' worth for any body at any scale.
TRAIL_CAPACITY = 512

#: Append a trail point once the body has swept this many radians about its attractor.
#: Constant angle rather than constant time or distance is what makes Mercury's trail
#: and Neptune's equally smooth.
TRAIL_ANGLE_STEP = 0.021          # ~1.2 degrees -> ~300 points per orbit

#: What a freshly made black hole looks like.  A hole with no spin and no disk
#: is a black disc and nothing else -- correct for a starved one, but a poor
#: first impression, so anything the child creates arrives already feeding.
DEFAULT_BLACK_HOLE_SPIN = 0.7
DEFAULT_BLACK_HOLE_DISK = 0.6


class Trail:
    """Ring buffer of past world positions."""

    __slots__ = ("points", "count", "head", "_last_angle")

    def __init__(self):
        self.points = np.zeros((TRAIL_CAPACITY, 2), dtype=np.float64)
        self.count = 0
        self.head = 0
        self._last_angle = None

    def clear(self):
        self.count = 0
        self.head = 0
        self._last_angle = None

    def maybe_append(self, pos, angle):
        """Record ``pos`` if the body has swept far enough since the last sample.

        ``angle`` is the body's angle about its dominant attractor, or None when it has
        no attractor -- in which case every call records.
        """
        if angle is not None and self._last_angle is not None:
            delta = abs((angle - self._last_angle + np.pi) % (2.0 * np.pi) - np.pi)
            if delta < TRAIL_ANGLE_STEP:
                return
        self._last_angle = angle
        self.points[self.head] = pos
        self.head = (self.head + 1) % TRAIL_CAPACITY
        self.count = min(self.count + 1, TRAIL_CAPACITY)

    def ordered(self):
        """The recorded points, oldest first."""
        if self.count < TRAIL_CAPACITY:
            return self.points[:self.count]
        return np.concatenate((self.points[self.head:], self.points[:self.head]))


class Body:
    """Identity and appearance of one simulated object."""

    __slots__ = ("id", "name", "kind", "material", "trail", "born_day",
                 "event_id", "no_collide_until", "seed", "rings", "spin", "disk",
                 "look", "locked_fields", "plan", "fuel_dv")

    def __init__(self, body_id, name, material, kind=KIND_PLANET, born_day=0.0, seed=0):
        self.id = body_id
        self.name = name
        self.kind = kind
        self.material = material
        self.trail = Trail()
        self.born_day = born_day
        #: Fragments born of the same shattering share an event id and ignore each other
        #: until ``no_collide_until``; without this they re-merge on the next substep and
        #: the planet appears to flicker and reassemble.
        self.event_id = 0
        self.no_collide_until = -1.0
        self.seed = seed
        #: Drawn with a ring system.  Set by scenarios for Saturn.
        self.rings = False
        #: Dimensionless spin, 0 .. units.MAX_SPIN.  Black holes only; it moves
        #: the accretion disk's inner edge and turns the jets on.
        self.spin = 0.0
        #: Feeding rate, 0 .. 1, driving how bright and how wide the accretion
        #: disk is drawn.  0 is a starved hole: shadow and photon ring only.
        self.disk = 0.0
        #: How this body wants to be drawn -- "terrestrial", "mars", ... --
        #: or None to draw from its material.  Pinned by the scenario on the
        #: named planets; ``edits.set_material`` drops it, because the
        #: material is then what the body is made of.  Appearance only, on
        #: the Body beside ``rings`` and ``spin`` because the force kernel
        #: never reads it.  Nothing inherits it: a fragment, a remnant made
        #: by ``world.add`` and a thrown comet all arrive with None.  A body
        #: that *survives* an accretion keeps its own ``Body`` and so keeps
        #: its look -- Earth swallowing a comet is still Earth.
        self.look = None
        #: Fields the child has pinned, named the way ``edits.edit_fields``
        #: names them.  On the Body rather than in World's arrays for the same
        #: reason ``spin`` and ``disk`` are: the O(n^2) force kernel never reads
        #: it.  A frozenset because rebinding is then the only way to change it,
        #: so two bodies can never end up sharing one mutable set -- and nothing
        #: inherits it: a fresh body, a collision fragment and a tidal fragment
        #: all arrive with nothing locked.
        self.locked_fields = frozenset()
        #: Maneuvers this craft will execute, soonest first.  A tuple for the
        #: same reason ``locked_fields`` is a frozenset: rebinding is then the
        #: only way to change it, so two bodies can never share one mutable
        #: list.  On the Body rather than in World's arrays because the O(n^2)
        #: force kernel never reads it.
        self.plan = ()
        #: Remaining delta-v budget, AU/day.  Zero for everything that is not
        #: a craft, and for a craft nobody has fuelled.
        self.fuel_dv = 0.0

    @property
    def is_star(self):
        return self.kind == KIND_STAR

    @property
    def is_black_hole(self):
        return self.kind == KIND_BLACK_HOLE

    @property
    def is_craft(self):
        return self.kind == KIND_CRAFT

    def __repr__(self):
        return "<Body %d %r %s>" % (self.id, self.name, self.kind)
