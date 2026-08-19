"""Per-body metadata and its orbit trail.

The numeric state (position, velocity, mass, radius) lives in the World's numpy arrays;
this holds everything that is not arithmetic.  One Body per live array row, kept
index-parallel by the World.
"""

import numpy as np

KIND_PLANET = "planet"
KIND_STAR = "star"
KIND_DEBRIS = "debris"

#: How many past positions a trail remembers.  Sampling is by angle swept (see Trail),
#: so this is about 1.7 orbits' worth for any body at any scale.
TRAIL_CAPACITY = 512

#: Append a trail point once the body has swept this many radians about its attractor.
#: Constant angle rather than constant time or distance is what makes Mercury's trail
#: and Neptune's equally smooth.
TRAIL_ANGLE_STEP = 0.021          # ~1.2 degrees -> ~300 points per orbit


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
                 "event_id", "no_collide_until", "seed", "rings")

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

    @property
    def is_star(self):
        return self.kind == KIND_STAR

    def __repr__(self):
        return "<Body %d %r %s>" % (self.id, self.name, self.kind)
