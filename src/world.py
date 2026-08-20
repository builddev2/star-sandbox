"""The simulated universe: structure-of-arrays state plus per-body metadata.

Numeric state is held in preallocated numpy arrays so the O(n^2) force kernel can run
without touching Python objects.  A parallel list of Body objects holds everything else,
kept index-parallel.

**Removal is a swap-remove**, which keeps the arrays contiguous but means indices are not
stable.  Every reference held across a frame -- selection, camera follow, the trajectory
predictor -- must therefore store a body *id* and resolve it through ``index_of``.  This
is the one discipline rule that prevents a whole class of bugs when a merge happens
mid-frame.

This module imports numpy and nothing from pygame; selftest asserts that.
"""

import numpy as np

from . import units
from .body import Body, KIND_PLANET

#: Beyond this the O(n^2) kernel cannot keep 60fps with enough substeps to stay accurate.
MAX_BODIES = 200


class World:
    def __init__(self, capacity=MAX_BODIES):
        self.capacity = capacity
        self.pos = np.zeros((capacity, 2), dtype=np.float64)
        self.vel = np.zeros((capacity, 2), dtype=np.float64)
        self.acc = np.zeros((capacity, 2), dtype=np.float64)
        self.mass = np.zeros(capacity, dtype=np.float64)
        self.radius = np.zeros(capacity, dtype=np.float64)
        #: The radius collisions actually test against.  Equal to ``radius``
        #: unless the render layer has stretched it to match what is drawn --
        #: see ``render/scale.contact_radii_au``. Physics that decides an
        #: *outcome* (energy, fragment count, ...) always uses the true
        #: ``radius``; only "did these two touch" uses this one.
        self.contact_radius = np.zeros(capacity, dtype=np.float64)
        self.n = 0

        self.bodies = []
        self._by_id = {}
        self._next_id = 1
        self.time = 0.0                 # days since the scenario started

    # --- membership ------------------------------------------------------------------

    def add(self, name, material, mass, radius, pos, vel,
            kind=KIND_PLANET, seed=0):
        """Append a body.  Returns the Body, or None if the world is full."""
        if self.n >= self.capacity:
            return None
        i = self.n
        self.pos[i] = pos
        self.vel[i] = vel
        self.acc[i] = 0.0
        self.mass[i] = mass
        self.radius[i] = radius
        self.contact_radius[i] = radius
        self.n += 1

        body = Body(self._next_id, name, material, kind, born_day=self.time, seed=seed)
        self._next_id += 1
        self.bodies.append(body)
        self._by_id[body.id] = i
        return body

    def remove(self, index):
        """Swap-remove the body at ``index``, keeping the arrays contiguous."""
        last = self.n - 1
        dead = self.bodies[index]
        if index != last:
            for arr in (self.pos, self.vel, self.acc):
                arr[index] = arr[last]
            self.mass[index] = self.mass[last]
            self.radius[index] = self.radius[last]
            self.contact_radius[index] = self.contact_radius[last]
            moved = self.bodies[last]
            self.bodies[index] = moved
            self._by_id[moved.id] = index
        self.bodies.pop()
        del self._by_id[dead.id]
        self.n -= 1

    def remove_many(self, indices):
        """Remove several bodies.  Descending order keeps the swap-remove valid."""
        for i in sorted(set(indices), reverse=True):
            self.remove(i)

    def index_of(self, body_id):
        """Current array row for a body id, or None if it no longer exists."""
        return self._by_id.get(body_id)

    def body_by_id(self, body_id):
        i = self._by_id.get(body_id)
        return None if i is None else self.bodies[i]

    def clear(self):
        self.n = 0
        self.bodies.clear()
        self._by_id.clear()
        self.time = 0.0

    # --- views -----------------------------------------------------------------------

    @property
    def positions(self):
        return self.pos[:self.n]

    @property
    def velocities(self):
        return self.vel[:self.n]

    @property
    def masses(self):
        return self.mass[:self.n]

    @property
    def radii(self):
        return self.radius[:self.n]

    @property
    def contact_radii(self):
        return self.contact_radius[:self.n]

    # --- diagnostics -----------------------------------------------------------------

    def total_momentum(self):
        n = self.n
        return (self.mass[:n, None] * self.vel[:n]).sum(axis=0)

    def center_of_mass(self):
        n = self.n
        total = self.mass[:n].sum()
        if total <= 0.0:
            return np.zeros(2)
        return (self.mass[:n, None] * self.pos[:n]).sum(axis=0) / total

    def kinetic_energy(self):
        n = self.n
        v2 = np.einsum("ij,ij->i", self.vel[:n], self.vel[:n])
        return 0.5 * float((self.mass[:n] * v2).sum())

    def potential_energy(self):
        n = self.n
        if n < 2:
            return 0.0
        d = self.pos[:n, None, :] - self.pos[None, :n, :]
        r = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
        np.fill_diagonal(r, np.inf)
        mm = self.mass[:n, None] * self.mass[None, :n]
        # Each pair counted twice, hence the halving.
        return -0.5 * units.G * float((mm / r).sum())

    def total_energy(self):
        return self.kinetic_energy() + self.potential_energy()

    def zero_net_momentum(self):
        """Shift into the barycentric frame so the whole system does not drift."""
        n = self.n
        total = self.mass[:n].sum()
        if total <= 0.0:
            return
        self.vel[:n] -= self.total_momentum() / total

    def clamp_contact_radii(self, candidate):
        """Clip candidate contact radii so none reaches halfway to whatever
        body is nearest, however far the render layer's own zoom-only cap
        stretched them.

        A planet and its innermost moon can sit close enough that even eight
        times the planet's true radius (render/scale.MAX_AIM_FACTOR) reaches
        past the halfway point at wide-enough zoom -- Jupiter and Io are
        0.0028 AU apart, and at the camera's widest zoom Jupiter's own drawn
        floor alone stretches its contact radius past half of that. The
        render layer cannot know this: it never sees another body's
        position. This is the layer that does, so this is where the second,
        geometry-aware half of the cap lives.
        """
        n = self.n
        candidate = np.asarray(candidate, dtype=np.float64)
        if n < 2:
            return candidate
        pos = self.pos[:n]
        d = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
        np.fill_diagonal(dist, np.inf)
        nearest = dist.min(axis=1)
        return np.maximum(np.minimum(candidate, 0.5 * nearest), self.radius[:n])
