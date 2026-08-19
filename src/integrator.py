"""Gravity and time stepping.

**Velocity Verlet, not RK4.**  RK4 is more accurate per step but dissipative: over
thousands of orbits it visibly spirals planets inward, which is exactly the failure an
attentive child would notice.  Verlet is symplectic -- its energy error is bounded and
oscillatory rather than growing without limit -- and it costs one force evaluation per
step instead of four.

**Power-of-two block timesteps.**  A continuously varying dt destroys symplecticity and
smuggles the secular drift back in, so dt is restricted to ``DT_BASE / 2**k``.  ``k`` is
chosen once per frame from the tightest pair in the system and is allowed to move by at
most one notch per frame, which stops it chattering between two values.
"""

import numpy as np

from .units import G

#: Coarsest permitted step.  Comfortably resolves the outer planets on its own.
DT_BASE = 4.0
#: Finest is DT_BASE / 2**MAX_K, about 21 seconds.
MAX_K = 14
#: Fraction of the tightest pair's dynamical time to use as the target step.
#: 0.05 gives roughly 126 steps per orbit of that pair.
ETA = 0.05

#: Plummer softening, purely a singularity guard so an overlapping pair cannot produce
#: an infinite acceleration or a NaN.  At 1e-5 AU (1496 km) it sits well below every
#: collision radius, so collisions always fire long before it becomes physically
#: relevant: the force error at the Moon's distance is under 1e-5.
SOFTENING = 1e-5


class Integrator:
    """Owns the preallocated scratch buffers for the O(n^2) force kernel."""

    def __init__(self, capacity):
        self.capacity = capacity
        self._d = np.empty((capacity, capacity, 2), dtype=np.float64)
        self._r2 = np.empty((capacity, capacity), dtype=np.float64)
        self._inv = np.empty((capacity, capacity), dtype=np.float64)
        self._eps2 = SOFTENING * SOFTENING
        self.k = 3                      # current block level
        #: Set by ``choose_k``; the step actually in use.
        self.dt = DT_BASE / 2 ** 3

    # --- forces ----------------------------------------------------------------------

    def accelerations(self, pos, mass, n, out):
        """Fill ``out[:n]`` with the gravitational acceleration on each body."""
        if n == 0:
            return out
        d = self._d[:n, :n]
        r2 = self._r2[:n, :n]
        inv = self._inv[:n, :n]

        # d[i, j] = pos[j] - pos[i], the direction from i towards j.
        np.subtract(pos[None, :n, :], pos[:n, None, :], out=d)
        np.einsum("ijk,ijk->ij", d, d, out=r2)
        r2 += self._eps2
        # Self-interaction: inf distance -> zero contribution, no divide-by-zero.
        r2.flat[:: n + 1] = np.inf
        np.power(r2, -1.5, out=inv)
        inv *= mass[None, :n]
        np.einsum("ij,ijk->ik", inv, d, out=out[:n])
        out[:n] *= G
        return out

    def refresh(self, world):
        """Recompute the cached accelerations.  Call after any change to the world."""
        self.accelerations(world.pos, world.mass, world.n, world.acc)

    # --- timestep selection ----------------------------------------------------------

    def dynamical_time(self, world):
        """Shortest two-body dynamical time in the system, in days.

        For a circular pair this is the orbital period over 2*pi, so scaling it by ETA
        gives a step that resolves the fastest orbit present.  Computed from the pair
        separations, which is why a moon tightens the step automatically without anyone
        having to identify it as a moon.
        """
        n = world.n
        if n < 2:
            return DT_BASE
        pos = world.pos[:n]
        mass = world.mass[:n]
        d = pos[None, :, :] - pos[:, None, :]
        r2 = np.einsum("ijk,ijk->ij", d, d)
        np.fill_diagonal(r2, np.inf)
        pair_mass = mass[:, None] + mass[None, :]
        # t^2 = r^3 / (G * (m_i + m_j))
        t2 = np.sqrt(r2) ** 3 / (G * pair_mass)
        return float(np.sqrt(t2.min()))

    def choose_k(self, world, allow_jump=1):
        """Pick the block level for this frame, moving at most ``allow_jump`` notches."""
        target = ETA * self.dynamical_time(world)
        target = max(target, DT_BASE / 2 ** MAX_K)
        wanted = int(np.ceil(np.log2(DT_BASE / target))) if target < DT_BASE else 0
        wanted = int(np.clip(wanted, 0, MAX_K))
        if wanted > self.k:
            self.k = min(wanted, self.k + allow_jump)
        elif wanted < self.k:
            self.k = max(wanted, self.k - allow_jump)
        self.dt = DT_BASE / 2 ** self.k
        return self.k

    # --- stepping --------------------------------------------------------------------

    def step(self, world, dt):
        """One velocity Verlet step (kick-drift-kick)."""
        n = world.n
        if n == 0:
            world.time += dt
            return
        half = 0.5 * dt
        vel = world.vel[:n]
        pos = world.pos[:n]
        acc = world.acc[:n]

        vel += half * acc
        pos += dt * vel
        self.accelerations(world.pos, world.mass, n, world.acc)
        vel += half * world.acc[:n]
        world.time += dt
