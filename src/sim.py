"""The simulation driver: one object that owns the world, the integrator and
collision resolution, and advances all three under a wall-clock budget.

**The budget is not optional.**  At ten years per second the integrator needs
thousands of substeps per frame.  Without a cap the window stops responding and
the child reads it as a crash; with one the simulation quietly degrades to slow
motion and the HUD reports the shortfall.

Imports numpy and nothing from pygame -- ``tests/test_layering.py`` asserts it.
"""

import math
import time

import numpy as np

from . import orbits, scenarios
from .collisions import CollisionResolver
from .integrator import Integrator

#: Milliseconds of a 60fps frame we are willing to spend on physics.
DEFAULT_BUDGET_S = 0.012


class Simulation:
    def __init__(self, world=None, seed=1234):
        self.world = world if world is not None else scenarios.solar_system()
        self.integrator = Integrator(self.world.capacity)
        self.integrator.refresh(self.world)
        self.resolver = CollisionResolver(seed)
        self._prev = np.zeros_like(self.world.pos)
        self.paused = False
        self.last_ratio = 1.0
        self.events = 0
        self._seed = seed

    # --- lifecycle --------------------------------------------------------------

    def reset(self, scenario="solar_system", **kwargs):
        scenarios.REGISTRY[scenario](world=self.world, **kwargs)
        self.integrator = Integrator(self.world.capacity)
        self.integrator.refresh(self.world)
        self.resolver = CollisionResolver(self._seed)
        self.last_ratio = 1.0
        self.events = 0

    def refresh(self):
        """Recompute cached accelerations.  Call after any edit to mass or membership."""
        self.integrator.refresh(self.world)

    # --- stepping ---------------------------------------------------------------

    def advance(self, days, budget_s=DEFAULT_BUDGET_S):
        """Advance up to ``days``, stopping early if the wall-clock budget runs out.

        Returns the days actually simulated and records ``last_ratio`` so the HUD
        can tell the child the sim is running behind.
        """
        self.events = 0
        if self.paused or days <= 0.0:
            self.last_ratio = 0.0 if self.paused else 1.0
            return 0.0

        started = time.perf_counter()
        done = 0.0
        world = self.world
        while done < days:
            self.integrator.choose_k(world)
            dt = min(self.integrator.dt, days - done)
            self._prev[:world.n] = world.pos[:world.n]
            self.integrator.step(world, dt)
            self.events += self.resolver.resolve(world, self.integrator, self._prev)
            done += dt
            if time.perf_counter() - started >= budget_s:
                break

        self.last_ratio = done / days
        return done

    # --- trails -----------------------------------------------------------------

    def update_trails(self):
        """Sample every body's trail once per frame, never per substep."""
        world = self.world
        for i in range(world.n):
            attractor = orbits.dominant_attractor(world, i)
            if attractor is None:
                angle = None
            else:
                rel = world.pos[i] - world.pos[attractor]
                angle = math.atan2(float(rel[1]), float(rel[0]))
            world.bodies[i].trail.maybe_append(world.pos[i], angle)
