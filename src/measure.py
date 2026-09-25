"""Measuring things: the ruler between two bodies, and the orbit timer.

Numbers a child can check against a book.  The ruler is plain geometry.  The
timer counts the angle a body sweeps about what it orbits, and when it has gone
all the way round it has *measured* an orbital period -- which can then be set
beside what Kepler's third law predicts from the orbit's size, and turned round
to weigh what it orbits (``a^3 / P^2`` in AU, years and suns).

**Sampled once a frame**, like the trails.  It measures by angle swept rather
than by time, so it is right at any speed -- up to the point where one sample
sweeps more than half a turn, which angle-unwrapping cannot resolve.  A body's
angular rate is not constant: near periapsis it runs at
``n * sqrt(1+e) / (1-e)**1.5``, far above its average, so a limit stated as a
flat fraction of the mean period is only safe for a circle.  The guard in
``sample`` shrinks by that same factor as the orbit gets eccentric, capping the
worst-case sweep *at periapsis* to a quarter turn rather than the average
sweep.  Past the limit, which way the body went between two samples is
ambiguous, and the timer says "too fast" and starts again rather than report a
number it cannot stand behind.  Measured: 200-day frames on a one-year, circular
orbit reported a 600-day "lap" before the guard existed; without the
eccentricity scaling, an e=0.5 orbit let an 80-day gap straddling periapsis
through and reported a 730-day "lap".

**Holds ids, never rows**: ``World.remove`` is a swap-remove and a timer lives
for hundreds of frames.

Imports numpy and nothing from pygame -- ``tests/test_layering.py`` asserts it.
"""

import math

from . import orbits, units

TWO_PI = 2.0 * math.pi
#: The longest gap between two samples, as a fraction of the *circular*
#: (mean-motion) period, that the timer will trust before scaling for
#: eccentricity.  ``sample`` multiplies this by ``(1-e)**1.5 / sqrt(1+e)`` --
#: the ratio of a circle's angular rate to the true rate at periapsis -- so
#: the actual limit shrinks as the orbit gets eccentric.  Unwrapping is exact
#: below half a turn; a flat quarter left unscaled is generous enough for a
#: circle but lets an eccentric orbit sweep past it unnoticed near periapsis.
MAX_SAMPLE_FRACTION = 0.25


def separation(world, i, j):
    """How far apart two bodies are, and how fast that is changing.

    ``centres`` (AU, centre to centre), ``gap`` (AU, surface to surface, never
    below zero), ``light_days`` (light's travel time over ``centres``),
    ``closing`` (AU/day, positive while they approach) and ``relative``
    (AU/day, the whole relative speed).
    """
    rel = world.pos[j] - world.pos[i]
    dv = world.vel[j] - world.vel[i]
    rx, ry = float(rel[0]), float(rel[1])
    vx, vy = float(dv[0]), float(dv[1])
    centres = math.hypot(rx, ry)
    closing = -(rx * vx + ry * vy) / centres if centres > 0.0 else 0.0
    return {
        "centres": centres,
        "gap": max(0.0, centres - float(world.radius[i]) - float(world.radius[j])),
        "light_days": centres / units.C,
        "closing": closing,
        "relative": math.hypot(vx, vy),
    }


def _elements(world, index, attractor):
    """The osculating-elements dict (`orbits.elements_from_state`) for
    ``index`` relative to ``attractor`` -- the one place `kepler_period` and
    `PeriodTimer.sample` compute ``mu`` and the relative state, so a sample
    that needs both the period and the eccentricity asks once.
    """
    mu = units.G * float(world.mass[attractor] + world.mass[index])
    return orbits.elements_from_state(mu, world.pos[index] - world.pos[attractor],
                                      world.vel[index] - world.vel[attractor])


def kepler_period(world, index, attractor):
    """``(period, a)``: what Kepler's third law says, in days, from the
    osculating semi-major axis in AU.  ``period`` is None for an unbound orbit.
    """
    el = _elements(world, index, attractor)
    return el["period"], el["a"]


def _angle(world, index, attractor):
    rel = world.pos[index] - world.pos[attractor]
    return math.atan2(float(rel[1]), float(rel[0]))


class PeriodTimer:
    """Times one body round whatever it orbits."""

    def __init__(self, world, index):
        attractor = orbits.dominant_attractor(world, index)
        self.body_id = world.bodies[index].id
        self.attractor_id = (world.bodies[attractor].id
                             if attractor is not None else None)
        #: "timing", "too fast", "not in orbit", "lost" or "moved on".
        self.status = "timing"
        self._restart(world, index, attractor)

    def _restart(self, world, index, attractor):
        #: World days at which each lap finished; ``laps[0]`` is the start.
        #: Interpolated between samples, never rounded to one.
        self.laps = [world.time]
        #: Signed radians swept since the start, so a retrograde orbit counts.
        self.swept = 0.0
        self._last_time = world.time
        if attractor is None or kepler_period(world, index, attractor)[0] is None:
            self.status = "not in orbit"
            return
        self._last_angle = _angle(world, index, attractor)

    def _rows(self, world):
        index = world.index_of(self.body_id)
        attractor = (world.index_of(self.attractor_id)
                     if self.attractor_id is not None else None)
        return index, attractor

    def sample(self, world):
        """Catch up to ``world.time``.  Once a frame, never per substep."""
        if self.status in ("lost", "moved on", "not in orbit"):
            return
        index, attractor = self._rows(world)
        if index is None or attractor is None:
            self.status = "lost"
            return
        if orbits.dominant_attractor(world, index) != attractor:
            self.status = "moved on"
            return
        el = _elements(world, index, attractor)
        period, e = el["period"], el["e"]
        if period is None:
            self.status = "not in orbit"
            return
        gap = world.time - self._last_time
        if gap <= 0.0:
            return                              # paused
        # Angular rate at periapsis exceeds the mean rate by
        # sqrt(1+e) / (1-e)**1.5, so the trusted gap shrinks by its inverse.
        limit = MAX_SAMPLE_FRACTION * period * (1.0 - e) ** 1.5 / math.sqrt(1.0 + e)
        if gap > limit:
            self._restart(world, index, attractor)
            self.status = "too fast"
            return
        self.status = "timing"
        angle = _angle(world, index, attractor)
        step = (angle - self._last_angle + math.pi) % TWO_PI - math.pi
        before = abs(self.swept)
        self.swept += step
        after = abs(self.swept)
        done = len(self.laps) - 1
        while after > before and after >= (done + 1) * TWO_PI:
            fraction = ((done + 1) * TWO_PI - before) / (after - before)
            self.laps.append(self._last_time + fraction * gap)
            done += 1
        self._last_angle = angle
        self._last_time = world.time

    @property
    def measured(self):
        """The most recent full lap, in days, or None before the first."""
        if len(self.laps) < 2:
            return None
        return self.laps[-1] - self.laps[-2]

    @property
    def fraction_round(self):
        """How far round the current lap is, 0..1."""
        return (abs(self.swept) % TWO_PI) / TWO_PI

    def mass_from_orbit(self, world):
        """Kepler's third law turned round: ``a^3 / P^2`` in AU and years is the
        mass of the pair in suns.  None before the first lap, or once the body
        or what it orbits is gone.
        """
        period = self.measured
        index, attractor = self._rows(world)
        if period is None or index is None or attractor is None:
            return None
        _, a = kepler_period(world, index, attractor)
        return a ** 3 / units.to_years(period) ** 2
