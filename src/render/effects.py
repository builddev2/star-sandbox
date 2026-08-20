"""Impact visuals: turns a physics ``ImpactEvent`` into pixels.

**Wall-clock driven.**  Every particle ages on real seconds, never simulated
days, so an impact reads the same whether the sim is running at one day a
second or ten years a second -- a flash that took its cue from simulated
time would be over before it began at high time-scale, and would freeze the
frame for a simulated eternity at low time-scale.

**Anchored in world space, sized in screen space.**  A particle carries a
world position and drifts by a world velocity, so panning and following the
impact both work for free. How big it draws is computed fresh every frame
from its own true-scale radius through ``render/scale.display_radius_px`` --
the same floor-vs-zoom curve a body itself uses -- so it stays visible at any
zoom without a body-sized flash ever swallowing the system at wide zoom.

**Budgeted.**  A hard cap on live particles, oldest evicted first, so a
hundred-body pile-up cannot drop the frame rate.

Five elements, spawned together per impact and distinguished only by their
``kind``: a flash (an additive disc that ramps white -> yellow -> orange ->
red as it cools), one or two shock rings, a spray of ejecta sparks tinted by
the target's own material, a slower dimmer debris plume for energetic hits,
and, for a star, an arc of prominence plus a brief brightening.
"""

import math

import numpy as np
import pygame

from . import palette
from .scale import display_radius_px

#: Hard ceiling on live particles; new ones evict the oldest once it is hit.
MAX_PARTICLES = 400

#: Cooling colour ramp for anything hot: white -> yellow -> orange -> red.
_HEAT_RAMP = (
    (0.0, (255, 255, 255)),
    (0.25, (255, 236, 170)),
    (0.55, (255, 160, 60)),
    (1.0, (150, 40, 20)),
)


def _lerp(a, b, t):
    return a + (b - a) * t


def _heat_colour(t):
    """A point along the white -> yellow -> orange -> red cooling ramp."""
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    for k in range(len(_HEAT_RAMP) - 1):
        t0, c0 = _HEAT_RAMP[k]
        t1, c1 = _HEAT_RAMP[k + 1]
        if t <= t1:
            local = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            return tuple(int(_lerp(c0[i], c1[i], local)) for i in range(3))
    return _HEAT_RAMP[-1][1]


#: How many target-radii out a spark travels over its lifetime.  A visual
#: constant, deliberately independent of the impact's actual physics speed:
#: at ten years a second the true relative velocity would send a spark
#: across the screen in a fraction of a real frame, or -- for a gentle touch
#: -- barely move it at all.  Anchoring to the target's own drawn size instead
#: of the physics velocity is what keeps the wall-clock promise honest.
SPARK_TRAVEL_RADII = 5.0
PLUME_TRAVEL_RADII = 2.5

FLASH_LIFE_S = 0.55
RING_LIFE_S = 0.85
SPARK_LIFE_S = 1.1
PLUME_LIFE_S = 1.8
FLARE_LIFE_S = 1.3

#: Below this many "target radii" of peak size, an impact does not get a
#: plume -- it is not big enough for slow, dim debris to read as anything but
#: noise next to the flash and sparks.
PLUME_MIN_ENERGY = 1.0e-4


class _Particle:
    __slots__ = ("kind", "pos", "vel", "age", "life", "r0", "r1",
                "colour0", "colour1", "seed")

    def __init__(self, kind, pos, vel, life, r0, r1, colour0, colour1, seed=0.0):
        self.kind = kind
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.age = 0.0
        self.life = life
        self.r0 = r0
        self.r1 = r1
        self.colour0 = colour0
        self.colour1 = colour1
        self.seed = seed

    @property
    def t(self):
        return 1.0 if self.life <= 0.0 else min(1.0, self.age / self.life)

    @property
    def alive(self):
        return self.age < self.life


class EffectField:
    """Owns every live particle and the RNG that shapes new ones."""

    def __init__(self, seed=0):
        self.particles = []
        self.rng = np.random.default_rng(seed)

    @property
    def count(self):
        return len(self.particles)

    # --- spawning ----------------------------------------------------------

    def spawn(self, event):
        """Turn one physics ``ImpactEvent`` into particles."""
        scale_r = max(event.target_radius, event.projectile_radius, 1e-12)
        # log(1 + energy) so a pebble and a planet both produce a sane size:
        # a linear scale would make anything short of a planetary strike
        # invisible next to the ones that matter.
        magnitude = math.log1p(max(event.energy, 0.0) / max(scale_r, 1e-30))
        flash_peak = scale_r * (1.5 + 0.6 * magnitude)

        pos = event.pos
        tones = event.tones
        # event.vel is a physics quantity in AU/day; every particle velocity
        # below is instead a purpose-built AU-per-real-second visual speed,
        # exactly so that no particle's screen motion depends on how fast the
        # bodies happened to be moving in simulated time (see module
        # docstring). Only its *direction* is worth keeping, for the flare's
        # orientation.
        no_drift = np.zeros(2)

        self._add(_Particle("flash", pos, no_drift, FLASH_LIFE_S,
                            scale_r * 0.3, flash_peak, (255, 255, 255), tones[2]))
        self._add(_Particle("ring", pos, no_drift, RING_LIFE_S,
                            scale_r * 0.5, flash_peak * 2.2, tones[2], tones[1]))
        if magnitude > 0.3:
            self._add(_Particle("ring", pos, no_drift, RING_LIFE_S * 1.3,
                                scale_r * 0.5, flash_peak * 3.2, tones[1], tones[0]))

        n_sparks = int(np.clip(6 + 4.0 * magnitude, 6, 40))
        angles = self.rng.uniform(0.0, 2.0 * math.pi, n_sparks)
        speeds = self.rng.uniform(0.5, 1.3, n_sparks)
        for k in range(n_sparks):
            direction = np.array([math.cos(angles[k]), math.sin(angles[k])])
            travel = SPARK_TRAVEL_RADII * scale_r * speeds[k]
            spark_vel = direction * (travel / SPARK_LIFE_S)
            self._add(_Particle("spark", pos, spark_vel, SPARK_LIFE_S * speeds[k],
                                scale_r * 0.12, scale_r * 0.03,
                                (255, 250, 235), tones[1], seed=float(k)))

        if magnitude > PLUME_MIN_ENERGY:
            n_plume = int(np.clip(3 + 2.0 * magnitude, 3, 14))
            angles = self.rng.uniform(0.0, 2.0 * math.pi, n_plume)
            speeds = self.rng.uniform(0.4, 1.0, n_plume)
            for k in range(n_plume):
                direction = np.array([math.cos(angles[k]), math.sin(angles[k])])
                travel = PLUME_TRAVEL_RADII * scale_r * speeds[k]
                plume_vel = direction * (travel / PLUME_LIFE_S)
                self._add(_Particle("plume", pos, plume_vel, PLUME_LIFE_S,
                                    scale_r * 0.4, scale_r * 0.9,
                                    tones[1], tones[0]))

        if event.on_star:
            vx, vy = event.vel
            arc_angle = math.atan2(vy, vx) if math.hypot(vx, vy) > 0 else 0.0
            self._add(_Particle("flare", pos, no_drift, FLARE_LIFE_S,
                                scale_r * 1.2, scale_r * 3.0,
                                (255, 250, 220), palette.STAR_GLOW, seed=arc_angle))

    def _add(self, particle):
        self.particles.append(particle)
        if len(self.particles) > MAX_PARTICLES:
            del self.particles[: len(self.particles) - MAX_PARTICLES]

    # --- stepping ------------------------------------------------------------

    def update(self, dt_seconds):
        """Advance every particle by real seconds and drop the expired ones."""
        if not self.particles:
            return
        for p in self.particles:
            p.age += dt_seconds
            p.pos = p.pos + p.vel * dt_seconds
        self.particles = [p for p in self.particles if p.alive]

    # --- drawing ---------------------------------------------------------------

    def draw(self, surface, camera):
        if not self.particles:
            return
        height = surface.get_height()
        zoom = camera.zoom
        for p in self.particles:
            t = p.t
            radius_au = _lerp(p.r0, p.r1, t)
            px = display_radius_px(max(radius_au, 0.0), zoom, height)
            sx, sy = camera.world_to_screen(p.pos)
            sx, sy = int(sx), int(sy)
            if px < 0.5:
                continue

            if p.kind == "flash":
                fade = 1.0 - t
                colour = _heat_colour(t)
                colour = tuple(int(c * fade) for c in colour)
                _blit_disc(surface, colour, sx, sy, px)
            elif p.kind == "ring":
                fade = max(0.0, 1.0 - t)
                colour = tuple(int(_lerp(p.colour0[i], p.colour1[i], t) * fade)
                               for i in range(3))
                pygame.draw.circle(surface, colour, (sx, sy), max(1, int(px)), 1)
            elif p.kind == "spark":
                fade = max(0.0, 1.0 - t)
                colour = tuple(int(_lerp(p.colour0[i], p.colour1[i], t) * fade)
                               for i in range(3))
                if 0 <= sx < surface.get_width() and 0 <= sy < surface.get_height():
                    surface.set_at((sx, sy), colour)
            elif p.kind == "plume":
                fade = max(0.0, 1.0 - t) * 0.7
                colour = tuple(int(_lerp(p.colour0[i], p.colour1[i], t) * fade)
                               for i in range(3))
                pygame.draw.circle(surface, colour, (sx, sy), max(1, int(px)))
            elif p.kind == "flare":
                fade = max(0.0, 1.0 - t)
                colour = tuple(int(_lerp(p.colour0[i], p.colour1[i], t) * fade)
                               for i in range(3))
                arc_r = max(1, int(px))
                rect = pygame.Rect(sx - arc_r, sy - arc_r, arc_r * 2, arc_r * 2)
                span = 0.6
                pygame.draw.arc(surface, colour, rect,
                                p.seed - span, p.seed + span, max(1, arc_r // 6))


def _blit_disc(surface, colour, sx, sy, radius_px):
    """An additive filled circle, clamped to something pygame can draw."""
    r = max(1, min(int(radius_px), 4000))
    rect = pygame.Rect(sx - r, sy - r, r * 2, r * 2)
    if not surface.get_rect().colliderect(rect.inflate(4, 4)):
        return
    glow = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
    pygame.draw.circle(glow, (*colour, 255), (r, r), r)
    surface.blit(glow, (sx - r, sy - r), special_flags=pygame.BLEND_ADD)
