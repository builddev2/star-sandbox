"""What an edit did, and why.

The sandbox is honest and silent: double the Sun's mass and Earth's year drops
from 365 days to 138, and the inspector shows the new number, destroys the old
one, and offers no reason.  This module is the memory that makes the comparison
possible and the vocabulary that explains it.

**It returns numbers and format tags, never text with numbers baked into it.**
``src`` may not import pygame (``tests/test_layering.py`` runs a subprocess that
asserts it), and the ``hud.format_*`` functions live on the far side of that
line.  So a ``Row`` carries ``before``, ``after`` and a tag naming which
formatter to use, and the render layer resolves it -- the same arrangement
``ImpactEvent`` and ``render/effects.py`` already have.

**Comparison is by orbital element, not by position.**  The world keeps running
between the baseline and the report -- at one month per second a 350 ms burst is
ten days of simulated time -- and elements are invariant under motion along the
orbit.  Comparing positions or speeds would report a planet's own progress round
its orbit as though the edit had caused it.

Snapshots are keyed by ``body.id`` and never by index: ``World.remove`` is a
swap-remove, and a snapshot outlives at least one frame.
"""

from dataclasses import dataclass

from . import orbits, thermal, units


@dataclass(frozen=True)
class BodyState:
    """One body, frozen at a moment, in the quantities an edit can move."""

    body_id: int
    name: str
    kind: str
    #: Display name of the substance ("Ice"), for the action line.
    material: str
    #: Lookup key ("rock_ice"), for the water-consequence test.
    material_key: str
    #: Where this substance boils, kelvin.
    vapour_k: float
    mass: float
    radius: float
    density: float
    #: Black-hole only, and zero for everything else.  Here rather than left
    #: out because `set_spin` and `set_disk` move nothing else in this record:
    #: without them a spin edit would diff to no rows at all and be silently
    #: unexplainable.
    spin: float
    disk: float
    #: What the body is at this instant -- what the inspector shows.
    temperature: float
    #: Where it is *heading*, or the pinned value for a star or a hole.  Every
    #: comparison uses this one: the instantaneous temperature relaxes on its
    #: own clock (a skin-depth time constant under an hour for anything warm),
    #: so it drifts during a burst whether or not the edit touched it.
    eq_temperature: float
    attractor: str
    attractor_id: int
    a: float
    e: float
    #: Days.  Zero when unbound -- elements_from_state returns None there, and
    #: a Row holds floats.  ``bound`` is what distinguishes the two cases.
    period: float
    periapsis: float
    bound: bool


@dataclass(frozen=True)
class Snapshot:
    #: body id -> BodyState
    bodies: dict


def snapshot(world):
    """Freeze every body in the quantities an edit can move.

    O(n^2), through ``thermal.equilibrium_temperatures`` and one
    ``dominant_attractor`` scan per body.  That is affordable because this runs
    once per *burst* -- once when a held stepper starts and once when it stops --
    and never per frame.
    """
    equilibrium = thermal.equilibrium_temperatures(world)
    bodies = {}
    for i in range(world.n):
        body = world.bodies[i]
        mass = float(world.mass[i])
        radius = float(world.radius[i])
        density = (units.density_from_mass_radius(mass, radius)
                   if radius > 0.0 else 0.0)

        # A radiator's temperature is set by its own physics, not by what falls
        # on it, so equilibrium_temperatures is the wrong number for a star or
        # a hole.  pinned_temperature is the one function that knows which.
        pinned = thermal.pinned_temperature(world, i)
        eq = float(equilibrium[i]) if pinned is None else float(pinned)

        attractor_index = orbits.dominant_attractor(world, i)
        if attractor_index is None:
            attractor, attractor_id = "", 0
            a = e = period = periapsis = 0.0
            bound = False
        else:
            attractor_body = world.bodies[attractor_index]
            attractor, attractor_id = attractor_body.name, attractor_body.id
            mu = units.G * (float(world.mass[attractor_index]) + mass)
            el = orbits.elements_from_state(
                mu,
                world.pos[i] - world.pos[attractor_index],
                world.vel[i] - world.vel[attractor_index])
            a, e = float(el["a"]), float(el["e"])
            bound = bool(el["bound"])
            period = float(el["period"]) if el["period"] is not None else 0.0
            periapsis = float(el["periapsis"])

        bodies[body.id] = BodyState(
            body_id=body.id, name=body.name, kind=body.kind,
            material=body.material.name, material_key=body.material.key,
            vapour_k=body.material.vapour_temperature_k,
            mass=mass, radius=radius, density=density,
            spin=float(body.spin), disk=float(body.disk),
            temperature=float(world.temperature[i]), eq_temperature=eq,
            attractor=attractor, attractor_id=attractor_id,
            a=a, e=e, period=period, periapsis=periapsis, bound=bound)
    return Snapshot(bodies=bodies)
