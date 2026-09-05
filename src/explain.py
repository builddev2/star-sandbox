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


# --- the report ---------------------------------------------------------------------

#: Relative change below which a row is noise rather than news.  One percent is
#: comfortably above float drift and comfortably below anything a scrub notch
#: does: a single mass notch is 6%.
NOISE = 0.01

#: How many rippled bodies a report names.  Three is what fits the panel at the
#: window floor and what a child will actually read.
MAX_RIPPLE = 3


@dataclass(frozen=True)
class Row:
    """One quantity, before and after.

    ``format`` names the ``hud.format_*`` function the render layer must use --
    ``"mass"``, ``"length"``, ``"period"``, ``"temperature"``, ``"density"`` or
    ``"plain"``.  It is a tag rather than a formatted string because this
    module may not import pygame, and the formatters live beyond that line.
    """

    label: str
    before: float
    after: float
    format: str


@dataclass(frozen=True)
class Change:
    name: str
    is_edited: bool
    rows: tuple


@dataclass(frozen=True)
class Report:
    #: "You doubled the Sun's mass."
    action: str
    #: The edited body first, then the ripple, most-moved first.
    changes: tuple
    #: Plain sentences.  Filled in by ``_why`` (Task 3).
    why: tuple = ()
    #: (general, substituted) pairs.  Filled in by ``_maths`` (Task 4).
    maths: tuple = ()


#: Every quantity a report can show, in the order it is shown.  One list for
#: every body, filtered down to whatever actually moved -- so the edited body
#: naturally shows Mass and Radius while a rippled planet naturally shows One
#: lap and Stretch, without either case being written out separately.
ROW_SPECS = (
    ("mass", "Mass", "mass"),
    ("radius", "Radius", "length"),
    ("density", "Density", "density"),
    ("eq_temperature", "Temp", "temperature"),
    # A hole's two dials.  set_spin and set_disk move nothing else in the
    # record, so without these a spin edit diffs to nothing and the panel
    # stays silent on the one body it has the most to say about.
    ("spin", "Spin", "plain"),
    ("disk", "Feeding", "plain"),
    ("period", "One lap", "period"),
    ("e", "Stretch", "plain"),
    ("periapsis", "Closest", "length"),
)


def _relative(before, after):
    """Change as a fraction of the larger of the two.  Zero when both are zero."""
    scale = max(abs(before), abs(after))
    return 0.0 if scale == 0.0 else abs(after - before) / scale


def _rows_for(before_state, after_state, noise):
    """Every row that moved by more than ``noise``, in ROW_SPECS order."""
    rows = []
    for attribute, label, form in ROW_SPECS:
        was = getattr(before_state, attribute)
        now = getattr(after_state, attribute)
        if _relative(was, now) > noise:
            rows.append(Row(label=label, before=was, after=now, format=form))
    return tuple(rows)


def _change_for(before, after, body_id, noise, is_edited):
    if body_id not in before.bodies or body_id not in after.bodies:
        return None
    rows = _rows_for(before.bodies[body_id], after.bodies[body_id], noise)
    if not rows:
        return None
    return Change(name=after.bodies[body_id].name, is_edited=is_edited, rows=rows)


def _score(change):
    """How much this body moved, for ranking one body against another.

    The most significant row present, in ``ROW_SPECS`` order -- not the
    largest fraction among a body's own rows.  ``rows`` is already in that
    order, because ``_rows_for`` built it by walking ``ROW_SPECS``, so the
    first entry *is* the most significant one.  Taking the max instead was
    the first thing tried, and it ranked Venus and Neptune above Mercury when
    the Sun's mass was doubled: both start on the two most nearly circular
    orbits in the solar system, so their eccentricity rows -- which have no
    natural scale, only the interval ``[0, 1)`` -- register a huge fraction
    of *themselves* while telling you nothing about how hard either planet
    was hit. Mass, radius, density and temperature carry real units and a
    real zero, so a fraction of one of those means the same thing on every
    body; a fraction of eccentricity or period does not, because a planet
    that started eccentric (Mercury) has less room left to move in than one
    that started circular, regardless of how strongly it was disturbed.
    Ranking by the first row lets the universal quantities settle it whenever
    one is present -- here, every ripple only ever shows Temp first, and its
    fraction is (to the ninth decimal place) the same +37% for every planet,
    because it is set by the Sun's own new temperature alone; what breaks the
    tie, correctly, in Mercury's favour is the CMB floor in the equilibrium
    formula, which forgives the coldest, most distant planets a little more
    of that rise than it forgives Mercury.  Only when nothing ahead of
    eccentricity or period moved does the ranking fall back to them.
    """
    row = change.rows[0]
    return _relative(row.before, row.after)


def diff(before, after, edited_id, field, max_ripple=MAX_RIPPLE, noise=NOISE):
    """What changed between two snapshots, ranked.  ``None`` if nothing did.

    ``field`` is the edited property -- one of ``edits.edit_fields``' names, or
    the pseudo-field ``"collapse"``.  It picks the wording, not the rows.

    The edited body always leads, even when the ripple moved further, because
    the report answers "what did *that* do".  Everything else is ranked by how
    much it moved and cut to ``max_ripple``: the ripple is the part he did not
    do himself, and it is what makes doubling the Sun a lesson rather than a
    fact he already knew.
    """
    if edited_id not in before.bodies or edited_id not in after.bodies:
        # Deleted, or merged away, between the baseline and the settle.  Saying
        # nothing beats reporting on a body that is gone.
        return None

    edited = _change_for(before, after, edited_id, noise, is_edited=True)

    ripple = []
    for body_id in after.bodies:
        if body_id == edited_id:
            continue
        change = _change_for(before, after, body_id, noise, is_edited=False)
        if change is not None:
            ripple.append(change)
    # Stable sort, so bodies that moved by the same amount keep world order.
    ripple.sort(key=_score, reverse=True)
    ripple = ripple[:max_ripple]

    if edited is None and not ripple:
        return None

    changes = tuple(([edited] if edited is not None else []) + ripple)
    return Report(action=_action(before, after, edited_id, field),
                  changes=changes)


def _action(before, after, edited_id, field):
    """The opening line: what he did, in his own terms."""
    was, now = before.bodies[edited_id], after.bodies[edited_id]
    name = now.name

    if field == "collapse":
        return "You collapsed %s into a black hole." % name
    if field == "material":
        return "You made %s out of %s." % (name, now.material.lower())
    if field == "mass":
        return "You %s." % _scaled(name, "mass", was.mass, now.mass)
    if field == "density":
        return "You made %s %s." % (
            name, "denser" if now.density > was.density else "less dense")
    if field == "temp":
        return "You %s %s." % (
            "heated" if now.eq_temperature > was.eq_temperature else "cooled", name)
    if field == "spin":
        return "You spun %s %s." % (
            name, "faster" if now.spin >= was.spin else "slower")
    if field == "disk":
        return "You changed how much %s is feeding on." % name
    return "You changed %s's %s." % (name, field)


def _scaled(name, quantity, was, now):
    """"doubled the Sun's mass" / "made Jupiter 12x heavier"."""
    if was <= 0.0:
        return "changed %s's %s" % (name, quantity)
    ratio = now / was
    if abs(ratio - 2.0) < 0.05:
        return "doubled %s's %s" % (name, quantity)
    if abs(ratio - 0.5) < 0.02:
        return "halved %s's %s" % (name, quantity)
    if ratio > 1.0:
        return "made %s %.3gx heavier" % (name, ratio)
    return "made %s %.3gx lighter" % (name, 1.0 / ratio)
