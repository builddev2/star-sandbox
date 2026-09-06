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
from .body import KIND_BLACK_HOLE


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
    #: The body this is about.  An id, not a name: collision fragments are
    #: minted as "Ejecta" and "Debris" (collisions.py), so several bodies in
    #: one world routinely share a name and a name cannot identify one.
    body_id: int
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
    # Source resolved per body by _temperature_attribute: the edited body's own
    # instantaneous temperature, everyone else's equilibrium.
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


def _temperature_attribute(is_edited):
    """Which temperature a Temp row compares.

    The edited body's own instantaneous ``temperature``, because that is the
    number he just set -- and because a temperature edit moves the body's
    equilibrium not at all: nothing heats itself, so heating Earth to 500 K
    leaves Earth's equilibrium bit-identical, and a report built on it would
    diff to nothing.  The panel would then stay silent on the one edit it has
    the most to say about.

    Everyone else's ``eq_temperature``, because their instantaneous values
    relax on their own clock and would otherwise report the thermal
    simulation's own progress during the burst as though the edit had caused
    it.
    """
    return "temperature" if is_edited else "eq_temperature"


def _rows_for(before_state, after_state, noise, is_edited):
    """Every row that moved by more than ``noise``, in ROW_SPECS order."""
    rows = []
    for attribute, label, form in ROW_SPECS:
        if attribute == "eq_temperature":
            attribute = _temperature_attribute(is_edited)
        was = getattr(before_state, attribute)
        now = getattr(after_state, attribute)
        if _relative(was, now) > noise:
            rows.append(Row(label=label, before=was, after=now, format=form))
    return tuple(rows)


def _change_for(before, after, body_id, noise, is_edited):
    if body_id not in before.bodies or body_id not in after.bodies:
        return None
    rows = _rows_for(before.bodies[body_id], after.bodies[body_id], noise, is_edited)
    if not rows:
        return None
    return Change(body_id=body_id, name=after.bodies[body_id].name,
                  is_edited=is_edited, rows=rows)


def _score(change):
    """How much this body moved: its largest single row change."""
    return max(_relative(row.before, row.after) for row in change.rows)


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
    edited_state = after.bodies[edited_id]
    return Report(action=_action(before, after, edited_id, field),
                  changes=changes,
                  why=why_lines(before, after, edited_id, field, changes,
                                material_key=edited_state.material_key,
                                vapour_k=edited_state.vapour_k),
                  maths=formulas(before, after, edited_id, field, changes))


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
            "heated" if now.temperature > was.temperature else "cooled", name)
    if field == "spin":
        return "You spun %s %s." % (
            name, "faster" if now.spin >= was.spin else "slower")
    if field == "disk":
        return "You changed how much %s is feeding on." % name
    return "You changed %s's %s." % (name, field)


def _plain_number(value):
    """A number with no exponent in it, wherever %g would reach for one.

    ``render/hud.py`` refuses exponent notation on the grounds that
    "5.823e+04" is unreadable to this audience, and a formula line is read by
    the same person.  %.3g gives up at 999.9, which a black hole reaches by
    swallowing a few heavy stars -- it accretes without an upper clamp, since
    MAX_MASS bounds the mass *stepper* and not a merge.

    Below a thousandth of a solar mass an exponent genuinely is the readable
    form: the alternative is sixteen leading zeros, and nothing in solar
    masses reads well down there.  That end is left as it is, knowingly.
    """
    if value >= 1000.0:
        return "{:,.0f}".format(value)
    if value >= 0.001:
        return "%.6g" % value
    return "%.3g" % value


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
        return "made %s %sx heavier" % (name, _plain_number(ratio))
    return "made %s %sx lighter" % (name, _plain_number(1.0 / ratio))


# --- why ----------------------------------------------------------------------------
#
# A table rather than a chain of ifs, so adding a case is a row.  Keyed on
# (field, direction, role):
#
#   direction  "up" or "down", by where the edited quantity went.
#   role       "attractor" when at least one rippled body orbits the edited one,
#              "self" otherwise.  A mixed ripple counts as "attractor", because
#              that is the case carrying the lesson.
#
# ``{name}`` is the edited body.  Lookup falls back from the most specific key
# to the least, and ``test_no_edit_falls_through_to_silence`` asserts that no
# combination reaches the end without a line.

WHY = {
    ("mass", "up", "attractor"):
        "A heavier {name} pulls harder, so everything around it falls inward "
        "and speeds up.  Round orbits become long ovals.",
    ("mass", "down", "attractor"):
        "A lighter {name} pulls more weakly, so everything around it swings "
        "outward and slows down.",
    ("mass", "up", "self"):
        "{name} got heavier.  Its density did not change, so it grew.",
    ("mass", "down", "self"):
        "{name} got lighter, so it shrank.",
    ("density", "up", "self"):
        "The same amount of stuff, packed tighter, so {name} shrank.",
    ("density", "down", "self"):
        "The same amount of stuff, spread thinner, so {name} swelled.",
    ("material", "", "self"):
        "{name} is made of something else now, and the new substance has a "
        "density of its own -- so the same mass takes up a different amount "
        "of room.",
    ("temp", "", "self"):
        "Heat is not a dial.  You set it, but nothing pins it there.",
    ("spin", "", "self"):
        "A faster-spinning hole drags space round with it.  Matter can orbit "
        "closer in before it must fall, and more of what falls comes back out "
        "as light.",
    ("disk", "", "self"):
        "This is how much matter is falling in.  None of it changes the "
        "hole's pull -- only how brightly it feeds.",
    ("collapse", "", "self"):
        "{name} collapsed into a black hole.  Its mass did not change, so "
        "nothing orbiting it moved at all -- but all of that mass now sits "
        "inside a horizon a few kilometres across.",
}


def lookup_why(field, direction, role):
    """The line for this edit, falling back from specific to general.

    Returns "" only if the table has a genuine hole, which
    ``test_no_edit_falls_through_to_silence`` forbids.
    """
    for key in ((field, direction, role), (field, "", role),
                (field, direction, "self"), (field, "", "self")):
        if key in WHY:
            return WHY[key]
    return ""


#: Water's phase boundaries at one atmosphere, kelvin.  Not a material property
#: -- these are the two numbers an eight-year-old already has a feel for.
WATER_FREEZES_K = 273.0
WATER_BOILS_K = 373.0

#: Materials that plausibly carry water, so the two lines above mean something.
WATERY = ("rock", "ice", "rock_ice")


def _role(after, edited_id, changes):
    """Returns "attractor" when anything in the ripple orbits the edited body."""
    for change in changes:
        if change.is_edited:
            continue
        state = after.bodies.get(change.body_id)
        if state is not None and state.attractor_id == edited_id:
            return "attractor"
    return "self"


#: The BodyState attribute each editable field moves.  Fields whose WHY entry
#: is direction-free are absent, and _direction returns "" for them rather than
#: comparing some other quantity and calling the answer a direction.  "temp"
#: reads the instantaneous ``temperature``, not ``eq_temperature`` -- nothing
#: heats itself, so a temperature edit leaves the edited body's own
#: equilibrium bit-identical and could never say which way the edit went.
DIRECTED = {"mass": "mass", "density": "density", "temp": "temperature"}


def _direction(was, now, field):
    attribute = DIRECTED.get(field)
    if attribute is None:
        return ""
    return "up" if getattr(now, attribute) >= getattr(was, attribute) else "down"


def _temperature_consequences(was, now, material_key, vapour_k):
    """The lines a temperature edit earns, in the order they matter.

    The three crossings test against ``temperature`` -- what he actually
    set -- not ``eq_temperature``, which does not move for the edited body at
    all (nothing heats itself).  The closing line keeps quoting
    ``eq_temperature``: that pairing is the point, naming both what he set and
    where sunlight is already pulling it back to.
    """
    lines = []
    crossed_up = was.temperature < WATER_BOILS_K <= now.temperature
    crossed_down = was.temperature > WATER_FREEZES_K >= now.temperature
    if material_key in WATERY and crossed_up:
        lines.append("Above %d K water boils.  Any ocean %s had is steam now."
                     % (WATER_BOILS_K, now.name))
    if material_key in WATERY and crossed_down:
        lines.append("Below %d K water freezes solid." % WATER_FREEZES_K)
    if was.temperature < vapour_k <= now.temperature:
        lines.append("Past %d K, the surface itself boils away." % vapour_k)
    # Always, on every temperature edit: this is the lesson set_temperature's
    # own docstring names, and nothing on screen has ever said it.
    lines.append(
        "Sunlight decides where it settles, and it is already sliding back "
        "toward %d K." % round(now.eq_temperature))
    return lines


def why_lines(before, after, edited_id, field, changes, material_key="",
              vapour_k=2000.0):
    """The sentences under the numbers.  Never empty for a real edit."""
    was, now = before.bodies[edited_id], after.bodies[edited_id]
    role = _role(after, edited_id, changes)
    line = lookup_why(field, _direction(was, now, field), role)
    lines = [line.format(name=now.name)] if line else []
    if field == "temp":
        lines += _temperature_consequences(was, now, material_key, vapour_k)
    return tuple(lines)


# --- the maths ----------------------------------------------------------------------
#
# Two lines each: the relationship, then the same thing with his numbers in it.
# The audience is eight and ahead of his age in maths, and a relationship with
# nothing substituted into it is a thing to be told rather than a thing to
# check.  The substituted line must *evaluate* to the figure the rows above it
# show -- see test_the_period_formula_evaluates_to_the_number_beside_it.


@dataclass(frozen=True)
class Formula:
    general: str
    substituted: str


def _period_formula(after, change):
    """Kepler's third, for the first rippled body -- the one that moved most."""
    state = after.bodies.get(change.body_id)
    if state is None or not state.bound or state.period <= 0.0:
        return None
    attractor = after.bodies.get(state.attractor_id)
    if attractor is None:
        return None
    return Formula(
        general="T = 2*pi * sqrt(a^3 / G M)",
        substituted="%s:  T = 2*pi * sqrt(%.3f^3 / (G x %s)) = %d days"
                    % (state.name, state.a, _plain_number(attractor.mass),
                       round(state.period)))


def _density_formula(state):
    return Formula(
        general="rho = m / (4/3 pi r^3)",
        substituted="rho = %.3g g/cm3 for %s"
                    % (units.to_gcm3(state.density), state.name))


def _horizon_formula(state):
    return Formula(
        general="r = 2 G M / c^2",
        substituted="r = %s km across, for %s Suns of mass"
                    % ("{:,.1f}".format(units.to_km(state.radius)),
                       _plain_number(state.mass)))


def _temperature_formula(state):
    return Formula(
        general="T^4 = (1-A)/4 * sum_j T_j^4 (R_j / d_j)^2",
        substituted="every body warms every other one; %s settles at %d K"
                    % (state.name, round(state.eq_temperature)))


def formulas(before, after, edited_id, field, changes):
    """Which relationships this edit is an illustration of."""
    edited = after.bodies[edited_id]
    out = []
    ripple = [c for c in changes if not c.is_edited]

    if field in ("mass", "collapse") and ripple:
        period = _period_formula(after, ripple[0])
        if period is not None:
            out.append(period)
    if field == "collapse" or edited.kind == KIND_BLACK_HOLE:
        out.append(_horizon_formula(edited))
    elif field in ("mass", "density", "material"):
        out.append(_density_formula(edited))
    if field == "temp":
        out.append(_temperature_formula(edited))

    # Never nothing: a "Show the maths" button that reveals an empty box is
    # worse than no button, and ui.py only draws it when maths is non-empty.
    if not out:
        out.append(_density_formula(edited))
    return tuple(out)
