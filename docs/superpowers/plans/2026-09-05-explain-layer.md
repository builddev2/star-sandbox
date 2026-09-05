# Explain Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After an edit settles, show a floating panel saying what moved -- on the
edited body and on the bodies it rippled to -- and why.

**Architecture:** A pure `src/explain.py` snapshots the world before an edit and
diffs it after, returning plain-data records with format *tags* rather than
formatted strings.  An `ExplainRecorder` on `AppState` coalesces a burst of
scrub notches into one report.  `src/render/explain.py` draws it, and `ui.py`
gives it its buttons.  This is the `ImpactEvent` -> `render/effects.py` contract
applied a second time.

**Tech Stack:** Python 3.11, numpy, pygame 2.6, pytest.

**Spec:** `docs/superpowers/specs/2026-09-05-explain-layer-design.md` -- read it
first.  Every task argues from it.

## Global Constraints

Copied verbatim from `CLAUDE.md` and the spec.  Every task's requirements
implicitly include this section.

- **`src/*.py` must never import pygame.**  `tests/test_layering.py` runs a
  subprocess that asserts it.  `src/explain.py` is physics-tier: it returns
  numbers and format tags, never formatted strings and never a `pygame.Rect`.
- **Simulation units are AU, solar masses, days.**  `src/units.py` is the single
  source of truth for `G` and every conversion factor.  No other module may
  define one.
- **Never hold an array index across a frame.**  `World.remove` is a swap-remove.
  Snapshots are keyed by `body.id`; every lookup goes through `World.index_of`.
- **Every mutation of mass, radius or membership must be followed by
  `Simulation.refresh()`.**  The existing `EditTool._scrub` already does this;
  do not move or duplicate that call.
- **Every action has a button, and every button carries its hotkey.**  A new key
  needs a row in `hud.KEY_HELP` *and* a `Control` carrying that hotkey, or
  `tests/test_ui.py::test_every_shortcut_is_also_a_button` fails.  A hotkey does
  not inherit a control's `enabled` gate -- the key path must duplicate any
  precondition the button carries.
- **Pixel-size constants scale by `internal_height / scale.REF_HEIGHT` (360);
  interface sizes come from `widgets.Metrics`.**  Neither may hard-code a pixel
  count.
- **Nothing may cache a size across frames.**  The window is resizable.
- Tests that touch pygame set `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")`
  at module top, **before** importing pygame.
- Run one test file with `./.venv/bin/python -m pytest tests/test_explain.py -v`.
  The system Python may lack pygame/numpy.  The whole suite is
  `./.venv/bin/python selftest.py` and takes about 3m35s -- it **exceeds Bash's
  120s default timeout**, so pass `timeout: 400000` when running it.
- Commit after every task.  Commit messages are lowercase-prefixed
  (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`) and say *why*, in the voice of
  the existing log.  End every commit message with:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014dQijk3k6M9E8c2UguEaRG
```

## File Structure

| File | Responsibility |
|---|---|
| `src/explain.py` *(create)* | Snapshot, diff, ranking, the `why` table, the formulas.  Pure.  Tasks 1-4. |
| `tests/test_explain.py` *(create)* | Everything in `src/explain.py`.  No pygame. |
| `src/app/state.py` *(modify)* | `ExplainRecorder`, burst coalescing, `reset` clearing it.  Task 5. |
| `src/app/tools.py` *(modify)* | Two `note()` calls in `EditTool`; the Fly readout stacking.  Tasks 5, 6. |
| `tests/test_explain_burst.py` *(create)* | Burst coalescing.  Needs the app layer, so needs the dummy driver.  Task 5. |
| `src/render/widgets.py` *(modify)* | `overlay_x` shared helper.  Task 6. |
| `src/render/explain.py` *(create)* | The panel: its rect and its drawing.  Task 6. |
| `src/app/ui.py` *(modify)* | `_build_explain`, the `view.explain` toggle, `perform` branches.  Task 7. |
| `src/app/input.py` *(modify)* | The `Y` global key.  Task 7. |
| `src/render/hud.py` *(modify)* | The `Y` row in `KEY_HELP`.  Task 7. |
| `src/app/main.py` *(modify)* | One `settle` call and one panel draw in `step_once`.  Task 7. |
| `tests/test_layout.py` *(modify)* | The panel fits at the window floor.  Task 7. |

---

### Task 1: Snapshot the world

**Files:**
- Create: `src/explain.py`
- Test: `tests/test_explain.py`

**Interfaces:**
- Consumes: `src.orbits.dominant_attractor(world, index) -> int | None`,
  `src.orbits.elements_from_state(mu, rel_pos, rel_vel) -> dict` with keys
  `a`, `e`, `period` (days, **`None` when unbound**), `periapsis`, `bound`;
  `src.thermal.equilibrium_temperatures(world) -> np.ndarray`;
  `src.thermal.pinned_temperature(world, i) -> float | None`;
  `src.units.G`, `src.units.density_from_mass_radius(mass, radius)`.
- Produces: `explain.BodyState`, `explain.Snapshot`, `explain.snapshot(world)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_explain.py`:

```python
"""The explain layer: what an edit did, and why.

Pure -- no pygame anywhere in this file, because src/explain.py is physics-tier
and tests/test_layering.py enforces that.
"""

import numpy as np
import pytest

from src import explain, orbits, scenarios, thermal, units


@pytest.fixture
def world():
    """Planets only.  Moons drive the timestep down by two orders of magnitude
    and nothing here needs them."""
    return scenarios.solar_system(moons=False)


def index_of_name(world, name):
    return [b.name for b in world.bodies].index(name)


def id_of(world, name):
    return world.bodies[index_of_name(world, name)].id


def test_snapshot_records_every_body_by_id(world):
    snap = explain.snapshot(world)
    assert len(snap.bodies) == world.n
    for body in world.bodies[:world.n]:
        assert body.id in snap.bodies
        assert snap.bodies[body.id].name == body.name


def test_snapshot_records_the_real_orbit(world):
    earth = explain.snapshot(world).bodies[id_of(world, "Earth")]
    assert earth.attractor == "Sun"
    assert earth.bound is True
    assert earth.a == pytest.approx(1.0, abs=0.01)
    assert earth.e == pytest.approx(0.0167, abs=0.001)
    assert earth.period == pytest.approx(365.3, abs=1.0)


def test_the_sun_gets_a_pinned_temperature_not_an_equilibrium_one(world):
    """thermal.equilibrium_temperatures is meaningless for a radiator: it
    computes what a body absorbs, and a star's temperature comes from fusion.
    pinned_temperature is the one function that knows which bodies are held."""
    sun = explain.snapshot(world).bodies[id_of(world, "Sun")]
    expected = units.star_temperature(float(world.mass[index_of_name(world, "Sun")]))
    assert sun.eq_temperature == pytest.approx(expected, rel=1e-9)
    assert sun.eq_temperature > 5000.0


def test_snapshot_carries_the_substance_and_the_hole_dials(world):
    """Every field a later task reads is populated here, so no task has to
    retrofit the record it was handed."""
    earth = explain.snapshot(world).bodies[id_of(world, "Earth")]
    assert earth.material == "Rock"
    assert earth.material_key == "rock"
    assert earth.vapour_k > 1000.0
    # Not a hole, so both dials read zero rather than being absent.
    assert earth.spin == 0.0 and earth.disk == 0.0


def test_an_unbound_body_has_no_period(world):
    """period is None out of elements_from_state; a Row needs a float, so an
    escaping body reports 0.0 and bound=False rather than crashing the diff."""
    i = index_of_name(world, "Mercury")
    world.vel[i] = world.vel[i] * 5.0        # far past escape
    mercury = explain.snapshot(world).bodies[id_of(world, "Mercury")]
    assert mercury.bound is False
    assert mercury.period == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_explain.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'src.explain'`

- [ ] **Step 3: Write the implementation**

Create `src/explain.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_explain.py -v`
Expected: 5 passed.

- [ ] **Step 5: Confirm the layering rule still holds**

Run: `./.venv/bin/python -m pytest tests/test_layering.py -v`
Expected: PASS.  If it fails, `src/explain.py` has pulled in pygame through an
import -- it must not.

- [ ] **Step 6: Commit**

```bash
git add src/explain.py tests/test_explain.py
git commit -m "$(cat <<'MSG'
feat: freeze a body in the quantities an edit can move

Keyed by id, not index -- World.remove is a swap-remove and a snapshot
outlives a frame.  A star's temperature comes from pinned_temperature
rather than the equilibrium sum, because a radiator's heat is set by its
own fusion and not by what falls on it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014dQijk3k6M9E8c2UguEaRG
MSG
)"
```

---

### Task 2: Diff two snapshots into a ranked report

**Files:**
- Modify: `src/explain.py`
- Test: `tests/test_explain.py`

**Interfaces:**
- Consumes: `explain.snapshot`, `explain.BodyState`, `explain.Snapshot` (Task 1).
- Produces: `explain.Row(label, before, after, format)`,
  `explain.Change(name, is_edited, rows)`,
  `explain.Report(action, changes, why, maths)`,
  `explain.diff(before, after, edited_id, field, max_ripple=3, noise=NOISE) -> Report | None`.
  `Report.why` and `Report.maths` are `()` until Tasks 3 and 4 fill them.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_explain.py`:

```python
# --- the diff -----------------------------------------------------------------------

def double_the_sun(world):
    """The worked example the spec is built around.  Returns (before, after)."""
    from src import edits
    before = explain.snapshot(world)
    i = index_of_name(world, "Sun")
    edits.set_mass(world, i, 2.0 * float(world.mass[i]))
    return before, explain.snapshot(world)


def rows_of(change):
    return {row.label: row for row in change.rows}


def test_doubling_the_sun_reports_earths_real_new_year(world):
    """These numbers came out of the simulation, not out of a textbook: an
    instantaneous doubling leaves Earth's position and speed alone, so the
    circle becomes an ellipse with the old radius as its farthest point."""
    before, after = double_the_sun(world)
    report = explain.diff(before, after, id_of(world, "Sun"), "mass")

    earth = next(c for c in report.changes if c.name == "Earth")
    lap = rows_of(earth)["One lap"]
    assert lap.before == pytest.approx(365.3, abs=1.0)
    assert lap.after == pytest.approx(138.2, abs=1.0)
    stretch = rows_of(earth)["Stretch"]
    assert stretch.before == pytest.approx(0.0167, abs=0.002)
    assert stretch.after == pytest.approx(0.492, abs=0.01)


def test_the_edited_body_leads_and_the_ripple_is_ranked(world):
    before, after = double_the_sun(world)
    report = explain.diff(before, after, id_of(world, "Sun"), "mass")

    assert report.changes[0].name == "Sun"
    assert report.changes[0].is_edited is True
    # Three rippled bodies, ordered by how much they moved -- not by index.
    assert [c.name for c in report.changes[1:]] == ["Mercury", "Venus", "Earth"]
    assert all(c.is_edited is False for c in report.changes[1:])


def test_the_edited_bodys_own_rows_are_the_ones_that_moved(world):
    before, after = double_the_sun(world)
    report = explain.diff(before, after, id_of(world, "Sun"), "mass")
    sun = rows_of(report.changes[0])
    assert "Mass" in sun
    assert sun["Mass"].after == pytest.approx(2.0 * sun["Mass"].before, rel=1e-9)
    assert sun["Mass"].format == "mass"
    # Density was held constant, so radius followed mass.  Density did not move
    # and must not be listed.
    assert "Density" not in sun


def test_editing_a_small_body_ripples_to_nobody(world):
    from src import edits
    before = explain.snapshot(world)
    i = index_of_name(world, "Mercury")
    edits.set_mass(world, i, 1.5 * float(world.mass[i]))
    after = explain.snapshot(world)
    report = explain.diff(before, after, id_of(world, "Mercury"), "mass")
    assert [c.name for c in report.changes] == ["Mercury"]


def test_an_edit_that_moved_nothing_is_no_report(world):
    before = explain.snapshot(world)
    after = explain.snapshot(world)
    assert explain.diff(before, after, id_of(world, "Sun"), "mass") is None


def test_a_body_deleted_before_the_burst_settled_is_no_report(world):
    """Snapshots are keyed by id, so the edited body is simply absent from the
    second one.  Reporting on a body that is gone is worse than saying nothing."""
    before = explain.snapshot(world)
    gone = id_of(world, "Mars")
    world.remove(index_of_name(world, "Mars"))
    after = explain.snapshot(world)
    assert explain.diff(before, after, gone, "mass") is None


def test_the_report_survives_the_world_running_underneath_it(world):
    """The claim the whole design rests on: elements are invariant under motion
    along the orbit, so the sim advancing between baseline and report cannot
    contaminate the diff.

    A tolerance, not equality: osculating elements really do drift a little
    under n-body perturbation -- Jupiter pulls Earth's around over a hundred
    days -- and a test asserting otherwise would assert a falsehood and fail at
    random.  1% is far tighter than the 62% change being reported.
    """
    from src.sim import Simulation
    sim = Simulation(world)
    still = explain.diff(*double_the_sun(world), id_of(world, "Sun"), "mass")

    world2 = scenarios.solar_system(moons=False)
    before = explain.snapshot(world2)
    from src import edits
    i = index_of_name(world2, "Sun")
    edits.set_mass(world2, i, 2.0 * float(world2.mass[i]))
    sim2 = Simulation(world2)
    sim2.refresh()
    sim2.advance(100.0)
    moved = explain.diff(before, explain.snapshot(world2),
                         id_of(world2, "Sun"), "mass")

    a = rows_of(next(c for c in still.changes if c.name == "Earth"))
    b = rows_of(next(c for c in moved.changes if c.name == "Earth"))
    assert b["One lap"].after == pytest.approx(a["One lap"].after, rel=0.01)
    assert b["Stretch"].after == pytest.approx(a["Stretch"].after, rel=0.01)


def test_temperature_rows_use_equilibrium_not_the_instantaneous_value(world):
    """And a guard on the guard: assert that reading world.temperature instead
    *would* have moved, so swapping the source back fails this test."""
    from src import edits
    from src.sim import Simulation
    i = index_of_name(world, "Earth")
    edits.set_temperature(world, i, 3000.0)
    settled = explain.snapshot(world)
    hot = float(world.temperature[i])

    sim = Simulation(world)
    sim.refresh()
    sim.advance(30.0)
    thermal.advance(world, 30.0)
    later = explain.snapshot(world)

    earth_id = id_of(world, "Earth")
    assert later.bodies[earth_id].eq_temperature == pytest.approx(
        settled.bodies[earth_id].eq_temperature, rel=0.01)
    # The instantaneous value, meanwhile, has fallen a long way.
    assert later.bodies[earth_id].temperature < 0.9 * hot
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_explain.py -v`
Expected: the four Task 1 tests pass; the new ones FAIL with
`AttributeError: module 'src.explain' has no attribute 'diff'`.

- [ ] **Step 3: Write the implementation**

Append to `src/explain.py`:

```python
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
```

`BodyState` already carries `material`, `material_key`, `vapour_k`, `spin` and
`disk` from Task 1, so nothing here retrofits the record.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_explain.py -v`
Expected: all pass.

If `test_the_edited_body_leads_and_the_ripple_is_ranked` fails on the *names*,
do not change the expectation to match the code -- print the scores and work
out why.  Mercury, Venus and Earth are the inner three and must move most.

- [ ] **Step 5: Commit**

```bash
git add src/explain.py tests/test_explain.py
git commit -m "$(cat <<'MSG'
feat: diff two snapshots, ranked by what moved most

The ripple is the point.  "You made the Sun heavier" is not a lesson --
he is the one who did it -- and "Earth's year is now 138 days" is.

Compared by orbital element rather than by position, so the world running
underneath the burst cannot report a planet's own progress round its
orbit as though the edit had caused it.  Tested by advancing a hundred
days between baseline and report and asserting the answer holds to 1%.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014dQijk3k6M9E8c2UguEaRG
MSG
)"
```

---

### Task 3: The `why` sentences

**Files:**
- Modify: `src/explain.py`
- Test: `tests/test_explain.py`

**Interfaces:**
- Consumes: `explain.Report`, `explain.diff` (Task 2); `src.materials.Material`
  fields `name` and `vapour_temperature_k`.
- Produces: `explain.WHY` (dict), `explain.why_lines(before, after, edited_id,
  field, changes) -> tuple[str, ...]`.  `diff` now fills `Report.why`.
  `diff` gains no new parameters.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_explain.py`:

```python
# --- why ----------------------------------------------------------------------------

def test_doubling_the_sun_explains_the_ripple_not_the_edit(world):
    report = explain.diff(*double_the_sun(world), id_of(world, "Sun"), "mass")
    said = " ".join(report.why)
    assert "heavier Sun" in said
    assert "pulls harder" in said
    assert "oval" in said


def test_collapse_says_the_orbits_did_not_move(world):
    """The most valuable line in the table: mass is conserved, so the ripple is
    empty, and the sentence is what makes an empty ripple the right answer."""
    from src import edits
    before = explain.snapshot(world)
    i = index_of_name(world, "Sun")
    edits.collapse(world, i)
    report = explain.diff(before, explain.snapshot(world),
                          id_of(world, "Sun"), "collapse")
    said = " ".join(report.why)
    assert "mass did not change" in said
    assert "nothing orbiting it moved" in said


def test_heating_a_world_past_boiling_says_so(world):
    from src import edits
    before = explain.snapshot(world)
    i = index_of_name(world, "Earth")
    edits.set_temperature(world, i, 500.0)
    report = explain.diff(before, explain.snapshot(world),
                          id_of(world, "Earth"), "temp")
    said = " ".join(report.why)
    assert "water boils" in said


def test_every_temperature_edit_admits_it_is_not_pinned(world):
    """edits.set_temperature's docstring calls the relaxation "the whole
    lesson", and nothing on screen has ever said it."""
    from src import edits
    before = explain.snapshot(world)
    i = index_of_name(world, "Earth")
    edits.set_temperature(world, i, 500.0)
    report = explain.diff(before, explain.snapshot(world),
                          id_of(world, "Earth"), "temp")
    said = " ".join(report.why)
    assert "nothing pins it there" in said


def test_spinning_a_hole_produces_a_report_at_all(world):
    """set_spin moves nothing but Body.spin, so this is the edit most likely
    to diff to nothing and leave the panel silent."""
    from src import edits
    i = index_of_name(world, "Sun")
    edits.collapse(world, i)
    before = explain.snapshot(world)
    edits.set_spin(world, i, 0.9)
    report = explain.diff(before, explain.snapshot(world),
                          id_of(world, "Sun"), "spin")
    assert report is not None
    assert any(r.label == "Spin" for r in report.changes[0].rows)
    assert "drags space" in " ".join(report.why)


def test_no_edit_falls_through_to_silence():
    """Every (field, direction, role) an edit can produce must find a line.
    A report with no explanation is the failure mode this whole feature exists
    to fix, so it must not be reachable."""
    fields = ("mass", "density", "temp", "material", "spin", "disk", "collapse")
    for field in fields:
        for direction in ("up", "down"):
            for role in ("attractor", "self"):
                assert explain.lookup_why(field, direction, role), \
                    "no line for %s/%s/%s" % (field, direction, role)
```

- [ ] **Step 2: Run to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_explain.py -k why -v`
Expected: FAIL, `module 'src.explain' has no attribute 'lookup_why'`.

- [ ] **Step 3: Write the implementation**

Append to `src/explain.py`:

```python
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
    """"attractor" when anything in the ripple orbits the edited body."""
    for change in changes:
        if change.is_edited:
            continue
        for state in after.bodies.values():
            if state.name == change.name and state.attractor_id == edited_id:
                return "attractor"
    return "self"


#: The BodyState attribute each editable field moves.  Fields whose WHY entry
#: is direction-free are absent, and _direction returns "" for them rather than
#: comparing some other quantity and calling the answer a direction.
DIRECTED = {"mass": "mass", "density": "density", "temp": "eq_temperature"}


def _direction(was, now, field):
    attribute = DIRECTED.get(field)
    if attribute is None:
        return ""
    return "up" if getattr(now, attribute) >= getattr(was, attribute) else "down"


def _temperature_consequences(was, now, material_key, vapour_k):
    """The lines a temperature edit earns, in the order they matter."""
    lines = []
    crossed_up = was.eq_temperature < WATER_BOILS_K <= now.eq_temperature
    crossed_down = was.eq_temperature > WATER_FREEZES_K >= now.eq_temperature
    if material_key in WATERY and crossed_up:
        lines.append("Above %d K water boils.  Any ocean %s had is steam now."
                     % (WATER_BOILS_K, now.name))
    if material_key in WATERY and crossed_down:
        lines.append("Below %d K water freezes solid." % WATER_FREEZES_K)
    if was.eq_temperature < vapour_k <= now.eq_temperature:
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
```

Then wire it into `diff`, reading the two material fields `BodyState` already
carries from Task 1.  Replace `diff`'s `return Report(...)` with:

```python
    edited_state = after.bodies[edited_id]
    return Report(action=_action(before, after, edited_id, field),
                  changes=changes,
                  why=why_lines(before, after, edited_id, field, changes,
                                material_key=edited_state.material_key,
                                vapour_k=edited_state.vapour_k))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_explain.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/explain.py tests/test_explain.py
git commit -m "$(cat <<'MSG'
feat: say why, from a table rather than a chain of ifs

The collapse line is the one that earns its place: mass is conserved, so
the ripple is empty, and the sentence is what makes an empty ripple read
as the right answer rather than as a bug.

Every temperature edit now admits it is not pinned -- the lesson
set_temperature's own docstring names and that nothing on screen has
ever told him.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014dQijk3k6M9E8c2UguEaRG
MSG
)"
```

---

### Task 4: The maths, with his numbers in it

**Files:**
- Modify: `src/explain.py`
- Test: `tests/test_explain.py`

**Interfaces:**
- Consumes: everything from Tasks 1-3.
- Produces: `explain.Formula(general, substituted)` and
  `explain.formulas(before, after, edited_id, field, changes) -> tuple[Formula, ...]`.
  `diff` now fills `Report.maths`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_explain.py`:

```python
# --- the maths ----------------------------------------------------------------------

def test_the_period_formula_evaluates_to_the_number_beside_it(world):
    """The substituted line is not decoration: worked by hand in the
    simulation's own units, 2*pi*sqrt(0.659^3 / (G * 2.00)) really is 138 days.
    If the two ever disagree, the panel is lying to a child who will check."""
    import math
    report = explain.diff(*double_the_sun(world), id_of(world, "Sun"), "mass")
    formula = next(f for f in report.maths if "T =" in f.general)
    assert "a^3" in formula.general

    earth = next(c for c in report.changes if c.name == "Earth")
    lap = {r.label: r for r in earth.rows}["One lap"]
    assert ("%d days" % round(lap.after)) in formula.substituted


def test_a_black_hole_gets_its_horizon_formula(world):
    from src import edits
    before = explain.snapshot(world)
    i = index_of_name(world, "Sun")
    edits.collapse(world, i)
    report = explain.diff(before, explain.snapshot(world),
                          id_of(world, "Sun"), "collapse")
    assert any("2 G M / c^2" in f.general for f in report.maths)


def test_every_report_carries_at_least_one_formula(world):
    report = explain.diff(*double_the_sun(world), id_of(world, "Sun"), "mass")
    assert len(report.maths) >= 1
    for formula in report.maths:
        assert formula.general and formula.substituted
```

- [ ] **Step 2: Run to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_explain.py -k maths -v`
Expected: FAIL, no attribute `Formula`.

- [ ] **Step 3: Write the implementation**

Append to `src/explain.py`:

```python
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
    state = next((s for s in after.bodies.values() if s.name == change.name), None)
    if state is None or not state.bound or state.period <= 0.0:
        return None
    attractor = after.bodies.get(state.attractor_id)
    if attractor is None:
        return None
    return Formula(
        general="T = 2*pi * sqrt(a^3 / G M)",
        substituted="T = 2*pi * sqrt(%.3f^3 / (G x %.3g)) = %d days"
                    % (state.a, attractor.mass, round(state.period)))


def _density_formula(state):
    return Formula(
        general="rho = m / (4/3 pi r^3)",
        substituted="rho = %.3g g/cm3 for %s"
                    % (units.to_gcm3(state.density), state.name))


def _horizon_formula(state):
    return Formula(
        general="r = 2 G M / c^2",
        substituted="r = %s km across, for %.3g Suns of mass"
                    % ("{:,.1f}".format(units.to_km(state.radius)), state.mass))


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
    if field == "collapse" or edited.kind == "black hole":
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
```

`edited.kind == "black hole"` must match `src/body.py`'s `KIND_BLACK_HOLE`
exactly -- import it (`from .body import KIND_BLACK_HOLE`) and compare against
the constant rather than a literal.

Wire it into `diff`'s return, alongside `why`:

```python
                  maths=formulas(before, after, edited_id, field, changes))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_explain.py -v`
Expected: all pass.

- [ ] **Step 5: Run the physics tier to confirm nothing else moved**

Run: `./.venv/bin/python -m pytest tests/test_orbits.py tests/test_thermal.py tests/test_edits.py tests/test_layering.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/explain.py tests/test_explain.py
git commit -m "$(cat <<'MSG'
feat: the maths, with his numbers substituted into it

A relationship with nothing substituted into it is a thing to be told
rather than a thing to check, and this reader will check.  The test
asserts the substituted line evaluates to the figure the rows above it
show, so the two cannot drift into the panel lying to him.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014dQijk3k6M9E8c2UguEaRG
MSG
)"
```

---

### Task 5: Coalesce a burst of scrub notches into one report

**Files:**
- Modify: `src/app/state.py`
- Modify: `src/app/tools.py` (`EditTool._scrub`, `EditTool._collapse`)
- Test: `tests/test_explain_burst.py` *(create)*

**Interfaces:**
- Consumes: `explain.snapshot`, `explain.diff` (Tasks 1-4).
- Produces: `state.ExplainRecorder` with
  `note(world, body_id, field, now_ms)`, `settle(world, now_ms)`,
  `dismiss()`, and attributes `report` (a `Report` or `None`) and
  `enabled` (bool, default `True`).  `AppState.explain` holds one.
  `state.SETTLE_MS = 350`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_explain_burst.py`:

```python
"""One report per burst, not one per scrub notch.

Mass scrubs by ratio at 1.06 a notch, three notches a press, repeating while
held -- doubling the Sun is a dozen-plus separate set_mass calls, and a report
per call would be a strobe.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest

from src.app.state import AppState, SETTLE_MS
from src.app.tools import EditTool
from src.render.camera import Camera
from src.render.draw import ViewSettings
from src.sim import Simulation


@pytest.fixture
def state():
    # Simulation() builds the solar system itself; no reset needed.
    return AppState(Simulation(), Camera(800, 600), ViewSettings())


def select(state, name):
    world = state.sim.world
    index = [b.name for b in world.bodies].index(name)
    state.select(world.bodies[index].id)
    return index


def test_a_held_stepper_produces_exactly_one_report(state):
    """Twelve notches, one lesson."""
    state.set_tool(EditTool)
    select(state, "Sun")
    now = 1000
    for _ in range(12):
        state.explain.note(state.sim.world, state.selected_id, "mass", now)
        state.tool._scrub(state, 3)
        now += 30
        state.explain.settle(state.sim.world, now)
        assert state.explain.report is None, "reported mid-burst"
    state.explain.settle(state.sim.world, now + SETTLE_MS + 1)
    assert state.explain.report is not None
    assert state.explain.report.changes[0].name == "Sun"


def test_changing_field_mid_hold_closes_the_first_burst(state):
    state.set_tool(EditTool)
    select(state, "Sun")
    state.explain.note(state.sim.world, state.selected_id, "mass", 1000)
    state.tool._scrub(state, 12)
    # A different field opens a new burst and closes the old one immediately.
    state.explain.note(state.sim.world, state.selected_id, "density", 1010)
    assert state.explain.report is not None
    first = state.explain.report
    assert any(r.label == "Mass" for r in first.changes[0].rows)


def test_a_burst_that_never_settles_reports_nothing(state):
    state.set_tool(EditTool)
    select(state, "Sun")
    state.explain.note(state.sim.world, state.selected_id, "mass", 1000)
    state.tool._scrub(state, 3)
    state.explain.settle(state.sim.world, 1000 + SETTLE_MS - 1)
    assert state.explain.report is None


def test_turning_explanations_off_takes_no_snapshot_at_all(state):
    """Off means off -- not recording silently and hiding the result."""
    state.explain.enabled = False
    state.set_tool(EditTool)
    select(state, "Sun")
    state.explain.note(state.sim.world, state.selected_id, "mass", 1000)
    assert state.explain._baseline is None
    state.tool._scrub(state, 12)
    state.explain.settle(state.sim.world, 1000 + SETTLE_MS + 1)
    assert state.explain.report is None


def test_restart_drops_a_standing_report_and_an_open_burst(state):
    """F5 clears the world and rebuilds every Body, so both refer to bodies
    that no longer exist."""
    state.set_tool(EditTool)
    select(state, "Sun")
    state.explain.note(state.sim.world, state.selected_id, "mass", 1000)
    state.tool._scrub(state, 12)
    state.explain.settle(state.sim.world, 1000 + SETTLE_MS + 1)
    assert state.explain.report is not None
    state.reset()
    assert state.explain.report is None
    assert state.explain._baseline is None


def test_scrubbing_through_the_tool_records_without_the_test_calling_note(state):
    """The real path: EditTool._scrub must call note() itself, or nothing the
    child does through the buttons or the arrow keys is ever explained."""
    state.set_tool(EditTool)
    select(state, "Sun")
    state.tool._scrub(state, 12)
    assert state.explain._baseline is not None
```

- [ ] **Step 2: Run to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_explain_burst.py -v`
Expected: FAIL, `ImportError: cannot import name 'SETTLE_MS'`.

- [ ] **Step 3: Write the implementation**

In `src/app/state.py`, add `from .. import explain` to the imports and this
class above `AppState`:

```python
#: Milliseconds of quiet that end a burst of edits.  Wall clock, never
#: simulated time -- this is a property of how fast a child's hand moves, and
#: it must behave identically at one hour a second and at ten years a second,
#: the same argument render/effects.py makes about particle velocities.
SETTLE_MS = 350


class ExplainRecorder:
    """One report per burst of edits, not one per scrub notch.

    Mass scrubs by ratio at 1.06 a notch, three notches a press, repeating
    while held: doubling the Sun is a dozen-plus separate ``set_mass`` calls.
    A report per call would be a strobe, and the baseline would be the previous
    notch rather than the state before he started.

    So the baseline is taken once, on the first edit of a burst, and the report
    is computed once, when the burst has been quiet for ``SETTLE_MS``.
    """

    def __init__(self):
        #: The finished report on show, or None.
        self.report = None
        #: Off means off: ``note`` returns before taking a snapshot, so the
        #: feature costs nothing at all rather than recording invisibly.
        self.enabled = True
        self._baseline = None
        self._body_id = None
        self._field = None
        self._last_ms = 0

    def note(self, world, body_id, field, now_ms):
        """Call *before* each write.  Opens or extends a burst."""
        if not self.enabled or body_id is None:
            return
        if self._baseline is not None and (self._body_id != body_id
                                           or self._field != field):
            # A different body or a different property is a different lesson.
            self._close(world)
        if self._baseline is None:
            self._baseline = explain.snapshot(world)
            self._body_id, self._field = body_id, field
        self._last_ms = now_ms

    def settle(self, world, now_ms):
        """Call once a frame.  Closes a burst that has gone quiet."""
        if self._baseline is None:
            return
        if now_ms - self._last_ms >= SETTLE_MS:
            self._close(world)

    def dismiss(self):
        self.report = None

    def clear(self):
        """Drop everything.  A restart rebuilds every Body, so a standing
        report and an open burst both name bodies that no longer exist."""
        self.report = None
        self._baseline = None
        self._body_id = self._field = None

    def _close(self, world):
        found = explain.diff(self._baseline, explain.snapshot(world),
                             self._body_id, self._field)
        if found is not None:
            self.report = found
        self._baseline = None
        self._body_id = self._field = None
```

In `AppState.__init__`, after `self.effects = EffectField()`:

```python
        #: The "what changed" report, and the burst that produces it.
        self.explain = ExplainRecorder()
```

In `AppState.reset`, after `self.sim.reset(self.scenario)`:

```python
        self.explain.clear()
```

In `src/app/tools.py`, `EditTool._scrub`, immediately after `field` is resolved
and **before** the first `edits.set_*` call:

```python
        # Before the write, so the baseline is the state he started from.  The
        # recorder coalesces the whole held-stepper burst into one report.
        state.explain.note(world, body.id, field, pygame.time.get_ticks())
```

In `EditTool._collapse`, before `edits.collapse(...)`:

```python
        state.explain.note(state.sim.world, state.sim.world.bodies[index].id,
                           "collapse", pygame.time.get_ticks())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_explain_burst.py -v`
Expected: all pass.

- [ ] **Step 5: Run the tool and state tiers**

Run: `./.venv/bin/python -m pytest tests/test_tools_edit.py tests/test_state.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/app/state.py src/app/tools.py tests/test_explain_burst.py
git commit -m "$(cat <<'MSG'
feat: one report per burst of edits, not one per scrub notch

Doubling the Sun is a dozen-plus set_mass calls through a held stepper.
A report per call would strobe, and every baseline but the first would be
the previous notch rather than the state he started from.

Wall-clock milliseconds, never simulated time: this measures how fast a
child's hand moves, and it must behave the same at ten years a second.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014dQijk3k6M9E8c2UguEaRG
MSG
)"
```

---

### Task 6: Draw the panel

**Files:**
- Create: `src/render/explain.py`
- Modify: `src/render/widgets.py` (add `overlay_x`)
- Modify: `src/app/tools.py` (`FlyTool.readout_x` calls the shared helper; the
  readout stacks below the explain panel)
- Test: `tests/test_explain_panel.py` *(create)*

**Interfaces:**
- Consumes: `explain.Report`, `explain.Row` (Tasks 1-4);
  `render.hud.format_mass/format_length/format_body_temperature`;
  `render.widgets.Metrics`, `render.widgets.body_list_width`.
- Produces: `widgets.overlay_x(metrics, show_body_list) -> int`;
  `render.explain.ExplainPanel(font)` with
  `rect(report, metrics, x, show_maths) -> pygame.Rect`,
  `draw(surface, report, metrics, x, show_maths) -> pygame.Rect`,
  and `format_row(row) -> (str, str)`.
  `AppState.explain_rect` (a `pygame.Rect` or `None`), set by the draw pass.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_explain_panel.py`:

```python
"""The panel: it says the right words, and it stays inside the window."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest

from src import edits, explain, scenarios
from src.render import widgets
from src.render.explain import ExplainPanel
from src.render.pixelfont import PixelFont
from src.render.widgets import Metrics


@pytest.fixture(scope="module", autouse=True)
def display():
    pygame.display.init()
    pygame.display.set_mode((64, 64))
    yield
    pygame.display.quit()


@pytest.fixture
def report():
    world = scenarios.solar_system(moons=False)
    before = explain.snapshot(world)
    i = [b.name for b in world.bodies].index("Sun")
    edits.set_mass(world, i, 2.0 * float(world.mass[i]))
    return explain.diff(before, explain.snapshot(world),
                        world.bodies[i].id, "mass")


@pytest.fixture
def panel():
    return ExplainPanel(PixelFont(8))


def test_a_period_row_is_formatted_as_a_period_not_as_a_bare_float(panel, report):
    earth = next(c for c in report.changes if c.name == "Earth")
    lap = next(r for r in earth.rows if r.label == "One lap")
    was, now = panel.format_row(lap)
    assert "d" in was or "yr" in was
    assert "138" in now


def test_the_panel_stays_inside_the_window(panel, report):
    surface = pygame.Surface((480, 270))
    m = Metrics(panel.font, 480, 270)
    x = widgets.overlay_x(m, show_body_list=True)
    rect = panel.draw(surface, report, m, x, show_maths=True)
    assert rect.right <= 480
    assert rect.top >= m.content_top
    assert rect.bottom <= m.content_bottom


def test_the_panel_starts_clear_of_the_body_list_drawer(panel):
    m = Metrics(panel.font, 480, 270)
    opened = widgets.overlay_x(m, show_body_list=True)
    closed = widgets.overlay_x(m, show_body_list=False)
    assert opened > closed
    assert opened >= widgets.body_list_width(m)


def test_showing_the_maths_makes_the_panel_taller(panel, report):
    m = Metrics(panel.font, 800, 600)
    x = widgets.overlay_x(m, show_body_list=False)
    assert (panel.rect(report, m, x, show_maths=True).height
            > panel.rect(report, m, x, show_maths=False).height)


def test_the_fly_readout_and_the_explain_panel_read_one_helper():
    """FlyTool.readout_x used to implement the 'clear of the drawer' rule
    itself.  Two copies of that rule is one of them eventually being wrong --
    the same argument scale.contact_radii_au makes about drawing and aiming."""
    import inspect
    from src.app.tools import FlyTool
    source = inspect.getsource(FlyTool.readout_x)
    assert "overlay_x" in source
    assert "body_list_width" not in source
```

- [ ] **Step 2: Run to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_explain_panel.py -v`
Expected: FAIL, `No module named 'src.render.explain'`.

- [ ] **Step 3: Add the shared helper**

In `src/render/widgets.py`, directly below `body_list_width`:

```python
def overlay_x(metrics, show_body_list):
    """Left edge for anything drawn over the world, in pixels.

    The body-list drawer is open by default and sits in exactly this corner, so
    an overlay that ignores it is drawn underneath the drawer and is simply
    invisible the first time it appears.

    One function because two overlays need the same answer -- the Fly tool's
    fuel readout and the explain panel -- and two copies of the rule is one of
    them eventually being wrong.  The same argument ``scale.contact_radii_au``
    makes about drawing and aiming.
    """
    if show_body_list:
        return metrics.pad + body_list_width(metrics) + metrics.gap
    return 8
```

Replace `FlyTool.readout_x`'s body in `src/app/tools.py` with:

```python
    def readout_x(self, state, metrics):
        """Left edge for the overlay readout, in pixels."""
        return widgets.overlay_x(metrics, state.show_body_list)
```

- [ ] **Step 4: Write the panel**

Create `src/render/explain.py`:

```python
"""The "what changed" panel.

Floating over the world rather than in a column, because both columns are
already full: the Edit panel is at the six-row floor ``tests/test_layout.py``
enforces, and the right column has the inspector, the action grid and the zoom
cluster colliding at small sizes.  A floating panel owes nothing to that layout
and so cannot push any of it off screen.

Every number arrives as a ``Row`` carrying a *format tag*, never as text.
``src/explain.py`` may not import pygame, so it cannot reach the ``hud.format_*``
functions; resolving the tag is this module's job, which keeps exactly one
place in the program knowing what a mass looks like written down.
"""

import pygame

from .. import units
from . import hud, palette

#: Characters wide.  The inspector's width (``hud.PANEL_WIDTH_CHARS``), so the
#: interface has one text-panel width rather than two.
PANEL_CHARS = hud.PANEL_WIDTH_CHARS

TITLE = "WHAT CHANGED"

#: Tag -> the function that turns a raw simulation number into words.  The one
#: table that crosses the physics/render line.
FORMATTERS = {
    "mass": hud.format_mass,
    "length": hud.format_length,
    "temperature": hud.format_body_temperature,
    "period": hud.orbit_period_label,
    "density": lambda density: "%.3g g/cm3" % units.to_gcm3(density),
    "plain": lambda value: "%.2f" % value,
}


class ExplainPanel:
    def __init__(self, font):
        self.font = font
        self.line = font.height + 2

    def format_row(self, row):
        """(before, after) as text, through the formatter the row names."""
        formatter = FORMATTERS.get(row.format, FORMATTERS["plain"])
        return formatter(row.before), formatter(row.after)

    def lines(self, report, width_chars, show_maths):
        """Every line the panel will draw, as (text, colour).

        Built once and used for both sizing and drawing, so the rect can never
        be a line out of step with what goes in it.
        """
        out = [(TITLE, palette.SELECT), ("", palette.TEXT)]
        out.append((report.action, palette.TEXT))
        out.append(("", palette.TEXT))
        for change in report.changes:
            out.append((change.name,
                        palette.SELECT if change.is_edited else palette.ACCENT))
            for row in change.rows:
                was, now = self.format_row(row)
                out.append(("  %-9s %s -> %s" % (row.label.lower(), was, now),
                            palette.TEXT))
        out.append(("", palette.TEXT))
        for sentence in report.why:
            out += [(part, palette.TEXT_DIM)
                    for part in _wrap(sentence, width_chars)]
        if show_maths and report.maths:
            out.append(("", palette.TEXT))
            for formula in report.maths:
                out += [(part, palette.ACCENT)
                        for part in _wrap(formula.general, width_chars)]
                out += [(part, palette.TEXT_DIM)
                        for part in _wrap(formula.substituted, width_chars)]
        return out

    def rect(self, report, metrics, x, show_maths):
        """Where the panel sits.  Clipped to the content band, never past it."""
        lines = self.lines(report, PANEL_CHARS, show_maths)
        width = self.font.width("X") * PANEL_CHARS + metrics.pad * 2
        height = len(lines) * self.line + metrics.pad * 2
        top = metrics.content_top
        room = metrics.content_bottom - top
        return pygame.Rect(x, top, width, min(height, max(self.line, room)))

    def draw(self, surface, report, metrics, x, show_maths):
        """Paint it, trimming rows from the end if the window is too short."""
        rect = self.rect(report, metrics, x, show_maths)
        box = pygame.Surface(rect.size, pygame.SRCALPHA)
        box.fill(tuple(palette.PANEL) + (232,))
        surface.blit(box, rect.topleft)
        pygame.draw.rect(surface, palette.PANEL_EDGE, rect, 1)

        room = (rect.height - metrics.pad * 2) // self.line
        for k, (text, colour) in enumerate(
                self.lines(report, PANEL_CHARS, show_maths)[:max(1, room)]):
            self.font.draw(surface, rect.x + metrics.pad,
                           rect.y + metrics.pad + k * self.line, text, colour)
        return rect


def _wrap(text, chars):
    """Greedy word wrap.  Never returns an empty list, so a blank line stays
    a blank line rather than vanishing out of the layout."""
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = (current + " " + word).strip()
        if len(candidate) > chars and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    lines.append(current)
    return lines
```

`FORMATTERS["period"]` needs `hud.orbit_period_label`, which does not exist yet.
Add it to `src/render/hud.py`, beside the other formatters, and make
`orbit_facts` use it so there is only one implementation:

```python
def orbit_period_label(days):
    """A period in whichever unit reads better.  Mercury is 88 days; Neptune
    is 165 years, and "60,190 days" tells a child nothing."""
    if days <= 0.0:
        return "escaping"
    if days >= 2.0 * units.DAYS_PER_YEAR:
        return "%.1f yr" % units.to_years(days)
    return "%.1f days" % days
```

Then in `orbit_facts`, replace the three-branch `period_label` block with:

```python
    period_label = "escaping" if not bound else orbit_period_label(el["period"])
```

- [ ] **Step 5: Stack the Fly readout below the panel**

In `src/app/state.py`, in `AppState.__init__`:

```python
        #: Where the explain panel was last drawn, so the Fly readout can start
        #: below it instead of underneath it.  None when no panel is up.
        self.explain_rect = None
```

In `AppState.reset`, beside `self.explain.clear()`, add `self.explain_rect = None`.

In `src/app/tools.py`, `FlyTool._draw_readout`, replace `y = 34` with:

```python
        # Below the explain panel when one is up: both are overlays in the same
        # corner, and an edit followed by a switch to Fly puts both on screen.
        panel = state.explain_rect
        y = 34 if panel is None else panel.bottom + widgets.Metrics(
            font, *surface.get_size()).gap
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_explain_panel.py tests/test_hud.py tests/test_tools_fly.py -v`
Expected: all pass.  `test_hud.py` covers `orbit_facts`; if a period assertion
now fails, the wording changed -- reconcile by keeping `orbit_facts`' existing
output exactly, not by editing the test.

- [ ] **Step 7: Commit**

```bash
git add src/render/explain.py src/render/widgets.py src/render/hud.py \
        src/app/tools.py src/app/state.py tests/test_explain_panel.py
git commit -m "$(cat <<'MSG'
feat: draw the what-changed panel, over the world

Floating rather than in a column: both columns are already full, and the
Edit panel is at the six-row floor test_layout.py enforces.

The 'start clear of the body-list drawer' rule now lives in one helper
that both overlays read, instead of a copy inside FlyTool -- two copies
of that rule is one of them eventually being wrong.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014dQijk3k6M9E8c2UguEaRG
MSG
)"
```

---

### Task 7: Buttons, the Y key, and the frame

**Files:**
- Modify: `src/app/ui.py` (`_build_explain`, `VIEW_BUTTONS`, `perform`)
- Modify: `src/app/input.py` (the `Y` global key)
- Modify: `src/render/hud.py` (`KEY_HELP`)
- Modify: `src/app/main.py` (`step_once`)
- Modify: `src/app/state.py` (`show_maths` flag)
- Test: `tests/test_ui.py`, `tests/test_layout.py`

**Interfaces:**
- Consumes: `render.explain.ExplainPanel` and `widgets.overlay_x` (Task 6);
  `AppState.explain`, `AppState.explain_rect` (Tasks 5-6).
- Produces: controls `explain.close`, `explain.maths`, `view.explain`;
  `AppState.show_maths` (bool, default `False`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ui.py` (reuse that file's existing `state` / `ui`
fixtures and its `find` helper -- read them before writing):

```python
# --- the explain panel --------------------------------------------------------------

def _make_a_report(state):
    from src.app.tools import EditTool
    state.set_tool(EditTool)
    world = state.sim.world
    index = [b.name for b in world.bodies].index("Sun")
    state.select(world.bodies[index].id)
    state.explain.note(world, state.selected_id, "mass", 0)
    state.tool._scrub(state, 12)
    state.explain.settle(world, 10_000)
    assert state.explain.report is not None


def test_the_explain_panel_has_a_close_button(state, ui):
    _make_a_report(state)
    ui.build(state, SIZE)
    assert find(ui, "explain.close") is not None
    ui.perform(find(ui, "explain.close"), state)
    assert state.explain.report is None


def test_the_maths_button_toggles_and_disappears_with_the_report(state, ui):
    _make_a_report(state)
    ui.build(state, SIZE)
    assert state.show_maths is False
    ui.perform(find(ui, "explain.maths"), state)
    assert state.show_maths is True
    state.explain.dismiss()
    ui.build(state, SIZE)
    assert find(ui, "explain.maths") is None


def test_y_is_documented_and_has_a_button(state, ui):
    from src.render.hud import KEY_HELP
    assert "Y" in {combo for combo, _ in KEY_HELP}
    ui.build(state, SIZE)
    assert find(ui, "view.explain").hotkey == "Y"


def test_turning_explanations_off_hides_a_standing_report(state, ui):
    _make_a_report(state)
    ui.build(state, SIZE)
    ui.perform(find(ui, "view.explain"), state)
    assert state.explain.enabled is False
    ui.build(state, SIZE)
    assert find(ui, "explain.close") is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_ui.py -k explain -v`
Expected: FAIL.  `test_every_shortcut_is_also_a_button` will also fail once
`KEY_HELP` gains `Y` and no button carries it -- that is the parity pressure
working, and it is fixed by Step 3, not by editing the table.

- [ ] **Step 3: Write the implementation**

`src/render/hud.py`, in `KEY_HELP`, after the `("L", "name labels")` row:

```python
    ("Y", "explanations on and off"),
```

`src/app/state.py`, in `AppState.__init__`:

```python
        #: Whether the explain panel is showing its formulas.  Session-wide
        #: rather than per-report: someone who wants the maths wants it every
        #: time, and someone who does not never opens it.
        self.show_maths = False
```

`src/app/ui.py`, add to `VIEW_BUTTONS` after the `view.labels` row:

```python
    ("view.explain", "help", "", "Y", "Explain what an edit did"),
```

and to the `flags` dict in `_build_view_buttons`:

```python
            "view.explain": state.explain.enabled,
```

Add `_build_explain` and call it from `build` immediately after
`_build_tool_context`:

```python
    def _build_explain(self, state, m):
        """The close and maths buttons on the what-changed panel.

        The panel itself is drawn by ``render/explain.py`` during the world
        pass; this only places the two controls on top of it, from the same
        rect, so a button cannot be drawn anywhere but on its own panel.
        """
        report = state.explain.report if state.explain.enabled else None
        if report is None or self.hud is None:
            state.explain_rect = None
            return
        rect = self._explain_panel.rect(
            report, m, widgets.overlay_x(m, state.show_body_list),
            state.show_maths)
        state.explain_rect = rect
        self._place([(Control("explain.close", icon="close", compact=True,
                              tooltip="Close  -- the next edit brings it back"),
                      pygame.Rect(rect.right - m.icon_btn_w - m.gap,
                                  rect.y + m.gap, m.icon_btn_w, m.btn_h))])
        if not report.maths:
            return
        control = Control(
            "explain.maths",
            label="Hide the maths" if state.show_maths else "Show the maths",
            tooltip="The relationship behind it, with your numbers in it")
        width = min(m.button_width(control), rect.width - 2 * m.gap)
        self._place([(control, pygame.Rect(rect.x + m.gap,
                                           rect.bottom - m.btn_h - m.gap,
                                           width, m.btn_h))])
```

`UiLayer.__init__` gains `self._explain_panel = ExplainPanel(font)` (import
`from ..render.explain import ExplainPanel`).

Branches in `UiLayer.perform`, beside the other `view.*` ones:

```python
        if cid == "view.explain":
            state.explain.enabled = not state.explain.enabled
            return True
        if cid == "explain.close":
            state.explain.dismiss()
            return True
        if cid == "explain.maths":
            state.show_maths = not state.show_maths
            return True
```

`src/app/input.py`, beside the other view keys:

```python
    elif key == pygame.K_y:
        state.explain.enabled = not state.explain.enabled
```

`src/app/main.py`, in `_build_render_stack`, beside the other pixel-sized
objects (nothing may cache a size across frames):

```python
        self.explain_panel = ExplainPanel(self.font)
```

In `step_once`, after `state.sim.update_trails()`:

```python
        state.explain.settle(state.sim.world, pygame.time.get_ticks())
```

and in the drawing sequence, after `state.tool.draw_overlay(...)`:

```python
        report = state.explain.report if state.explain.enabled else None
        if report is not None:
            m = self.hud.metrics(surface.get_width(), surface.get_height())
            state.explain_rect = self.explain_panel.draw(
                surface, report, m,
                widgets.overlay_x(m, state.show_body_list), state.show_maths)
        else:
            state.explain_rect = None
```

**Ordering note:** the Fly readout reads `state.explain_rect`, and
`tool.draw_overlay` runs *before* this block.  So move the explain draw to
immediately **before** `state.tool.draw_overlay(...)`, not after -- otherwise
the Fly readout stacks against last frame's rect.  The panel then sits under
the tool overlay, which is correct: an aiming line must not be hidden by a
readout.

- [ ] **Step 4: Add the layout test**

In `tests/test_layout.py`, following the file's existing parametrised style
(read `cases()` first), add:

```python
def test_the_explain_panel_fits_at_every_window_size():
    """A panel drawn past the content band is a panel whose close button
    cannot be clicked."""
    from src import edits, explain, scenarios
    from src.render import widgets
    from src.render.explain import ExplainPanel

    world = scenarios.solar_system(moons=False)
    before = explain.snapshot(world)
    i = [b.name for b in world.bodies].index("Sun")
    edits.set_mass(world, i, 2.0 * float(world.mass[i]))
    report = explain.diff(before, explain.snapshot(world),
                          world.bodies[i].id, "mass")

    for width, height in WINDOWS + [config.min_internal(3)]:
        font = PixelFont(max(6, height // 45))
        panel = ExplainPanel(font)
        m = widgets.Metrics(font, width, height)
        for drawer in (True, False):
            for maths in (True, False):
                rect = panel.rect(report, m,
                                  widgets.overlay_x(m, drawer), maths)
                assert rect.right <= width, (width, height, drawer, maths)
                assert rect.top >= m.content_top
                assert rect.bottom <= m.content_bottom
```

- [ ] **Step 5: Run the interface tests**

Run: `./.venv/bin/python -m pytest tests/test_ui.py tests/test_layout.py tests/test_smoke.py -v`
Expected: all pass, including
`test_every_shortcut_is_also_a_button`.

- [ ] **Step 6: Run the whole suite**

Run: `./.venv/bin/python selftest.py`
(Use `timeout: 400000` -- this takes about 3m35s and exceeds Bash's default.)
Expected: every test passes.  1020 before this work, more now.

- [ ] **Step 7: Update `CLAUDE.md`**

Add to the Architecture table's `src/*.py` cell: `explain.py` (what an edit did
and why, as plain data).  Add to the Invariants list:

```markdown
- **The explain layer compares orbital *elements*, never positions**
  (`src/explain.py`).  The world keeps running between an edit's baseline and
  its report -- at one month a second a 350 ms burst is ten days -- and
  elements are invariant under motion along the orbit, so a position
  comparison would report a planet's own progress round its orbit as though
  the edit had caused it.  Temperature rows use
  `thermal.equilibrium_temperature` for the same reason, and a star's comes
  from `thermal.pinned_temperature`, since a radiator's heat is set by fusion
  rather than by what falls on it.  **One report per burst, not per scrub
  notch**: `ExplainRecorder` (`app/state.py`) takes the baseline on the first
  edit and computes the report after `SETTLE_MS` of quiet, in wall-clock
  milliseconds -- doubling the Sun is a dozen-plus `set_mass` calls through a
  held stepper.
- **`widgets.overlay_x` is where an overlay starts**, so the Fly readout and
  the explain panel cannot disagree about where the body-list drawer ends.
```

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "$(cat <<'MSG'
feat: the what-changed panel gets its buttons and its key

Y turns explanations off for someone who has stopped wanting them, and
carries the same precondition through the key path as through the button,
since a hotkey never passes UiLayer.perform's enabled gate.

The panel draws before the tool overlay, not after: the Fly readout
stacks against its rect, and an aiming line must not end up behind a
readout.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014dQijk3k6M9E8c2UguEaRG
MSG
)"
```

---

## Verification before calling this done

Run `./.venv/bin/python selftest.py` (timeout 400000) and paste the summary
line.  Then run the app and check by hand, because no test can:

```bash
./.venv/bin/python -m src.app.main
```

1. Press `3` for Edit, click the Sun, hold `+` on the Mass row until it doubles.
   The panel appears once, after the hold stops -- not during it.
2. It names the Sun first, then three inner planets, with Earth's One lap
   reading about 365 d -> 138 d.
3. `Show the maths` reveals a period formula whose substituted line ends in the
   same number the Earth row shows.
4. Press `Y`.  The panel goes.  Press `Y` again, edit again, it comes back.
5. Press `F5`.  No stale panel survives the restart.
6. Select a planet, press `4` for Fly, launch a craft.  The fuel readout sits
   below the explain panel, not underneath it.
7. Press `W` down to the smallest window.  Nothing is cut off; the close button
   is still clickable.
