# Measure Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A fifth tool, Measure. It has a ruler between two bodies (distance, gap, light time, closing speed) and an orbit timer that measures a period and sets it beside Kepler's third law.

**Architecture:** The physics goes in a new pure-numpy module, `src/measure.py`, with `separation()`, `kepler_period()` and `PeriodTimer`. The timer is owned by `AppState` so it survives tool switches, and it is sampled once a frame in `App.step_once`. `MeasureTool` in `src/app/tools.py` provides two panel rows and an overlay readout. The readout is built as `(text, colour)` lines so tests can check wording without counting pixels.

**Tech Stack:** Python 3.11, numpy, pygame-ce, pytest.

**Spec:** `docs/superpowers/specs/2026-09-24-measure-tool-design.md`

## Global Constraints

- `src/measure.py` must never import pygame. Add it to `PHYSICS_MODULES` in `tests/test_layering.py`.
- Simulation units are AU, solar masses and days. `G`, `C`, `SEC_PER_DAY` and `to_years` come from `src/units.py`. Define no constants of your own.
- Hold body **ids**, never array rows, across frames. Resolve them with `world.index_of(id)`, which returns `None` for a body that is gone.
- The timer is sampled **once a frame, never per substep**.
- Every documented key needs a button carrying that hotkey (`tests/test_ui.py` parity test), and every key path re-checks its own precondition.
- Don't hard-code pixel counts in new drawing code. Derive sizes from `widgets.Metrics`.
- Any new test file that touches pygame must set `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")` before importing pygame.
- Run the full suite with `./.venv/bin/python -m pytest tests/ -q`, and **pass `timeout: 300000` to the Bash tool**. The suite takes about 220 s and the default 120 s timeout silently backgrounds it. For the inner loop, run single files.
- Commit messages end with the attribution lines from the session's system reminder.

**Mutation rule for reviewers.** Every test below names the one-line deletion that should make it fail. Check that by making the deletion, not by reading the code.

---

### Task 1: `src/measure.py`, the ruler and the orbit timer

**Files:**
- Create: `src/measure.py`
- Create: `tests/test_measure.py`
- Modify: `tests/test_layering.py` (add `"src.measure"` to `PHYSICS_MODULES`)

**Interfaces:**
- Produces:
  - `measure.separation(world, i, j) -> dict` with float keys `centres` (AU), `gap` (AU, ≥ 0), `light_days` (days), `closing` (AU/day, positive while approaching) and `relative` (AU/day).
  - `measure.kepler_period(world, index, attractor) -> (period_days | None, a_au)`. `period_days` is `None` when the orbit is unbound.
  - `measure.PeriodTimer(world, index)`. Attributes: `body_id`, `attractor_id` (or `None`), `status` (one of `"timing"`, `"too fast"`, `"not in orbit"`, `"lost"`, `"moved on"`), `laps` (list of world days, where `laps[0]` is the start) and `swept` (signed radians). Methods: `.sample(world)`, `.measured` (property, days or `None`), `.fraction_round` (property, 0..1) and `.mass_from_orbit(world)` (solar masses or `None`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_measure.py`:

```python
"""The ruler and the orbit timer: numbers a child can check against a book."""

import math

import numpy as np
import pytest

from src import materials, measure, scenarios, units
from src.sim import Simulation
from src.world import World


def _pair(separation_au, vel_b=(0.0, 0.0), radius=0.0):
    world = World()
    rock = materials.get("rock")
    world.add("A", rock, units.EARTH_MASS, radius, np.zeros(2), np.zeros(2))
    world.add("B", rock, units.EARTH_MASS, radius,
              np.array([separation_au, 0.0]), np.array(vel_b))
    return world


# --- the ruler ---------------------------------------------------------------------

def test_light_crosses_one_au_in_499_seconds():
    # Deleting the division by units.C makes this fail.
    s = measure.separation(_pair(1.0), 0, 1)
    assert s["centres"] == pytest.approx(1.0)
    assert s["light_days"] * units.SEC_PER_DAY == pytest.approx(499.005, abs=0.01)


def test_the_gap_is_surface_to_surface_and_never_negative():
    # Leaving out either radius, or the max(0, ...), makes this fail.
    assert measure.separation(_pair(1.0, radius=0.1), 0, 1)["gap"] == pytest.approx(0.8)
    assert measure.separation(_pair(0.1, radius=0.1), 0, 1)["gap"] == 0.0


def test_closing_is_positive_while_they_approach():
    # B sits on +x and moves towards A.  Flipping the sign makes this fail.
    approaching = measure.separation(_pair(1.0, vel_b=(-0.01, 0.0)), 0, 1)
    receding = measure.separation(_pair(1.0, vel_b=(0.01, 0.0)), 0, 1)
    assert approaching["closing"] == pytest.approx(0.01)
    assert receding["closing"] == pytest.approx(-0.01)


def test_sideways_motion_is_relative_speed_but_not_closing():
    s = measure.separation(_pair(1.0, vel_b=(0.0, 0.02)), 0, 1)
    assert s["closing"] == pytest.approx(0.0, abs=1e-15)
    assert s["relative"] == pytest.approx(0.02)


# --- Kepler ------------------------------------------------------------------------

def test_kepler_gives_a_year_at_one_au():
    world = scenarios.two_body()
    period, a = measure.kepler_period(world, 1, 0)
    assert a == pytest.approx(1.0, rel=1e-6)
    assert period == pytest.approx(365.256, rel=1e-4)


# --- the timer ---------------------------------------------------------------------

def _time(world, index, frame_days, laps=1, limit_days=3000.0):
    """Advance in frame-sized chunks, sampling once a chunk, like the app."""
    sim = Simulation(world=world)
    timer = measure.PeriodTimer(sim.world, index)
    while len(timer.laps) < laps + 1 and sim.world.time < limit_days:
        sim.advance(frame_days, budget_s=1e9)
        timer.sample(sim.world)
        if timer.status != "timing":
            break
    return sim, timer


def test_a_lap_matches_kepler_to_a_twentieth_of_a_percent():
    """7-day frames: without interpolating the crossing, the lap lands on a
    frame boundary and the error is ~1.6%.  With it, measured 0.035%.
    Replacing the interpolated crossing with world.time makes this fail."""
    sim, timer = _time(scenarios.two_body(), 1, 7.0)
    kepler, _ = measure.kepler_period(sim.world, 1, 0)
    assert timer.status == "timing"
    assert timer.measured == pytest.approx(kepler, rel=5e-4)


def test_a_retrograde_orbit_is_timed_too():
    # Counting laps on the signed angle instead of its magnitude makes this fail.
    world = scenarios.two_body()
    world.vel[:world.n] *= -1.0
    sim, timer = _time(world, 1, 7.0)
    assert timer.swept < 0.0
    assert timer.measured == pytest.approx(365.256, rel=1e-3)


def test_a_frame_longer_than_a_quarter_orbit_is_refused():
    """200-day frames on a 365-day orbit: the angle between samples cannot be
    unwrapped.  Without the guard the timer reported 600 days as a lap."""
    sim, timer = _time(scenarios.two_body(), 1, 200.0)
    assert timer.status == "too fast"
    assert timer.measured is None


def test_the_timer_recovers_once_time_slows_down():
    # Leaving out the status reset to "timing" in sample() makes this fail.
    sim = Simulation(world=scenarios.two_body())
    timer = measure.PeriodTimer(sim.world, 1)
    sim.advance(200.0, budget_s=1e9)
    timer.sample(sim.world)
    assert timer.status == "too fast"
    while len(timer.laps) < 2 and sim.world.time < 2000.0:
        sim.advance(5.0, budget_s=1e9)
        timer.sample(sim.world)
    assert timer.status == "timing"
    assert timer.measured == pytest.approx(365.256, rel=1e-3)


def test_earth_takes_a_sidereal_year():
    """The number in the book: 365.256 days.  Planets only, 6-day frames,
    which is what the one-year-a-second setting gives."""
    world = scenarios.solar_system(moons=False)
    earth = [b.name for b in world.bodies].index("Earth")
    sim, timer = _time(world, earth, 6.0)
    assert timer.measured == pytest.approx(365.256, rel=5e-4)


def test_timing_the_earth_weighs_the_sun():
    """a^3 / P^2 in AU and years is the pair's mass in suns.  Using the
    period in days instead of years makes this fail."""
    sim, timer = _time(scenarios.two_body(), 1, 7.0)
    assert timer.mass_from_orbit(sim.world) == pytest.approx(
        1.0 + units.EARTH_MASS, rel=2e-3)


def test_no_mass_before_the_first_lap():
    sim = Simulation(world=scenarios.two_body())
    timer = measure.PeriodTimer(sim.world, 1)
    sim.advance(30.0, budget_s=1e9)
    timer.sample(sim.world)
    assert timer.measured is None
    assert timer.mass_from_orbit(sim.world) is None
    assert 0.05 < timer.fraction_round < 0.12      # 30 of 365 days


def test_the_sun_is_not_in_orbit():
    world = scenarios.two_body()
    timer = measure.PeriodTimer(world, 0)
    assert timer.attractor_id is None
    assert timer.status == "not in orbit"
    timer.sample(world)                           # must not raise
    assert timer.status == "not in orbit"


def test_a_removed_body_is_lost_not_a_crash():
    # Deleting the `index is None` check makes this raise.
    sim = Simulation(world=scenarios.two_body())
    timer = measure.PeriodTimer(sim.world, 1)
    sim.world.remove(1)
    sim.refresh()
    sim.advance(1.0, budget_s=1e9)
    timer.sample(sim.world)
    assert timer.status == "lost"


def test_a_body_captured_by_another_star_has_moved_on():
    """Carry the secondary to a second, far heavier star and set it orbiting
    there.  Deleting the dominant_attractor comparison makes this fail."""
    sim = Simulation(world=scenarios.two_body())
    world = sim.world
    timer = measure.PeriodTimer(world, 1)
    far = np.array([50.0, 0.0])
    world.add("Other", materials.get("hydrogen"), 5.0, units.SUN_RADIUS,
              far, np.zeros(2), kind=world.bodies[0].kind)
    world.pos[1] = far + np.array([0.5, 0.0])
    world.vel[1] = np.array([0.0, math.sqrt(units.G * 5.0 / 0.5)])
    sim.refresh()
    sim.advance(1.0, budget_s=1e9)
    timer.sample(world)
    assert timer.status == "moved on"
```

In `tests/test_layering.py`, change the end of `PHYSICS_MODULES` from `"src.maneuver", "src.save",` to `"src.maneuver", "src.save", "src.measure",`.

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `./.venv/bin/python -m pytest tests/test_measure.py -q`
Expected: every test errors with `ImportError: cannot import name 'measure'`.

- [ ] **Step 3: Write `src/measure.py`**

```python
"""Measuring things: the ruler between two bodies, and the orbit timer.

Numbers a child can check against a book.  The ruler is plain geometry.  The
timer counts the angle a body sweeps about what it orbits, and when it has gone
all the way round it has *measured* an orbital period -- which can then be set
beside what Kepler's third law predicts from the orbit's size, and turned round
to weigh what it orbits (``a^3 / P^2`` in AU, years and suns).

**Sampled once a frame**, like the trails.  It measures by angle swept rather
than by time, so it is right at any speed -- up to the point where one frame
carries the body more than a quarter of the way round.  Past that, which way it
went between two samples is ambiguous, and the timer says "too fast" and starts
again rather than report a number it cannot stand behind.  Measured: 200-day
frames on a one-year orbit reported a 600-day "lap" before the guard existed.

**Holds ids, never rows**: ``World.remove`` is a swap-remove and a timer lives
for hundreds of frames.

Imports numpy and nothing from pygame -- ``tests/test_layering.py`` asserts it.
"""

import math

from . import orbits, units

TWO_PI = 2.0 * math.pi
#: The longest gap between two samples, as a fraction of the predicted period,
#: that the timer will trust.  Unwrapping is exact below half a turn; a quarter
#: leaves room for an eccentric orbit racing through periapsis faster than its
#: average.
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


def kepler_period(world, index, attractor):
    """``(period, a)``: what Kepler's third law says, in days, from the
    osculating semi-major axis in AU.  ``period`` is None for an unbound orbit.
    """
    mu = units.G * float(world.mass[attractor] + world.mass[index])
    el = orbits.elements_from_state(mu, world.pos[index] - world.pos[attractor],
                                    world.vel[index] - world.vel[attractor])
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
        period, _ = kepler_period(world, index, attractor)
        if period is None:
            self.status = "not in orbit"
            return
        gap = world.time - self._last_time
        if gap <= 0.0:
            return                              # paused
        if gap > MAX_SAMPLE_FRACTION * period:
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
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `./.venv/bin/python -m pytest tests/test_measure.py tests/test_layering.py -q`
Expected: all 15 pass. This code and these tests were run together before the plan was written: 15 passed in 0.2 s.

- [ ] **Step 5: Mutation check**

Apply each deletion named in the test docstrings one at a time, and confirm the named test fails:
1. Replace `self._last_time + fraction * gap` with `world.time`.
2. Replace `if gap > MAX_SAMPLE_FRACTION * period:` with `if False:`.
3. Replace `units.to_years(period)` with `period`.

Revert after each one.

- [ ] **Step 6: Commit**

```bash
git add src/measure.py tests/test_measure.py tests/test_layering.py
git commit -m "feat: measure.py -- a ruler, and an orbit timer that checks Kepler"
```

---

### Task 2: Formatters for light time and orbital periods

**Files:**
- Modify: `src/render/hud.py` (after `format_speed`, around line 99)
- Test: `tests/test_hud.py` (append)

**Interfaces:**
- Produces: `hud.format_light_time(days) -> str` and `hud.format_period(days) -> str`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_hud.py`)

```python
def test_light_time_reads_in_the_unit_a_book_uses():
    from src import units
    from src.render.hud import format_light_time
    assert format_light_time(1.0 / units.C) == "8.3 min"          # the Sun
    assert format_light_time(30.07 / units.C) == "4.2 hr"         # Neptune
    assert format_light_time(0.00257 / units.C) == "1.3 s"        # the Moon
    assert format_light_time(2.0) == "2.0 days"


def test_a_period_is_four_figures_in_a_sensible_unit():
    from src.render.hud import format_period
    assert format_period(365.2564) == "365.3 d"
    assert format_period(87.969) == "87.97 d"
    assert format_period(0.5) == "12 hr"
    assert format_period(60190.0) == "164.8 yr"
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `./.venv/bin/python -m pytest tests/test_hud.py -q -k "light_time or period_is"`
Expected: FAIL with `ImportError: cannot import name 'format_light_time'`.

- [ ] **Step 3: Implement** (in `src/render/hud.py`, directly after `format_speed`)

```python
def format_light_time(days):
    """How long light takes, in whichever unit reads naturally -- "8.3 min"
    from the Sun, "1.3 s" from the Moon, the numbers the books quote."""
    seconds = days * units.SEC_PER_DAY
    if seconds < 60.0:
        return "%.1f s" % seconds
    if seconds < 3600.0:
        return "%.1f min" % (seconds / 60.0)
    if seconds < units.SEC_PER_DAY:
        return "%.1f hr" % (seconds / 3600.0)
    return "{:,.1f} days".format(days)


def format_period(days):
    """An orbital period to four figures: "365.3 d", "12 hr", "164.8 yr".

    Four figures because the timer is compared against Kepler, and three would
    hide every difference worth seeing.
    """
    if days < 1.0:
        return "%.4g hr" % (days * 24.0)
    if days < 1000.0:
        return "%.4g d" % days
    return "%.4g yr" % units.to_years(days)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `./.venv/bin/python -m pytest tests/test_hud.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/render/hud.py tests/test_hud.py
git commit -m "feat: light time and orbital periods in the units a book uses"
```

---

### Task 3: `AppState` owns the timer, and the frame loop samples it

**Files:**
- Modify: `src/app/state.py`. Change the import on line 13 to `from .. import explain, measure, save, units`. Add a field in `__init__` after `self.dialog = None`. Add the methods after `stop_selected`. Clear the timer in `reset()` and `load_slot()`.
- Modify: `src/app/main.py`. In `step_once`, after `state.sim.update_trails()`, add `state.sample_timer()`.
- Test: `tests/test_state.py` and `tests/test_smoke.py` (append)

**Interfaces:**
- Consumes: `measure.PeriodTimer` (Task 1).
- Produces: `AppState.period_timer` (a `PeriodTimer` or `None`), `AppState.start_timer() -> bool`, `AppState.stop_timer()` and `AppState.sample_timer()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_state.py`:

```python
# --- the orbit timer ---------------------------------------------------------------

def _earth(st):
    return next(b for b in st.sim.world.bodies if b.name == "Earth")


def test_start_timer_needs_a_selection(state):
    assert state.start_timer() is False
    assert state.period_timer is None


def test_start_timer_times_the_selected_body(state):
    state.select(_earth(state).id)
    assert state.start_timer() is True
    assert state.period_timer.body_id == _earth(state).id
    state.stop_timer()
    assert state.period_timer is None


def test_the_timer_survives_a_tool_switch(state):
    # The reason it lives on the state and not the tool.
    from src.app.tools import EditTool, SelectTool
    state.set_tool(SelectTool)
    state.select(_earth(state).id)
    state.start_timer()
    timer = state.period_timer
    state.set_tool(EditTool)
    assert state.period_timer is timer


def test_restart_drops_the_timer(state):
    # Deleting the line in reset() makes this fail.
    state.select(_earth(state).id)
    state.start_timer()
    state.reset()
    assert state.period_timer is None


def test_load_drops_the_timer(saving_state):
    # Deleting the line in load_slot() makes this fail.
    st = saving_state
    st.open_saves()
    st.save_slot(1)
    st.close_dialog()
    st.select(_earth(st).id)
    st.start_timer()
    st.open_saves()
    assert st.load_slot(1) is True
    assert st.period_timer is None
```

Append to `tests/test_smoke.py`:

```python
def test_the_frame_loop_samples_the_orbit_timer(app):
    """Deleting state.sample_timer() from step_once makes this fail: the
    timer would never see the world move."""
    state = app.state
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    state.select(earth.id)
    assert state.start_timer()
    for _ in range(5):
        app.step_once()
    assert state.period_timer.swept != 0.0
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `./.venv/bin/python -m pytest tests/test_state.py tests/test_smoke.py -q -k "timer"`
Expected: FAIL with `AttributeError: 'AppState' object has no attribute 'start_timer'`.

- [ ] **Step 3: Implement**

In `AppState.__init__`, after `self.dialog = None`:

```python
        #: The orbit the Measure tool is timing, or None.  Here rather than on
        #: the tool because a tool is rebuilt every time it is picked, and
        #: timing Neptune should survive a look at the Edit panel.  Not saved:
        #: it is a stopwatch, not part of the world.
        self.period_timer = None
```

After `stop_selected`:

```python
    def start_timer(self):
        """Time the selected body round what it orbits.  False with nothing
        selected -- the key path relies on this check, since a key never
        passes through a disabled button's gate."""
        index = self.selected_index
        if index is None:
            return False
        self.period_timer = measure.PeriodTimer(self.sim.world, index)
        return True

    def stop_timer(self):
        self.period_timer = None

    def sample_timer(self):
        """Once a frame, after the physics -- never per substep."""
        if self.period_timer is not None:
            self.period_timer.sample(self.sim.world)
```

In `reset()`, after `self.body_list_offset = 0`, add `self.period_timer = None`.
In `load_slot()`, after `self.effects.clear()`, add `self.period_timer = None`.

In `src/app/main.py` `step_once`, change

```python
        state.sim.update_trails()
```

to

```python
        state.sim.update_trails()
        # The orbit timer, on the same once-a-frame grain as the trails.
        state.sample_timer()
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `./.venv/bin/python -m pytest tests/test_state.py tests/test_smoke.py -q`
Expected: all pass.

- [ ] **Step 5: Mutation check.** Delete each of the three new lines in turn (`reset`, `load_slot`, `step_once`) and confirm its named test fails. Revert.

- [ ] **Step 6: Commit**

```bash
git add src/app/state.py src/app/main.py tests/test_state.py tests/test_smoke.py
git commit -m "feat: the state owns an orbit timer, sampled once a frame"
```

---

### Task 4: `MeasureTool`, key 5, its hotbar button, its icons and its keys

**Files:**
- Modify: `src/app/tools.py`. Import `measure` in the existing `from .. import ...` line. Add `class MeasureTool` after `FlyTool`, before `_mean_radius` (around line 1831). Register `pygame.K_5: MeasureTool` in `TOOLS`.
- Modify: `src/app/ui.py:67-72`. Add the row to `TOOL_BUTTONS`.
- Modify: `src/render/widgets.py`. Add `"measure"` and `"clock"` icons in `draw_icon`, after the `"fly"` branch.
- Modify: `src/render/hud.py`. Change `KEY_HELP`.
- Modify: `tests/test_ui.py:755-757` (the parity test's tool list) and `tests/test_layout.py:23,38` (import and `TOOLS`).
- Create: `tests/test_tools_measure.py`

**Interfaces:**
- Consumes: `AppState.start_timer/stop_timer/period_timer` (Task 3), and `TargetLock` and `DragPan` (existing).
- Produces: `tools.MeasureTool`, with `name = "Measure"` and control ids `measure.target`, `measure.time` and `measure.stop`. Task 5 adds `readout_lines(state)` and `draw_overlay`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tools_measure.py`:

```python
"""The Measure tool: its panel, its keys, and where it sits."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest

from src import scenarios
from src.app.state import AppState
from src.app.tools import MeasureTool
from src.render.camera import Camera
from src.render.draw import ViewSettings
from src.sim import Simulation

SIZE = (640, 360)


@pytest.fixture
def state():
    pygame.init()
    sim = Simulation(world=scenarios.solar_system(moons=False))
    app = AppState(sim, Camera(*SIZE), ViewSettings())
    app.set_tool(MeasureTool)
    return app


def _body(state, name):
    return next(b for b in state.sim.world.bodies if b.name == name)


def _key(key):
    return pygame.event.Event(pygame.KEYDOWN, key=key, mod=0)


def _controls(state):
    return {c.id: c for row in state.tool.controls(state, None) for c in row}


def test_the_tool_registers_under_key_five():
    from src.app.tools import TOOL_BY_NAME, TOOLS
    assert TOOLS[pygame.K_5] is MeasureTool
    assert TOOL_BY_NAME["measure"] is MeasureTool


def test_the_hotbar_has_a_measure_button():
    from src.app.ui import TOOL_BUTTONS
    assert ("tool.measure", "Measure") in {(b[0], b[1]) for b in TOOL_BUTTONS}


def test_timing_is_disabled_and_says_why_with_nothing_selected(state):
    time = _controls(state)["measure.time"]
    assert not time.enabled
    assert time.label == "Pick a body first"
    assert not _controls(state)["measure.stop"].enabled


def test_the_button_starts_timing_the_selection(state):
    # Deleting the measure.time branch of perform makes this fail.
    state.select(_body(state, "Mars").id)
    time = _controls(state)["measure.time"]
    assert time.enabled and time.hotkey == "P"
    assert state.tool.perform(time, state)
    assert state.period_timer.body_id == _body(state, "Mars").id
    assert state.tool.perform(_controls(state)["measure.stop"], state)
    assert state.period_timer is None


def test_p_times_the_selection_and_does_nothing_without_one(state):
    assert state.tool.on_event(_key(pygame.K_p), state)
    assert state.period_timer is None
    state.select(_body(state, "Mars").id)
    state.tool.on_event(_key(pygame.K_p), state)
    assert state.period_timer.body_id == _body(state, "Mars").id


def test_x_then_click_locks_the_target(state):
    earth = _body(state, "Earth")
    state.camera.frame_body(state.sim.world, state.sim.world.index_of(earth.id))
    x, y = state.camera.world_to_screen(state.sim.world.pos[state.sim.world.index_of(earth.id)])
    state.tool.on_event(_key(pygame.K_x), state)
    assert state.tool.picking_target
    state.tool.on_event(pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, button=1, pos=(int(x), int(y))), state)
    assert state.tool.target_id == earth.id
    assert _controls(state)["measure.target"].label == "Target: Earth"


def test_a_plain_click_selects(state):
    mars = _body(state, "Mars")
    index = state.sim.world.index_of(mars.id)
    state.camera.frame_body(state.sim.world, index)
    pos = tuple(int(v) for v in state.camera.world_to_screen(state.sim.world.pos[index]))
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos), state)
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=pos), state)
    assert state.selected_id == mars.id


def test_every_control_explains_itself(state):
    state.select(_body(state, "Mars").id)
    state.start_timer()
    for control in _controls(state).values():
        assert control.hint(), control.id


def test_the_keys_are_documented():
    from src.render.hud import KEY_HELP
    combos = {combo for combo, _ in KEY_HELP}
    assert "P" in combos
    assert any("5" in combo.split() for combo in combos)


@pytest.mark.parametrize("name", ["measure", "clock"])
def test_the_new_icons_draw_something(name):
    # Deleting either draw_icon branch leaves the surface blank.
    from src.render.widgets import draw_icon
    surface = pygame.Surface((24, 24))
    surface.fill((0, 0, 0))
    draw_icon(surface, name, pygame.Rect(2, 2, 20, 20), (255, 255, 255))
    assert pygame.mask.from_threshold(surface, (255, 255, 255), (60, 60, 60, 255)).count() > 10
```

Note: `controls(state, metrics)` does not read `metrics` in this tool, so the tests pass `None`.

In `tests/test_ui.py`, change `test_every_shortcut_is_also_a_button` to import and loop over `MeasureTool` as well:

```python
    from src.app.tools import EditTool, FlyTool, LaunchTool, MeasureTool, SelectTool
    keys = set()
    for tool in (SelectTool, LaunchTool, EditTool, FlyTool, MeasureTool):
```

In `tests/test_layout.py`, import `MeasureTool` on line 23 and set `TOOLS = (SelectTool, LaunchTool, EditTool, FlyTool, MeasureTool)`.

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `./.venv/bin/python -m pytest tests/test_tools_measure.py -q`
Expected: collection error, `ImportError: cannot import name 'MeasureTool'`.

- [ ] **Step 3: Implement**

`src/app/tools.py`: change `from .. import edits, maneuver, materials, orbits, units` to `from .. import edits, maneuver, materials, measure, orbits, units`. Add the class after `FlyTool` (Task 5 will extend it):

```python
class MeasureTool(TargetLock, DragPan, Tool):
    """A ruler between two bodies, and a stopwatch for an orbit.

    Numbers he can check against a book: the Sun is 8.3 light-minutes away, a
    year is 365.256 days, and ``a^3 / P^2`` weighs whatever is being orbited.
    The physics is ``src/measure.py``; the timer itself lives on ``AppState``,
    so switching tools does not stop it.
    """

    name = "Measure"
    hint = ("Click a body, then X and a second body to measure between them  -  "
            "P times the selected body all the way round its orbit")
    context_title = "Measure"
    no_target_label = "Measure to..."

    def __init__(self):
        TargetLock.__init__(self)
        DragPan.__init__(self)

    def controls(self, state, metrics):
        selected = state.selected_index is not None
        timing = state.period_timer is not None
        return [
            [Control("measure.target", label=self._target_label(state),
                     icon="target", hotkey="X",
                     active=(self.picking_target
                             or self._target_index(state.sim.world) is not None),
                     tooltip="Click a body to measure the distance to it")],
            [Control("measure.time",
                     label=("Pick a body first" if not selected
                            else "Time it again" if timing else "Time its orbit"),
                     icon="clock", hotkey="P", enabled=selected,
                     tooltip=("Time it all the way round, and check Kepler"
                              if selected else "Click a planet first, then press this")),
             Control("measure.stop", icon="close", compact=True, enabled=timing,
                     tooltip="Stop timing")],
        ]

    def perform(self, control, state):
        cid = control.id
        if cid == "measure.target":
            self.picking_target = not self.picking_target
            return True
        if cid == "measure.time":
            state.start_timer()
            return True
        if cid == "measure.stop":
            state.stop_timer()
            return True
        return False

    def on_event(self, event, state):
        """Keys.  ``P`` relies on ``start_timer``'s own check for a selection,
        because a key never passes through the disabled button's gate."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_x:
                self.picking_target = not self.picking_target
                return True
            if event.key == pygame.K_p:
                state.start_timer()
                return True
        if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self._pick_target(event, state)):
            return True
        return self._pan_event(event, state)

    def _on_click(self, pos, state):
        index = state.pick_body_at(pos[0], pos[1])
        if index is not None:
            state.select(state.sim.world.bodies[index].id)
        else:
            state.selected_id = None
```

In `TOOLS` add `pygame.K_5: MeasureTool,` after `pygame.K_4: FlyTool,`, and change the `TOOL_BY_NAME` comment from "The same four tools" to "The same tools".

`src/app/ui.py`, `TOOL_BUTTONS`: after the `tool.fly` row add

```python
    ("tool.measure", "Measure", "measure", "5", "Measure a distance, time an orbit"),
```

`src/render/widgets.py`, `draw_icon`: after the `"fly"` branch add

```python
    elif name == "measure":
        # A ruler: a bar with ticks along its top edge, every other one longer.
        bar = pygame.Rect(x, cy - h // 4, w, max(3, h // 2))
        pygame.draw.rect(surface, colour, bar, 1)
        for k in range(1, 5):
            tx = x + k * w // 5
            tick = bar.height // 2 if k % 2 == 0 else bar.height // 3
            pygame.draw.line(surface, colour, (tx, bar.top), (tx, bar.top + tick), 1)
    elif name == "clock":
        pygame.draw.aacircle(surface, colour, (cx, cy), r, 1)
        pygame.draw.line(surface, colour, (cx, cy), (cx, cy - r + 2), 1)
        pygame.draw.line(surface, colour, (cx, cy), (cx + r // 2, cy), 1)
```

`src/render/hud.py`, `KEY_HELP`: replace

```python
    ("1 2 3 4", "Look / Throw / Edit / Fly"),
```

with

```python
    ("1 2 3 4 5", "Look / Throw / Edit / Fly / Measure"),
```

Replace `("X", "lock a target (Throw, Fly)"),` with `("X", "lock a target (Throw, Fly, Measure)"),`. After the `N` row add

```python
    ("P", "time the selected body's orbit (Measure)"),
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `./.venv/bin/python -m pytest tests/test_tools_measure.py tests/test_ui.py tests/test_layout.py -q`
Expected: all pass. The layout test now builds the Measure panel at every window size and at both densities. Measured headroom at the smallest window: the hotbar needs 152 of 480 px with the new button. The help table already overflows the smallest window by 9–11 rows and is clipped. That problem predates this plan; one more row doesn't change it.

- [ ] **Step 5: Mutation check.** Delete the `measure.time` branch in `perform`, then the `K_p` branch in `on_event`, then each icon branch. Confirm the named tests fail each time. Revert.

- [ ] **Step 6: Commit**

```bash
git add src/app/tools.py src/app/ui.py src/render/widgets.py src/render/hud.py \
        tests/test_tools_measure.py tests/test_ui.py tests/test_layout.py
git commit -m "feat: a Measure tool on key 5 -- lock a target, time an orbit"
```

---

### Task 5: The overlay, a line between the two bodies and the readout

**Files:**
- Modify: `src/app/tools.py` (`MeasureTool`: add `GAP_SHOWN_BELOW_AU`, `readout_lines`, `_timer_lines` and `draw_overlay`)
- Test: `tests/test_tools_measure.py` (append)

**Interfaces:**
- Consumes: `measure.separation`, `measure.kepler_period` and `PeriodTimer` (Task 1); `hud.format_light_time` and `hud.format_period` (Task 2).
- Produces: `MeasureTool.readout_lines(state) -> list[tuple[str, tuple]]`, as `(text, rgb)`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_tools_measure.py`)

```python
def _two_body_state():
    pygame.init()
    sim = Simulation(world=scenarios.two_body())
    app = AppState(sim, Camera(*SIZE), ViewSettings())
    app.set_tool(MeasureTool)
    return app


def _texts(state):
    return [text for text, _ in state.tool.readout_lines(state)]


def test_nothing_to_say_with_nothing_chosen(state):
    assert state.tool.readout_lines(state) == []


def test_the_ruler_reads_light_minutes_to_the_sun():
    """One AU exactly, from the two-body scenario: 8.3 light-minutes."""
    st = _two_body_state()
    st.select(st.sim.world.bodies[1].id)
    st.tool.target_id = st.sim.world.bodies[0].id
    texts = _texts(st)
    assert texts[0] == "Secondary to Primary"
    assert "Apart  1 AU" in texts
    assert "Light takes 8.3 min" in texts
    assert not any(t.startswith("Gap") for t in texts)      # too far to matter


def test_the_gap_appears_once_they_are_close():
    # Earth to the Moon is well inside 0.02 AU.
    pygame.init()
    sim = Simulation(world=scenarios.solar_system())
    st = AppState(sim, Camera(*SIZE), ViewSettings())
    st.set_tool(MeasureTool)
    st.select(_body(st, "Earth").id)
    st.tool.target_id = _body(st, "Moon").id
    assert any(t.startswith("Gap") and t.endswith("km") for t in _texts(st))


def test_the_timer_reads_measured_against_kepler():
    """After one lap: the measured period, Kepler's, how far apart, and the
    mass it implies.  Removing any of the four lines fails this."""
    st = _two_body_state()
    st.select(st.sim.world.bodies[1].id)
    st.start_timer()
    while st.period_timer.measured is None and st.sim.world.time < 1000.0:
        st.sim.advance(5.0, budget_s=1e9)
        st.sample_timer()
    texts = _texts(st)
    assert texts[0] == "Timing Secondary round Primary"
    assert "Measured    365.3 d" in texts
    assert "Kepler says 365.3 d" in texts
    assert any(t.startswith("Off by 0.0") for t in texts)
    assert "a^3/P^2 weighs 1 Suns" in texts


def test_too_fast_is_said_out_loud():
    st = _two_body_state()
    st.select(st.sim.world.bodies[1].id)
    st.start_timer()
    st.sim.advance(200.0, budget_s=1e9)
    st.sample_timer()
    assert "Too fast to time -- slow down" in _texts(st)


def test_a_line_is_drawn_between_the_two():
    """A pixel a quarter of the way along the line is lit only when there is
    a target.  Deleting the aaline call makes this fail."""
    from src.render.pixelfont import PixelFont
    st = _two_body_state()
    st.show_body_list = False
    st.camera.log_zoom = math.log(SIZE[0] / 3.0)     # pixels per AU: 3 AU across
    st.camera.centre[:] = 0.0
    st.select(st.sim.world.bodies[1].id)
    st.tool.target_id = st.sim.world.bodies[0].id
    world = st.sim.world
    a = st.camera.world_to_screen(world.pos[1])
    b = st.camera.world_to_screen(world.pos[0])
    probe = (int(round(a[0] + 0.25 * (b[0] - a[0]))), int(round(a[1] + 0.25 * (b[1] - a[1]))))
    # A probe off screen would make this test pass or fail for no reason.
    assert pygame.Rect(0, 0, *SIZE).collidepoint(probe)
    surface = pygame.Surface(SIZE)
    surface.fill((0, 0, 0))
    st.tool.draw_overlay(surface, st, PixelFont(12))
    lit = max(surface.get_at((probe[0] + dx, probe[1] + dy))[:3]
              for dx in (-1, 0, 1) for dy in (-1, 0, 1))
    assert sum(lit) > 150
```

Add `import math` to the top of `tests/test_tools_measure.py`. `Camera` keeps `centre` (British spelling) and `log_zoom`, where `exp(log_zoom)` is pixels per AU.

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `./.venv/bin/python -m pytest tests/test_tools_measure.py -q`
Expected: the new tests fail with `AttributeError: 'MeasureTool' object has no attribute 'readout_lines'`, and the line test fails because the inherited `draw_overlay` draws nothing.

- [ ] **Step 3: Implement** (inside `MeasureTool`)

```python
    #: The surface gap only differs from the centre distance in a digit anyone
    #: reads once both are in kilometres -- the same 0.02 AU at which
    #: ``hud.format_length`` switches unit.
    GAP_SHOWN_BELOW_AU = 0.02

    def readout_lines(self, state):
        """What the overlay says, as ``(text, colour)``: the ruler, then the timer.

        Returned rather than drawn so a test can read the words.
        """
        world = state.sim.world
        lines = []
        here = state.selected_index
        there = self._target_index(world)
        if here is not None and there is not None and here != there:
            s = measure.separation(world, here, there)
            lines.append(("%s to %s" % (world.bodies[here].name,
                                        world.bodies[there].name), palette.SELECT))
            lines.append(("Apart  %s" % hud.format_length(s["centres"]), palette.TEXT))
            if s["centres"] < self.GAP_SHOWN_BELOW_AU:
                lines.append(("Gap    %s" % hud.format_length(s["gap"]), palette.TEXT))
            lines.append(("Light takes %s" % hud.format_light_time(s["light_days"]),
                          palette.TEXT_DIM))
            verb = "Closing" if s["closing"] >= 0.0 else "Opening"
            lines.append(("%s at %s" % (verb, hud.format_speed(abs(s["closing"]))),
                          palette.TEXT_DIM))
        if state.period_timer is not None:
            lines.extend(self._timer_lines(world, state.period_timer))
        return lines

    @staticmethod
    def _timer_lines(world, timer):
        index = world.index_of(timer.body_id)
        if index is None:
            return [("The timed body is gone", palette.WARN)]
        name = world.bodies[index].name
        attractor = (world.index_of(timer.attractor_id)
                     if timer.attractor_id is not None else None)
        if attractor is None:
            return [("%s is not orbiting anything" % name, palette.WARN)]
        around = world.bodies[attractor].name
        lines = [("Timing %s round %s" % (name, around), palette.SELECT)]
        if timer.status == "too fast":
            lines.append(("Too fast to time -- slow down", palette.WARN))
        elif timer.status == "moved on":
            lines.append(("It left %s behind" % around, palette.WARN))
        elif timer.status == "not in orbit":
            lines.append(("Not in orbit -- it will never come back", palette.WARN))
        else:
            lines.append(("%d%% round" % int(timer.fraction_round * 100),
                          palette.TEXT_DIM))
        measured = timer.measured
        kepler, _ = measure.kepler_period(world, index, attractor)
        if measured is not None:
            lines.append(("Measured    %s" % hud.format_period(measured), palette.TEXT))
        if kepler is not None:
            lines.append(("Kepler says %s" % hud.format_period(kepler), palette.TEXT))
        if measured is not None and kepler is not None:
            lines.append(("Off by %.2f%%" % (100.0 * abs(measured / kepler - 1.0)),
                          palette.TEXT_DIM))
        mass = timer.mass_from_orbit(world)
        if mass is not None:
            lines.append(("a^3/P^2 weighs %s" % hud.format_mass(mass),
                          palette.AIM_GOOD))
        return lines

    def draw_overlay(self, surface, state, font):
        world = state.sim.world
        camera = state.camera
        m = widgets.Metrics(font, *surface.get_size())
        here = state.selected_index
        there = self._target_index(world)
        if there is not None:
            tx, ty = camera.world_to_screen(world.pos[there])
            pygame.draw.aacircle(surface, palette.SELECT, (int(tx), int(ty)),
                                 max(3, m.btn_h // 4), 1)
        if here is not None and there is not None and here != there:
            ends = camera.world_to_screen_array(world.pos[[here, there]])
            # Clamped before the cast, as FlyTool._stroke does: a zoomed-in
            # view puts a far body at 1e12 px, past the C int pygame wants.
            np.clip(ends, -32000, 32000, out=ends)
            (ax, ay), (bx, by) = ends.astype(int).tolist()
            pygame.draw.aaline(surface, palette.SELECT, (ax, ay), (bx, by))
        x = widgets.overlay_x(m, state.show_body_list)
        # Below the explain panel when one is up, as the Fly readout does.
        panel = state.explain_rect
        y = m.content_top if panel is None else panel.bottom + m.gap
        step = font.height + 2
        for text, colour in self.readout_lines(state):
            font.draw(surface, x, y, text, colour)
            y += step
```

`camera.world_to_screen_array` returns a fresh `np.empty` float array, so clipping it in place is safe.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `./.venv/bin/python -m pytest tests/test_tools_measure.py tests/test_smoke.py -q`
Expected: all pass. `test_pressing_every_button_in_the_real_app_keeps_it_running` in `test_smoke.py` now presses the Measure buttons too.

- [ ] **Step 5: Mutation check.** Delete the `aaline` call, then the `Measured` append, then the `mass_from_orbit` block. Confirm a test fails each time. Revert.

- [ ] **Step 6: Commit**

```bash
git add src/app/tools.py tests/test_tools_measure.py
git commit -m "feat: the Measure readout -- distance, light time, and a lap against Kepler"
```

---

### Task 6: Update the docs, and run the whole suite

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Edit `CLAUDE.md`**

1. In the Architecture table's `src/*.py` cell, after the `save.py` clause, add: `and \`measure.py\` (the ruler between two bodies and the orbit timer that checks Kepler's third law)`.
2. In the `step_once` frame description, change `→ sim.update_trails →` to `→ sim.update_trails → state.sample_timer →`.
3. Under **Invariants**, add:

```markdown
- **The orbit timer measures by angle swept, sampled once a frame, and refuses
  when a frame is more than a quarter orbit** (`src/measure.py`). Past that the
  angle between two samples cannot be unwrapped -- measured, 200-day frames on a
  one-year orbit reported a 600-day lap. The lap's end is interpolated between
  the two samples straddling it (0.035% at 7-day frames against 1.6% without).
  The timer lives on `AppState`, not the tool, so it survives a tool switch;
  Restart and Load drop it, and it is not saved.
```

4. Under **Adding a tool**, fix the claim that "the buttons follow automatically". Replace it with: "register it in `TOOLS`, keyed by pygame key constant (`TOOL_BY_NAME` is derived from that dict), and add its row to `ui.TOOL_BUTTONS` -- the hotbar reads that table, not `TOOLS`. Add it to the tool lists in `tests/test_layout.py` and the parity test in `tests/test_ui.py`."
5. In **Commands**, update the test count to whatever Step 2 reports.

- [ ] **Step 2: Run the whole suite** (Bash `timeout: 300000`)

Run: `./.venv/bin/python -m pytest tests/ -q`
Expected: every test passes. The count should be 1284 plus the new tests.

- [ ] **Step 3: Launch the app and look**

Run: `./run.command`. Press `5`, click Earth, press `X` and click the Sun. The readout should say about 8.2 light-minutes, because Earth is near perihelion at J2000. Press `P` and set the speed to one year a second. Once the lap completes, the readout should show a period of about 365.3 d, a Kepler value within a few hundredths of a percent, and "a^3/P^2 weighs 1 Suns". Then pick Io at the default speed and confirm "Too fast to time -- slow down".

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: the Measure tool, the orbit timer's rules, and how a tool reaches the hotbar"
```
