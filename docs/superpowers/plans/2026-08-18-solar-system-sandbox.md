# Solar System Sandbox Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the rendering, interaction and collision layers that turn the existing headless gravity core into a playable 2D pixel-art solar system sandbox for an 8-year-old.

**Architecture:** Three layers. `src/*.py` is pure numpy physics that never imports pygame (asserted by test). `src/render/` owns every pixel. `src/app/` owns the frame loop, input routing, and one class per interaction tool. Bodies are addressed by stable **id**, never by array index, because `World.remove` swap-removes.

**Tech Stack:** Python 3.11+, numpy >= 1.26, pygame >= 2.5, pytest.

**Spec:** `docs/superpowers/specs/2026-08-18-solar-system-sandbox-design.md`

## Global Constraints

- Simulation units are **AU, solar masses, days**. `src/units.py` is the only place a constant or conversion factor may be defined. No other module hard-codes one.
- **No module under `src/` (top level) may import pygame.** Only `src/render/` and `src/app/` may. Task 1 enforces this with a test.
- **Never hold an array index across a frame.** Selection, camera follow, and collision bookkeeping store `body.id` and resolve via `World.index_of(body_id)`, which returns `None` if the body is gone.
- `World.MAX_BODIES` is 200 and must not be raised; the O(n^2) kernel is budgeted for it.
- Internal render buffer is `display_size // S`, default `S = 2`. All pixel-size constants scale by `u = internal_height / 360`.
- Every mutation of mass, radius or membership is followed by `integrator.refresh(world)`.
- Angles in radians everywhere except the source data tables in `scenarios.py`.
- Tests live in `tests/`, run by pytest. `selftest.py` at repo root is a thin runner so the whole suite is one command.

---

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `src/units.py` | Constants, unit conversion, sphere geometry | exists |
| `src/body.py` | Body identity, trails | exists, +1 slot |
| `src/materials.py` | Seven materials, densities, tones | exists |
| `src/world.py` | SoA state, swap-remove, id map | exists |
| `src/integrator.py` | Verlet, block timesteps, force kernel | exists |
| `src/orbits.py` | Kepler, elements <-> state, attractor | exists |
| `src/scenarios.py` | J2000 solar system | exists |
| `src/collisions.py` | Swept detection, merge, shatter | **new** |
| `src/sim.py` | Simulation: budgeted advance, trails | **new** |
| `src/render/palette.py` | Theme colours | **new** |
| `src/render/pixelfont.py` | Non-antialiased text | **new** |
| `src/render/camera.py` | world <-> screen, zoom, pan, follow | **new** |
| `src/render/scale.py` | Display-radius law | **new** |
| `src/render/sprites.py` | Procedural discs, LRU cache | **new** |
| `src/render/starfield.py` | Parallax background | **new** |
| `src/render/draw.py` | World rendering | **new** |
| `src/render/hud.py` | Panels, bars, help overlay | **new** |
| `src/app/config.py` | Resolution derivation, display setup | **new** |
| `src/app/state.py` | AppState, time-scale ladder | **new** |
| `src/app/tools.py` | Tool base, Select, Launch, Edit | **new** |
| `src/app/input.py` | Event routing, keybindings | **new** |
| `src/app/main.py` | Frame loop, entry point | **new** |
| `selftest.py` | One-command test runner | **new** |
| `run.command` | Double-clickable launcher | **new** |

**Milestones:** Task 6 = physics complete and proven headless. Task 12 = the solar system is on screen and navigable. Task 16 = things collide and shatter. Task 19 = shippable.

---

## Task 1: Repo scaffolding and the import-discipline rule

**Files:**
- Create: `.gitignore`, `selftest.py`, `tests/__init__.py`, `tests/test_layering.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: nothing
- Produces: `selftest.py` runnable as `python selftest.py`; pytest available.

- [ ] **Step 1: Initialise the repository**

```bash
git init
git add -A
git commit -m "chore: existing physics core"
```

- [ ] **Step 2: Add .gitignore**

```
__pycache__/
*.pyc
.venv/
venv/
.pytest_cache/
.DS_Store
```

- [ ] **Step 3: Add pytest to requirements.txt**

The file currently has no trailing newline after `numpy>=1.26`. Rewrite it whole:

```
pygame>=2.5
numpy>=1.26
pytest>=8.0
```

- [ ] **Step 4: Write the failing test**

`tests/test_layering.py`:

```python
"""The physics layer must stay headless.

If a physics module ever imports pygame, the orbital mechanics stop being
testable without a display and the whole test suite becomes hostage to SDL.
This test is the enforcement.
"""

import importlib
import subprocess
import sys

PHYSICS_MODULES = [
    "src.units", "src.body", "src.materials", "src.world",
    "src.integrator", "src.orbits", "src.scenarios",
    "src.collisions", "src.sim",
]


def test_physics_layer_never_imports_pygame():
    # Run in a subprocess so an earlier test importing pygame cannot mask this.
    code = (
        "import sys;"
        "mods=%r;"
        "[__import__(m) for m in mods];"
        "assert 'pygame' not in sys.modules, sorted(sys.modules)"
    ) % (PHYSICS_MODULES,)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_every_physics_module_imports():
    for name in PHYSICS_MODULES:
        importlib.import_module(name)
```

- [ ] **Step 5: Run it and watch it fail**

Run: `python -m pytest tests/test_layering.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.collisions'`. That is correct; those two modules arrive in Tasks 3 and 6.

- [ ] **Step 6: Create the two module stubs so the layering test can go green**

`src/collisions.py`:

```python
"""Collision detection and resolution.  Filled in by Tasks 3-5."""
```

`src/sim.py`:

```python
"""Simulation driver.  Filled in by Task 6."""
```

- [ ] **Step 7: Write selftest.py**

```python
#!/usr/bin/env python3
"""Run the whole test suite in one command."""

import sys

import pytest

if __name__ == "__main__":
    sys.exit(pytest.main(["-q", "tests"]))
```

- [ ] **Step 8: Run and verify it passes**

Run: `python selftest.py`
Expected: PASS, 2 passed.

- [ ] **Step 9: Commit**

```bash
git add .gitignore requirements.txt selftest.py tests src/collisions.py src/sim.py
git commit -m "test: enforce headless physics layer; add pytest harness"
```

---

## Task 2: Regression tests for the existing physics core

Nothing here changes production code. It pins down the behaviour the rest of the
plan depends on, and it is the only proof that the orbital mechanics inherited
from the previous session are actually correct.

**Files:**
- Create: `tests/test_orbits.py`, `tests/test_integrator.py`, `tests/test_world.py`

**Interfaces:**
- Consumes: `src.orbits`, `src.integrator`, `src.world`, `src.scenarios`, `src.units`
- Produces: nothing new; establishes the numeric tolerances later tasks reuse.

- [ ] **Step 1: Write the Kepler round-trip test**

`tests/test_orbits.py`:

```python
import math

import pytest

from src import orbits, units


@pytest.mark.parametrize("a,e,lon_peri,nu", [
    (1.0, 0.0, 0.0, 0.0),
    (1.5, 0.2, 0.7, 1.3),
    (5.2, 0.048, 0.257, -2.1),
    (30.1, 0.0086, 0.785, 3.0),
])
def test_elements_state_roundtrip(a, e, lon_peri, nu):
    mu = units.G * 1.0
    pos, vel = orbits.state_from_elements(mu, a, e, lon_peri, nu)
    el = orbits.elements_from_state(mu, pos, vel)
    assert el["a"] == pytest.approx(a, rel=1e-10)
    assert el["e"] == pytest.approx(e, abs=1e-10)
    assert el["bound"] is True
    # Angles wrap, so compare their difference reduced to (-pi, pi].
    d = (el["true_anomaly"] - nu + math.pi) % (2 * math.pi) - math.pi
    assert d == pytest.approx(0.0, abs=1e-9)


def test_kepler_solver_matches_definition():
    """M = E - e sin E must hold for the E implied by the returned true anomaly."""
    e = 0.35
    for mean in (0.0, 0.5, 1.9, 3.0, 5.5):
        nu = orbits.true_anomaly_from_mean(mean, e)
        big_e = 2.0 * math.atan2(
            math.sqrt(1.0 - e) * math.sin(nu * 0.5),
            math.sqrt(1.0 + e) * math.cos(nu * 0.5),
        )
        recovered = big_e - e * math.sin(big_e)
        d = (recovered - mean + math.pi) % (2 * math.pi) - math.pi
        assert d == pytest.approx(0.0, abs=1e-10)


def test_dominant_attractor_of_moon_is_earth():
    from src import scenarios
    world = scenarios.solar_system()
    names = [b.name for b in world.bodies]
    moon = names.index("Moon")
    assert names[orbits.dominant_attractor(world, moon)] == "Earth"


def test_dominant_attractor_of_earth_is_sun():
    from src import scenarios
    world = scenarios.solar_system()
    names = [b.name for b in world.bodies]
    earth = names.index("Earth")
    assert names[orbits.dominant_attractor(world, earth)] == "Sun"
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_orbits.py -v`
Expected: PASS. If `test_dominant_attractor_of_moon_is_earth` fails, stop — `orbits.dominant_attractor` is broken and every orbit readout in the HUD depends on it.

- [ ] **Step 3: Write the integrator tests**

`tests/test_integrator.py`:

```python
import numpy as np
import pytest

from src import scenarios, units
from src.integrator import DT_BASE, Integrator


def _run(world, days, integrator=None):
    integrator = integrator or Integrator(world.capacity)
    integrator.refresh(world)
    done = 0.0
    while done < days:
        integrator.choose_k(world)
        dt = min(integrator.dt, days - done)
        integrator.step(world, dt)
        done += dt
    return integrator


def test_two_body_conserves_energy_over_100_orbits():
    """Velocity Verlet exists to stop planets spiralling in.  This is that test."""
    world = scenarios.two_body(separation=1.0, secondary_mass=units.EARTH_MASS)
    integrator = Integrator(world.capacity)
    integrator.refresh(world)
    e0 = world.total_energy()
    _run(world, 100 * units.DAYS_PER_YEAR, integrator)
    drift = abs((world.total_energy() - e0) / e0)
    assert drift < 1e-6, "energy drift %.3g" % drift


def test_earth_returns_to_its_start_after_one_year():
    world = scenarios.solar_system(moons=False)
    names = [b.name for b in world.bodies]
    earth = names.index("Earth")
    start = world.pos[earth].copy()
    _run(world, units.DAYS_PER_YEAR)
    offset = np.linalg.norm(world.pos[earth] - start)
    # 0.5% of the orbit's circumference.
    assert offset < 0.005 * 2 * np.pi, "Earth missed its start by %.4f AU" % offset


def test_neptune_year_is_165_years():
    from src import orbits
    world = scenarios.solar_system(moons=False)
    names = [b.name for b in world.bodies]
    nep = names.index("Neptune")
    sun = names.index("Sun")
    mu = units.G * (world.mass[sun] + world.mass[nep])
    el = orbits.elements_from_state(
        mu, world.pos[nep] - world.pos[sun], world.vel[nep] - world.vel[sun])
    years = el["period"] / units.DAYS_PER_YEAR
    assert years == pytest.approx(164.8, rel=0.005)


def test_choose_k_moves_at_most_one_notch():
    world = scenarios.solar_system(moons=True)
    integrator = Integrator(world.capacity)
    integrator.refresh(world)
    previous = integrator.k
    for _ in range(20):
        k = integrator.choose_k(world)
        assert abs(k - previous) <= 1
        previous = k


def test_timestep_is_always_a_power_of_two_fraction():
    world = scenarios.solar_system()
    integrator = Integrator(world.capacity)
    integrator.refresh(world)
    integrator.choose_k(world)
    ratio = DT_BASE / integrator.dt
    assert ratio == pytest.approx(round(ratio))
    assert (round(ratio) & (round(ratio) - 1)) == 0


def test_momentum_is_conserved_by_stepping():
    world = scenarios.solar_system(moons=False)
    p0 = world.total_momentum().copy()
    _run(world, 500.0)
    assert np.allclose(world.total_momentum(), p0, atol=1e-14)
```

- [ ] **Step 4: Run it**

Run: `python -m pytest tests/test_integrator.py -v`
Expected: PASS. The 100-orbit test takes a few seconds; that is expected.

- [ ] **Step 5: Write the world/id-discipline tests**

`tests/test_world.py`:

```python
import numpy as np

from src import materials, units
from src.world import World


def _add(world, name, mass=units.EARTH_MASS, x=1.0):
    return world.add(name, materials.get("rock"), mass, units.EARTH_RADIUS,
                     np.array([x, 0.0]), np.zeros(2))


def test_ids_survive_swap_removes():
    world = World()
    bodies = [_add(world, "b%d" % i, x=float(i + 1)) for i in range(5)]
    world.remove(1)          # swap-removes: last row moves into slot 1
    world.remove(0)
    alive = [b for b in bodies if world.index_of(b.id) is not None]
    assert len(alive) == 3
    for body in alive:
        i = world.index_of(body.id)
        assert world.bodies[i] is body, "id map desynchronised from rows"


def test_removed_ids_resolve_to_none():
    world = World()
    a = _add(world, "a", x=1.0)
    b = _add(world, "b", x=2.0)
    world.remove(world.index_of(a.id))
    assert world.index_of(a.id) is None
    assert world.index_of(b.id) is not None


def test_remove_many_is_order_independent():
    world = World()
    bodies = [_add(world, "b%d" % i, x=float(i + 1)) for i in range(6)]
    keep = {bodies[0].id, bodies[3].id}
    world.remove_many([1, 2, 4, 5])
    assert world.n == 2
    assert {b.id for b in world.bodies} == keep


def test_positions_view_tracks_n():
    world = World()
    _add(world, "a")
    assert world.positions.shape == (1, 2)
    _add(world, "b", x=2.0)
    assert world.positions.shape == (2, 2)


def test_solar_system_barycentre_is_at_rest():
    from src import scenarios
    world = scenarios.solar_system()
    assert np.allclose(world.total_momentum(), 0.0, atol=1e-18)
```

- [ ] **Step 6: Run the whole suite**

Run: `python selftest.py`
Expected: PASS, all tests green.

- [ ] **Step 7: Commit**

```bash
git add tests
git commit -m "test: pin down orbital mechanics, energy conservation and id discipline"
```

---
## Task 3: Swept collision detection

A rock thrown hard covers more than a planet's diameter in one substep, so an
endpoint distance check misses it entirely and the rock sails through Jupiter.
Detection is therefore a closest-approach test along the segment each pair
travelled during the step.

**Files:**
- Create: `tests/test_collision_detect.py`
- Modify: `src/collisions.py`

**Interfaces:**
- Consumes: `World`, `src.units`
- Produces:
  - `Collision` dataclass with fields `i: int` (larger mass), `j: int` (smaller), `s: float` (fraction of the step at closest approach, 0..1), `v_rel: float` (relative speed, AU/day)
  - `find_collisions(world, prev_pos) -> list[Collision]`, sorted by `s` ascending

- [ ] **Step 1: Write the failing tests**

`tests/test_collision_detect.py`:

```python
import numpy as np

from src import collisions, materials, units
from src.world import World

ROCK = materials.get("rock")


def _world_with(pairs):
    """pairs: list of (mass, radius, pos, vel)."""
    world = World()
    for k, (mass, radius, pos, vel) in enumerate(pairs):
        world.add("b%d" % k, ROCK, mass, radius, np.array(pos, float), np.array(vel, float))
    return world


def _stepped(world, dt):
    """Return prev_pos and advance positions linearly by dt (no gravity)."""
    prev = world.pos.copy()
    world.pos[:world.n] += world.vel[:world.n] * dt
    return prev


def test_separated_bodies_do_not_collide():
    r = units.EARTH_RADIUS
    world = _world_with([
        (units.EARTH_MASS, r, (0.0, 0.0), (0.0, 0.0)),
        (units.EARTH_MASS, r, (1.0, 0.0), (0.0, 0.0)),
    ])
    prev = _stepped(world, 1.0)
    assert collisions.find_collisions(world, prev) == []


def test_overlapping_bodies_collide():
    r = units.EARTH_RADIUS
    world = _world_with([
        (units.EARTH_MASS, r, (0.0, 0.0), (0.0, 0.0)),
        (units.EARTH_MASS, r, (r, 0.0), (0.0, 0.0)),
    ])
    prev = _stepped(world, 1.0)
    hits = collisions.find_collisions(world, prev)
    assert len(hits) == 1


def test_fast_body_passing_clean_through_is_still_caught():
    """The anti-tunnelling case: it starts far left, ends far right, and at no
    point is either endpoint inside the planet."""
    r = units.EARTH_RADIUS
    speed = 200.0 * r          # covers 200 Earth radii in one step
    world = _world_with([
        (units.EARTH_MASS, r, (0.0, 0.0), (0.0, 0.0)),
        (1e-12, r * 0.01, (-100.0 * r, 0.0), (speed, 0.0)),
    ])
    prev = _stepped(world, 1.0)
    assert np.linalg.norm(world.pos[1] - world.pos[0]) > 50 * r  # ended far away
    hits = collisions.find_collisions(world, prev)
    assert len(hits) == 1, "fast body tunnelled through the planet"
    assert hits[0].i == 0, "larger mass must be recorded first"
    assert hits[0].j == 1


def test_near_miss_is_not_a_hit():
    r = units.EARTH_RADIUS
    world = _world_with([
        (units.EARTH_MASS, r, (0.0, 0.0), (0.0, 0.0)),
        (1e-12, r * 0.01, (-100.0 * r, 3.0 * r), (200.0 * r, 0.0)),
    ])
    prev = _stepped(world, 1.0)
    assert collisions.find_collisions(world, prev) == []


def test_hits_are_sorted_earliest_first():
    r = units.EARTH_RADIUS
    world = _world_with([
        (units.EARTH_MASS, r, (0.0, 0.0), (0.0, 0.0)),
        (1e-12, r * 0.01, (-10.0 * r, 0.0), (20.0 * r, 0.0)),   # arrives late
        (1e-12, r * 0.01, (-2.0 * r, 0.0), (20.0 * r, 0.0)),    # arrives early
    ])
    prev = _stepped(world, 1.0)
    hits = collisions.find_collisions(world, prev)
    assert len(hits) >= 2
    assert hits[0].s <= hits[1].s


def test_same_event_fragments_ignore_each_other():
    r = units.EARTH_RADIUS
    world = _world_with([
        (units.EARTH_MASS, r, (0.0, 0.0), (0.0, 0.0)),
        (units.EARTH_MASS, r, (r, 0.0), (0.0, 0.0)),
    ])
    for body in world.bodies:
        body.event_id = 7
        body.no_collide_until = 5.0
    world.time = 1.0
    prev = _stepped(world, 1.0)
    assert collisions.find_collisions(world, prev) == []


def test_event_immunity_expires():
    r = units.EARTH_RADIUS
    world = _world_with([
        (units.EARTH_MASS, r, (0.0, 0.0), (0.0, 0.0)),
        (units.EARTH_MASS, r, (r, 0.0), (0.0, 0.0)),
    ])
    for body in world.bodies:
        body.event_id = 7
        body.no_collide_until = 5.0
    world.time = 9.0
    prev = _stepped(world, 1.0)
    assert len(collisions.find_collisions(world, prev)) == 1
```

- [ ] **Step 2: Run and watch it fail**

Run: `python -m pytest tests/test_collision_detect.py -v`
Expected: FAIL — `module 'src.collisions' has no attribute 'find_collisions'`.

- [ ] **Step 3: Implement detection**

Replace the contents of `src/collisions.py`:

```python
"""Collision detection and resolution.

**Detection is swept, not point-in-time.**  A body thrown at 200 Earth radii per
step is outside the planet at the start of the step and outside it at the end;
only a test along the path it travelled sees the hit.  For each pair we minimise
|r0 + s*d| over s in [0, 1], where r0 is the relative position at the start of
the step and d the relative displacement during it.  Exact for linear motion,
one vectorised O(n^2) pass, and it cannot be fooled by speed.

Fragments born of the same shattering share an ``event_id`` and ignore each
other until ``no_collide_until``; without that they re-merge on the next substep
and the planet appears to flicker and reassemble.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Collision:
    #: Index of the more massive body.  Resolution always treats this as the target.
    i: int
    #: Index of the less massive body.
    j: int
    #: Fraction of the step at closest approach, 0..1.  Used only for ordering.
    s: float
    #: Relative speed at the end of the step, AU/day.
    v_rel: float


def find_collisions(world, prev_pos):
    """Every pair whose swept paths came within the sum of their radii.

    ``prev_pos`` is the position array as it was before the step.  Returns a list
    of :class:`Collision` sorted earliest first, with the more massive body in
    ``i``.  Pairs sharing a live ``event_id`` are excluded.
    """
    n = world.n
    if n < 2:
        return []

    p0 = prev_pos[:n]
    p1 = world.pos[:n]
    step = p1 - p0

    # r0[i, j] = p0[j] - p0[i];  d[i, j] = step[j] - step[i]
    r0 = p0[None, :, :] - p0[:, None, :]
    d = step[None, :, :] - step[:, None, :]

    dd = np.einsum("ijk,ijk->ij", d, d)
    rd = np.einsum("ijk,ijk->ij", r0, d)

    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(dd > 0.0, -rd / dd, 0.0)
    np.clip(s, 0.0, 1.0, out=s)

    closest = r0 + s[:, :, None] * d
    dist2 = np.einsum("ijk,ijk->ij", closest, closest)

    radii = world.radius[:n]
    touch = radii[:, None] + radii[None, :]
    hit = dist2 < touch * touch
    np.fill_diagonal(hit, False)
    hit = np.triu(hit)

    rows, cols = np.nonzero(hit)
    if rows.size == 0:
        return []

    mass = world.mass[:n]
    vel = world.vel[:n]
    bodies = world.bodies
    now = world.time
    out = []
    for a, b in zip(rows.tolist(), cols.tolist()):
        ba, bb = bodies[a], bodies[b]
        if (ba.event_id and ba.event_id == bb.event_id
                and now < ba.no_collide_until):
            continue
        big, small = (a, b) if mass[a] >= mass[b] else (b, a)
        dv = vel[small] - vel[big]
        out.append(Collision(big, small, float(s[a, b]),
                             float(np.hypot(dv[0], dv[1]))))

    out.sort(key=lambda c: c.s)
    return out
```

- [ ] **Step 4: Run and verify it passes**

Run: `python -m pytest tests/test_collision_detect.py -v`
Expected: PASS, 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/collisions.py tests/test_collision_detect.py
git commit -m "feat: swept collision detection that cannot be tunnelled"
```

---

## Task 4: Merging

**Files:**
- Create: `tests/test_collision_merge.py`
- Modify: `src/collisions.py`

**Interfaces:**
- Consumes: `Collision` from Task 3, `src.units`, `src.body.KIND_STAR`
- Produces:
  - `impact_energy(world, c) -> float`
  - `binding_of_target(world, c) -> float`
  - `is_catastrophic(world, c) -> bool`
  - `merge(world, c) -> int` — returns the surviving body's **current index** after the removal reshuffles rows

- [ ] **Step 1: Write the failing tests**

`tests/test_collision_merge.py`:

```python
import numpy as np
import pytest

from src import collisions, materials, units
from src.body import KIND_STAR
from src.world import World

ROCK = materials.get("rock")


def _pair(m1, r1, v1, m2, r2, v2, gap):
    world = World()
    world.add("big", ROCK, m1, r1, np.zeros(2), np.array(v1, float))
    world.add("small", ROCK, m2, r2, np.array([gap, 0.0]), np.array(v2, float))
    return world


def test_merge_conserves_mass_and_momentum_exactly():
    r = units.EARTH_RADIUS
    world = _pair(units.EARTH_MASS, r, (0.1, 0.0),
                  units.MOON_MASS, r * 0.27, (-0.05, 0.02), r)
    m0 = world.masses.sum()
    p0 = world.total_momentum().copy()

    c = collisions.Collision(0, 1, 0.5, 0.15)
    collisions.merge(world, c)

    assert world.n == 1
    assert world.masses.sum() == pytest.approx(m0, rel=1e-15)
    assert np.allclose(world.total_momentum(), p0, rtol=1e-14, atol=1e-24)


def test_merged_radius_follows_from_mass_and_mean_density():
    r = units.EARTH_RADIUS
    world = _pair(units.EARTH_MASS, r, (0.0, 0.0),
                  units.EARTH_MASS, r, (0.0, 0.0), r)
    d0 = units.density_from_mass_radius(units.EARTH_MASS, r)

    collisions.merge(world, collisions.Collision(0, 1, 0.5, 0.0))

    total = 2 * units.EARTH_MASS
    expected = units.radius_from_mass_density(total, d0)
    assert world.radius[0] == pytest.approx(expected, rel=1e-12)
    assert world.radius[0] > r, "merging two Earths must make something bigger"


def test_merge_keeps_the_larger_body_identity():
    r = units.EARTH_RADIUS
    world = _pair(units.EARTH_MASS, r, (0.0, 0.0),
                  units.MOON_MASS, r * 0.27, (0.0, 0.0), r)
    survivor_id = world.bodies[0].id
    idx = collisions.merge(world, collisions.Collision(0, 1, 0.5, 0.0))
    assert world.bodies[idx].id == survivor_id
    assert world.bodies[idx].name == "big"


def test_merge_returns_a_valid_index_after_the_swap_remove():
    """remove() swap-removes, so the survivor's row can move.  The returned
    index must be the one you can actually use."""
    world = World()
    for k in range(4):
        world.add("b%d" % k, ROCK, units.EARTH_MASS * (k + 1), units.EARTH_RADIUS,
                  np.array([k * 5.0, 0.0]), np.zeros(2))
    survivor_id = world.bodies[1].id
    idx = collisions.merge(world, collisions.Collision(1, 0, 0.5, 0.0))
    assert world.bodies[idx].id == survivor_id
    assert world.index_of(survivor_id) == idx


def test_crossing_the_ignition_threshold_makes_a_star():
    half = units.STAR_MASS_THRESHOLD * 0.6
    r = units.SUN_RADIUS * 0.3
    world = _pair(half, r, (0.0, 0.0), half, r, (0.0, 0.0), r)
    idx = collisions.merge(world, collisions.Collision(0, 1, 0.5, 0.0))
    assert world.mass[idx] > units.STAR_MASS_THRESHOLD
    assert world.bodies[idx].kind == KIND_STAR


def test_gentle_touch_is_not_catastrophic():
    r = units.EARTH_RADIUS
    world = _pair(units.EARTH_MASS, r, (0.0, 0.0),
                  units.MOON_MASS, r * 0.27, (1e-5, 0.0), r)
    c = collisions.Collision(0, 1, 0.5, 1e-5)
    assert collisions.is_catastrophic(world, c) is False


def test_hypervelocity_impact_is_catastrophic():
    r = units.EARTH_RADIUS
    # 500 km/s: far above Earth's 11.2 km/s escape velocity.
    fast = 500.0 / units.AUD_TO_KMS
    world = _pair(units.EARTH_MASS, r, (0.0, 0.0),
                  units.EARTH_MASS * 0.1, r * 0.5, (fast, 0.0), r)
    c = collisions.Collision(0, 1, 0.5, fast)
    assert collisions.is_catastrophic(world, c) is True
```

- [ ] **Step 2: Run and watch it fail**

Run: `python -m pytest tests/test_collision_merge.py -v`
Expected: FAIL — `has no attribute 'merge'`.

- [ ] **Step 3: Implement the energy test and merge**

Append to `src/collisions.py`:

```python
from . import units
from .body import KIND_DEBRIS, KIND_PLANET, KIND_STAR


def impact_energy(world, c):
    """Kinetic energy of the impact in the pair's centre-of-mass frame.

    Using the reduced mass is what makes this the energy actually available to do
    damage: the part of the kinetic energy that is not just the pair drifting.
    """
    mi, mj = world.mass[c.i], world.mass[c.j]
    total = mi + mj
    if total <= 0.0:
        return 0.0
    reduced = mi * mj / total
    return 0.5 * reduced * c.v_rel * c.v_rel


def binding_of_target(world, c):
    """Gravitational binding energy of the larger body: what holds it together."""
    return units.binding_energy(world.mass[c.i], world.radius[c.i])


def is_catastrophic(world, c):
    """True when the impact carries more energy than the target's self-gravity."""
    return bool(impact_energy(world, c) >= binding_of_target(world, c))


def merge(world, c):
    """Combine the pair into the larger body.  Returns its index afterwards.

    Mass and momentum are conserved exactly.  Density becomes the mass-weighted
    mean of the two, and the radius then follows from mass and density, so the
    result is a physically sized sphere rather than a made-up number.
    """
    i, j = c.i, c.j
    mi, mj = float(world.mass[i]), float(world.mass[j])
    total = mi + mj

    momentum = mi * world.vel[i] + mj * world.vel[j]
    centre = (mi * world.pos[i] + mj * world.pos[j]) / total

    di = units.density_from_mass_radius(mi, float(world.radius[i]))
    dj = units.density_from_mass_radius(mj, float(world.radius[j]))
    density = (mi * di + mj * dj) / total

    survivor = world.bodies[i]
    world.pos[i] = centre
    world.vel[i] = momentum / total
    world.mass[i] = total
    world.radius[i] = units.radius_from_mass_density(total, density)

    if total >= units.STAR_MASS_THRESHOLD:
        survivor.kind = KIND_STAR
    elif survivor.kind == KIND_DEBRIS and total >= units.EARTH_MASS * 0.01:
        survivor.kind = KIND_PLANET

    world.remove(j)
    # remove() swap-removes, so i may no longer be the survivor's row.
    return world.index_of(survivor.id)
```

- [ ] **Step 4: Run and verify it passes**

Run: `python -m pytest tests/test_collision_merge.py -v`
Expected: PASS, 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/collisions.py tests/test_collision_merge.py
git commit -m "feat: merge collisions conserving mass, momentum and density"
```

---
## Task 5: Shattering

The payoff feature. Note the honesty constraint: fragment kinetic energy is
invented to look plausible, not derived from a fracture model. **Momentum is
conserved exactly, and that is what the tests assert.**

**Files:**
- Create: `tests/test_collision_shatter.py`
- Modify: `src/collisions.py`, `src/body.py` (add `rings` slot)

**Interfaces:**
- Consumes: everything from Tasks 3-4
- Produces:
  - `CollisionResolver(seed=1234)` with `.rng`, `.resolve(world, integrator, prev_pos) -> int`
  - `CollisionResolver.shatter(world, c) -> list[int]`
  - Module constants `MIN_FRAGMENTS = 6`, `MAX_FRAGMENTS = 12`

- [ ] **Step 1: Add the `rings` slot to Body**

`src/body.py`, in `Body.__slots__`, add `"rings"`, and in `__init__` add
`self.rings = False`. Saturn needs it in Task 11 and adding it now avoids a
second edit to a physics module later.

```python
    __slots__ = ("id", "name", "kind", "material", "trail", "born_day",
                 "event_id", "no_collide_until", "seed", "rings")
```

and at the end of `__init__`:

```python
        #: Drawn with a ring system.  Set by scenarios for Saturn.
        self.rings = False
```

- [ ] **Step 2: Write the failing tests**

`tests/test_collision_shatter.py`:

```python
import numpy as np
import pytest

from src import collisions, materials, units
from src.body import KIND_DEBRIS
from src.world import World

ROCK = materials.get("rock")


def _impact(target_mass=units.EARTH_MASS, ratio=0.1, speed_kms=500.0):
    r = units.EARTH_RADIUS
    world = World()
    world.add("Target", ROCK, target_mass, r, np.zeros(2), np.zeros(2))
    world.add("Bullet", ROCK, target_mass * ratio, r * 0.5,
              np.array([r, 0.0]), np.array([-speed_kms / units.AUD_TO_KMS, 0.0]))
    c = collisions.Collision(0, 1, 0.5, speed_kms / units.AUD_TO_KMS)
    return world, c


def test_shatter_conserves_mass_exactly():
    world, c = _impact()
    m0 = world.masses.sum()
    resolver = collisions.CollisionResolver(seed=1)
    resolver.shatter(world, c)
    assert world.masses.sum() == pytest.approx(m0, rel=1e-12)


def test_shatter_conserves_momentum_exactly():
    world, c = _impact()
    p0 = world.total_momentum().copy()
    resolver = collisions.CollisionResolver(seed=1)
    resolver.shatter(world, c)
    assert np.allclose(world.total_momentum(), p0, rtol=1e-10,
                       atol=1e-12 * abs(p0).max())


def test_fragment_count_is_within_bounds():
    world, c = _impact()
    resolver = collisions.CollisionResolver(seed=1)
    idx = resolver.shatter(world, c)
    assert collisions.MIN_FRAGMENTS <= len(idx) <= collisions.MAX_FRAGMENTS
    assert world.n == len(idx)


def test_fragments_share_an_event_id_and_an_immunity_window():
    world, c = _impact()
    resolver = collisions.CollisionResolver(seed=1)
    resolver.shatter(world, c)
    event_ids = {b.event_id for b in world.bodies}
    assert len(event_ids) == 1
    assert 0 not in event_ids
    for b in world.bodies:
        assert b.no_collide_until > world.time


def test_fragments_do_not_immediately_recollide():
    """The immunity window is the difference between a shattering and a flicker."""
    world, c = _impact()
    resolver = collisions.CollisionResolver(seed=1)
    resolver.shatter(world, c)
    prev = world.pos.copy()
    world.pos[:world.n] += world.vel[:world.n] * 1e-4
    assert collisions.find_collisions(world, prev) == []


def test_fragments_are_debris():
    world, c = _impact()
    resolver = collisions.CollisionResolver(seed=1)
    resolver.shatter(world, c)
    assert all(b.kind == KIND_DEBRIS for b in world.bodies)


def test_one_fragment_is_clearly_the_remnant():
    world, c = _impact()
    resolver = collisions.CollisionResolver(seed=1)
    resolver.shatter(world, c)
    masses = sorted(world.masses.tolist(), reverse=True)
    assert masses[0] >= 0.25 * sum(masses), "no dominant remnant"


def test_fragment_radii_follow_from_mass_and_density():
    world, c = _impact()
    density = units.density_from_mass_radius(units.EARTH_MASS, units.EARTH_RADIUS)
    resolver = collisions.CollisionResolver(seed=1)
    resolver.shatter(world, c)
    for k in range(world.n):
        actual = units.density_from_mass_radius(world.mass[k], world.radius[k])
        # All fragments share the mean density of the two parents.
        assert actual == pytest.approx(density, rel=0.5)


def test_shatter_falls_back_to_merge_when_the_world_is_full():
    world, c = _impact()
    # Pack the world so fewer than MIN_FRAGMENTS slots remain.
    while world.n < world.capacity - 2:
        world.add("filler", ROCK, 1e-14, units.EARTH_RADIUS * 0.01,
                  np.array([100.0 + world.n, 0.0]), np.zeros(2))
    before = world.n
    resolver = collisions.CollisionResolver(seed=1)
    resolver.shatter(world, c)
    assert world.n == before - 1, "should have merged, not shattered"


def test_resolver_returns_event_count_and_refreshes_the_integrator():
    from src.integrator import Integrator
    world, c = _impact()
    integrator = Integrator(world.capacity)
    integrator.refresh(world)
    prev = world.pos.copy()
    world.pos[:world.n] += world.vel[:world.n] * 1.0
    resolver = collisions.CollisionResolver(seed=1)
    events = resolver.resolve(world, integrator, prev)
    assert events == 1
    # Accelerations must match the post-collision configuration.
    expected = np.zeros_like(world.acc)
    integrator.accelerations(world.pos, world.mass, world.n, expected)
    assert np.allclose(world.acc[:world.n], expected[:world.n])
```

- [ ] **Step 3: Run and watch it fail**

Run: `python -m pytest tests/test_collision_shatter.py -v`
Expected: FAIL — `has no attribute 'CollisionResolver'`.

- [ ] **Step 4: Implement the resolver**

Append to `src/collisions.py`:

```python
#: Fewer than this and a shattering reads as a bad merge rather than a break-up.
MIN_FRAGMENTS = 6
#: More than this and the world fills with debris the O(n^2) kernel cannot afford.
MAX_FRAGMENTS = 12

#: Fragments are placed on a circle this many target-radii out, so they start
#: clear of one another rather than interpenetrating.
FRAGMENT_RING = 1.15
#: Immunity lasts long enough for fragments to travel this many target radii.
IMMUNITY_RADII = 5.0


class CollisionResolver:
    """Owns the RNG and the event counter for collision outcomes.

    Kept as an object rather than module functions so that tests are
    deterministic and independent: two resolvers with the same seed produce the
    same debris field, and no global state leaks between tests.
    """

    def __init__(self, seed=1234):
        self.rng = np.random.default_rng(seed)
        self._next_event_id = 1

    # --- outcomes --------------------------------------------------------------

    def shatter(self, world, c):
        """Break the pair into fragments.  Returns their indices.

        Falls back to :func:`merge` when there is not room for a credible number
        of fragments -- a two-piece "shattering" looks like a bug.
        """
        i, j = c.i, c.j
        mi, mj = float(world.mass[i]), float(world.mass[j])
        total = mi + mj
        r_big = float(world.radius[i])

        momentum = mi * world.vel[i] + mj * world.vel[j]
        centre = (mi * world.pos[i] + mj * world.pos[j]) / total
        drift = momentum / total

        di = units.density_from_mass_radius(mi, r_big)
        dj = units.density_from_mass_radius(mj, float(world.radius[j]))
        density = (mi * di + mj * dj) / total

        # How badly did the impact overshoot what held the target together?
        binding = binding_of_target(world, c)
        excess = impact_energy(world, c) / binding if binding > 0.0 else 1.0
        excess = max(excess, 1.0)

        wanted = int(MIN_FRAGMENTS + 2.0 * np.log2(excess))
        room = world.capacity - (world.n - 2)
        count = int(min(wanted, MAX_FRAGMENTS, room))
        if count < MIN_FRAGMENTS:
            # A two-piece "shattering" reads as a bug, not a break-up.
            merge(world, c)
            return []

        parent = world.bodies[i]
        material = parent.material
        event_id = self._next_event_id
        self._next_event_id += 1

        # Mass split: one dominant remnant, the rest a random partition.
        remnant_share = float(np.clip(0.50 - 0.08 * np.log2(excess), 0.30, 0.50))
        weights = self.rng.random(count - 1) + 0.15
        weights *= (1.0 - remnant_share) / weights.sum()
        shares = np.concatenate(([remnant_share], weights))
        masses = shares * total

        kick = units.escape_velocity(total, r_big)
        angles = (np.arange(count) / count) * 2.0 * np.pi
        angles += self.rng.normal(0.0, 0.25, count)
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        speeds = kick * self.rng.uniform(0.55, 1.45, count)
        # The remnant keeps most of the mass, so give it least of the kick.
        speeds[0] *= 0.35

        positions = centre + directions * (FRAGMENT_RING * r_big)
        velocities = drift + directions * speeds[:, None]

        # Correct so total momentum is exactly what it was before the impact.
        error = (masses[:, None] * velocities).sum(axis=0) - momentum
        velocities -= error / total

        world.remove_many([i, j])

        made = []
        for k in range(count):
            mass = float(masses[k])
            body = world.add(
                "Fragment", material, mass,
                units.radius_from_mass_density(mass, density),
                positions[k], velocities[k],
                kind=KIND_DEBRIS, seed=int(self.rng.integers(0, 1 << 30)),
            )
            if body is None:
                break
            body.event_id = event_id
            body.no_collide_until = world.time + IMMUNITY_RADII * r_big / max(kick, 1e-12)
            made.append(world.index_of(body.id))
        return made

    # --- driver ----------------------------------------------------------------

    def resolve(self, world, integrator, prev_pos):
        """Detect and resolve every collision from the step just taken.

        Returns the number of events.  Indices shift as bodies are removed, so
        only the first collision of a batch is trusted directly; the rest are
        re-detected on the next call, which is one substep later and invisible.
        """
        hits = find_collisions(world, prev_pos)
        if not hits:
            return 0
        c = hits[0]
        if is_catastrophic(world, c):
            self.shatter(world, c)
        else:
            merge(world, c)
        integrator.refresh(world)
        return 1
```

- [ ] **Step 5: Run and verify it passes**

Run: `python -m pytest tests/test_collision_shatter.py -v`
Expected: PASS, 10 passed.

If `test_shatter_conserves_momentum_exactly` fails, the momentum correction is
wrong — check that `masses.sum()` equals `total` before the correction is applied.

- [ ] **Step 6: Run the whole suite**

Run: `python selftest.py`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/collisions.py src/body.py tests/test_collision_shatter.py
git commit -m "feat: shattering with exact momentum conservation and fragment immunity"
```

---

## Task 6: The Simulation driver

**Files:**
- Create: `src/sim.py`, `tests/test_sim.py`

**Interfaces:**
- Consumes: `World`, `Integrator`, `CollisionResolver`, `orbits.dominant_attractor`
- Produces:
  - `Simulation(world=None, seed=1234)` with attributes `world`, `integrator`, `resolver`, `paused` (bool), `last_ratio` (float), `events` (int, collisions in the last advance)
  - `Simulation.advance(days, budget_s=0.012) -> float` — returns days actually simulated
  - `Simulation.update_trails() -> None`
  - `Simulation.refresh() -> None`
  - `Simulation.reset(scenario="solar_system") -> None`

- [ ] **Step 1: Write the failing tests**

`tests/test_sim.py`:

```python
import time

import numpy as np
import pytest

from src import units
from src.sim import Simulation


def test_advance_returns_the_days_it_managed():
    sim = Simulation()
    done = sim.advance(1.0, budget_s=1.0)
    assert done == pytest.approx(1.0, rel=1e-9)
    assert sim.world.time == pytest.approx(1.0, rel=1e-9)


def test_paused_simulation_does_not_move():
    sim = Simulation()
    sim.paused = True
    before = sim.world.pos.copy()
    assert sim.advance(10.0) == 0.0
    assert np.array_equal(sim.world.pos, before)


def test_wall_clock_budget_is_respected():
    """Without this the window freezes at high time scales and reads as a crash."""
    sim = Simulation()
    start = time.perf_counter()
    done = sim.advance(50_000.0, budget_s=0.02)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, "budget ignored, took %.3fs" % elapsed
    assert done < 50_000.0
    assert sim.last_ratio < 1.0


def test_ratio_is_one_when_the_budget_is_ample():
    sim = Simulation()
    sim.advance(0.5, budget_s=5.0)
    assert sim.last_ratio == pytest.approx(1.0, rel=1e-9)


def test_trails_record_points_as_bodies_move():
    sim = Simulation()
    earth = next(b for b in sim.world.bodies if b.name == "Earth")
    assert earth.trail.count == 0
    for _ in range(40):
        sim.advance(3.0, budget_s=5.0)
        sim.update_trails()
    assert earth.trail.count > 3
    assert earth.trail.ordered().shape[1] == 2


def test_reset_restores_the_solar_system():
    sim = Simulation()
    sim.advance(100.0, budget_s=5.0)
    sim.world.remove(sim.world.n - 1)
    sim.reset()
    assert sim.world.time == 0.0
    assert any(b.name == "Neptune" for b in sim.world.bodies)
    assert np.allclose(sim.world.total_momentum(), 0.0, atol=1e-18)


def test_saturn_has_rings():
    sim = Simulation()
    saturn = next(b for b in sim.world.bodies if b.name == "Saturn")
    assert saturn.rings is True


def test_energy_is_stable_over_a_long_run():
    sim = Simulation()
    e0 = sim.world.total_energy()
    for _ in range(200):
        sim.advance(20.0, budget_s=5.0)
    drift = abs((sim.world.total_energy() - e0) / e0)
    assert drift < 1e-5, "energy drift %.3g" % drift
```

- [ ] **Step 2: Run and watch it fail**

Run: `python -m pytest tests/test_sim.py -v`
Expected: FAIL — `cannot import name 'Simulation'`.

- [ ] **Step 3: Mark Saturn's rings in the scenario**

In `src/scenarios.py`, inside `solar_system`, after the planet loop, add:

```python
    world.bodies[index_by_name["Saturn"]].rings = True
```

- [ ] **Step 4: Implement Simulation**

Replace `src/sim.py`:

```python
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
```

- [ ] **Step 5: Run and verify it passes**

Run: `python -m pytest tests/test_sim.py -v`
Expected: PASS, 8 passed.

- [ ] **Step 6: Run the whole suite — this is the Milestone A gate**

Run: `python selftest.py`
Expected: PASS, everything green. The physics is now complete and proven, with
no display involved anywhere.

- [ ] **Step 7: Commit**

```bash
git add src/sim.py src/scenarios.py tests/test_sim.py
git commit -m "feat: Simulation driver with wall-clock budget and trail sampling"
```

---
## Task 7: Display bootstrap — and the Retina question

**This task answers the one open risk in the spec.** Everything downstream sizes
itself from the number this reports, so do it before writing any rendering code.

**Files:**
- Create: `src/render/__init__.py`, `src/app/__init__.py`, `src/app/config.py`, `src/app/main.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: pygame
- Produces:
  - `internal_size(display_w, display_h, s) -> (int, int)`
  - `font_px(s) -> int`
  - `DisplayContext` with attributes `surface` (the internal draw surface), `width`, `height`, `s`, `physical` (tuple)
  - `open_display(s=2, fullscreen=True) -> DisplayContext`
  - `DisplayContext.set_grid(s) -> None`

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py`:

```python
import pytest

from src.app.config import font_px, internal_size


@pytest.mark.parametrize("w,h,s,expected", [
    (3456, 2234, 2, (1728, 1117)),
    (3456, 2234, 3, (1152, 744)),
    (3456, 2234, 4, (864, 558)),
    (1920, 1080, 2, (960, 540)),
])
def test_internal_size_divides_exactly(w, h, s, expected):
    assert internal_size(w, h, s) == expected


def test_internal_size_never_returns_zero():
    assert internal_size(10, 10, 4) == (2, 2)


def test_font_stays_legible_at_every_grid():
    """Text must stay ~24 physical pixels tall whatever the pixel grid, or a
    fine grid would render the inspector panel unreadable."""
    for s in (2, 3, 4):
        physical = font_px(s) * s
        assert 18 <= physical <= 36, "s=%d gives %dpx text" % (s, physical)


def test_font_never_smaller_than_eight_internal_pixels():
    for s in (2, 3, 4, 6):
        assert font_px(s) >= 8
```

- [ ] **Step 2: Run and watch it fail**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — `No module named 'src.app'`.

- [ ] **Step 3: Create the packages**

`src/render/__init__.py` and `src/app/__init__.py` — both containing only:

```python
"""Pixel layer."""
```

and

```python
"""Application layer."""
```

- [ ] **Step 4: Implement config.py**

`src/app/config.py`:

```python
"""Display setup and the constants that depend on the pixel grid.

The internal buffer is always ``display // S`` for integer S, so the upscale is
exact, the aspect ratio is preserved, and nothing is letterboxed.  Drawing small
and letting SDL present big is what keeps the pixel grid coherent; doing the
upscale in software would cost several milliseconds a frame at 7.7 megapixels.
"""

import pygame

#: Default pixel grid.  S=2 on a 16" MacBook Pro gives a 1728x1117 buffer:
#: high-resolution pixel art rather than retro-chunky.
DEFAULT_S = 2
GRID_CHOICES = (2, 3, 4)

#: Target text height in *physical* pixels, held constant across grids.
TARGET_TEXT_PHYSICAL_PX = 24
MIN_FONT_PX = 8


def internal_size(display_w, display_h, s):
    """Internal buffer size for a display and pixel-grid divisor."""
    return (max(1, display_w // s), max(1, display_h // s))


def font_px(s):
    """Font height in internal pixels, chosen so physical size stays constant."""
    return max(MIN_FONT_PX, round(TARGET_TEXT_PHYSICAL_PX / s))


class DisplayContext:
    """The window plus the internal surface everything draws into."""

    def __init__(self, s=DEFAULT_S, fullscreen=True):
        self.fullscreen = fullscreen
        self.physical = self._desktop_size()
        self.s = s
        self.surface = None
        self._open()

    @staticmethod
    def _desktop_size():
        sizes = pygame.display.get_desktop_sizes()
        return sizes[0] if sizes else (1920, 1080)

    def _open(self):
        width, height = internal_size(self.physical[0], self.physical[1], self.s)
        flags = pygame.SCALED
        if self.fullscreen:
            flags |= pygame.FULLSCREEN
        self.surface = pygame.display.set_mode((width, height), flags, vsync=1)
        self.width, self.height = self.surface.get_size()

    def set_grid(self, s):
        """Change the pixel grid live.  Callers must clear any sprite cache."""
        self.s = s
        self._open()

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self._open()


def open_display(s=DEFAULT_S, fullscreen=True):
    pygame.display.init()
    pygame.display.set_caption("Solar System Sandbox")
    return DisplayContext(s, fullscreen)
```

- [ ] **Step 5: Run and verify the tests pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS, 4 passed.

- [ ] **Step 6: Write the Retina probe**

`src/app/main.py`, first version — it opens a window, prints what it actually
got, and quits on any key:

```python
"""Entry point and frame loop."""

import sys

import pygame

from .config import open_display


def probe():
    """Report what SDL actually gave us.  Answers the Retina question."""
    pygame.init()
    ctx = open_display()
    info = {
        "desktop_size": pygame.display.get_desktop_sizes(),
        "internal_surface": ctx.surface.get_size(),
        "window_size": pygame.display.get_window_size(),
        "driver": pygame.display.get_driver(),
    }
    for key, value in info.items():
        print("%-18s %s" % (key, value))
    pygame.quit()
    return info


def main():
    probe()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7: Run the probe and record the answer**

Run: `python -m src.app.main`

Compare `window_size` against the panel's true resolution:

- On a 16" MacBook Pro the native panel is **3456x2234**.
- If `desktop_size` reports `(3456, 2234)`, SDL is giving us the real backing
  store and `S=2` yields the intended 1728x1117 buffer. Nothing to change.
- If it reports `(1728, 1117)` — logical points — then the effective ceiling is
  half the panel. In that case set `DEFAULT_S = 1` in `config.py`, which gives a
  1728x1117 buffer from a 1728x1117 desktop, and adjust `GRID_CHOICES` to
  `(1, 2, 3)`. Everything downstream is expressed in terms of `internal_height`
  and adapts automatically.

**Write the observed numbers into a comment at the top of `config.py`** so the
next person does not have to re-derive them.

- [ ] **Step 8: Commit**

```bash
git add src/render/__init__.py src/app tests/test_config.py
git commit -m "feat: display bootstrap with grid-independent sizing"
```

---

## Task 8: Palette and text

**Files:**
- Create: `src/render/palette.py`, `src/render/pixelfont.py`, `tests/test_pixelfont.py`

**Interfaces:**
- Consumes: pygame, `src.app.config.font_px`
- Produces:
  - `palette` module constants: `SPACE`, `PANEL`, `PANEL_EDGE`, `TEXT`, `TEXT_DIM`, `ACCENT`, `WARN`, `SELECT`, `TRAIL_BASE`, `STAR_GLOW`, `AIM_GOOD`, `AIM_ESCAPE`
  - `PixelFont(px)` with `.height`, `.width(text) -> int`, `.draw(surface, x, y, text, colour) -> int`

**Deviation from the spec, recorded deliberately:** the spec called for an
embedded 5x7 bitmap font. This uses pygame's built-in font with **antialiasing
off** instead. Rendered into the internal buffer and scaled up by SDL, hard-edged
glyphs read as pixel text, it costs no glyph data to author, and it supports the
full character set the numeric panel needs. `PixelFont` is a narrow interface, so
swapping in a real bitmap font later touches nothing but this file.

- [ ] **Step 1: Write the failing test**

`tests/test_pixelfont.py`:

```python
import os

import pygame
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    pygame.init()
    pygame.display.set_mode((320, 200))
    yield
    pygame.quit()


def test_font_reports_a_sensible_height():
    from src.render.pixelfont import PixelFont
    font = PixelFont(12)
    assert 8 <= font.height <= 24


def test_width_grows_with_text_length():
    from src.render.pixelfont import PixelFont
    font = PixelFont(12)
    assert font.width("XXXXXXXXXX") > font.width("X")
    assert font.width("") == 0


def test_draw_returns_the_advance_and_marks_pixels():
    from src.render import palette
    from src.render.pixelfont import PixelFont
    font = PixelFont(12)
    surface = pygame.Surface((200, 40))
    surface.fill(palette.SPACE)
    advance = font.draw(surface, 2, 2, "Neptune", palette.TEXT)
    assert advance == font.width("Neptune")
    # Something must actually have been drawn.
    assert pygame.transform.average_color(surface)[:3] != palette.SPACE[:3]


def test_text_is_not_antialiased():
    """Antialiased glyphs blur against the pixel grid.  Every drawn pixel must be
    either the background or exactly the text colour."""
    from src.render import palette
    from src.render.pixelfont import PixelFont
    font = PixelFont(12)
    surface = pygame.Surface((120, 30))
    surface.fill((0, 0, 0))
    font.draw(surface, 1, 1, "AVW", (255, 255, 255))
    colours = {surface.get_at((x, y))[:3]
               for x in range(120) for y in range(30)}
    assert colours <= {(0, 0, 0), (255, 255, 255)}, sorted(colours)
```

- [ ] **Step 2: Run and watch it fail**

Run: `python -m pytest tests/test_pixelfont.py -v`
Expected: FAIL — `No module named 'src.render.palette'`.

- [ ] **Step 3: Implement palette.py**

```python
"""Theme colours.

Body colours are *not* here -- they live on the Material, because they are a
property of the substance rather than of the theme.  This module owns only the
chrome: space, panels, text, and the overlays the tools draw.
"""

SPACE = (6, 8, 16)
PANEL = (14, 18, 30)
PANEL_EDGE = (44, 54, 78)
TEXT = (226, 234, 246)
TEXT_DIM = (128, 142, 168)
ACCENT = (108, 196, 232)
WARN = (236, 152, 72)
SELECT = (250, 226, 108)
TRAIL_BASE = (74, 96, 132)
STAR_GLOW = (255, 238, 190)
#: Aiming preview: a bound orbit vs one that leaves forever.
AIM_GOOD = (124, 226, 160)
AIM_ESCAPE = (236, 112, 112)
```

- [ ] **Step 4: Implement pixelfont.py**

```python
"""Text rendering with the antialiasing off.

Antialiased glyphs blur against a pixel grid and break the look the moment SDL
scales the buffer up.  Rendering hard-edged into the internal buffer and letting
the upscale enlarge whole pixels is what makes the text read as pixel art.

The interface is deliberately narrow -- height, width, draw -- so a real bitmap
font can replace the implementation later without touching a single caller.
"""

import pygame


class PixelFont:
    def __init__(self, px):
        if not pygame.font.get_init():
            pygame.font.init()
        self.px = px
        self._font = pygame.font.Font(None, px)
        self.height = self._font.get_height()
        self._cache = {}

    def width(self, text):
        if not text:
            return 0
        return self._font.size(text)[0]

    def render(self, text, colour):
        """Cached, non-antialiased surface for one string in one colour."""
        key = (text, colour)
        surface = self._cache.get(key)
        if surface is None:
            surface = self._font.render(text, False, colour)
            if len(self._cache) > 512:
                self._cache.clear()
            self._cache[key] = surface
        return surface

    def draw(self, target, x, y, text, colour):
        """Blit ``text`` at ``(x, y)``.  Returns the horizontal advance."""
        if not text:
            return 0
        surface = self.render(text, colour)
        target.blit(surface, (x, y))
        return surface.get_width()
```

- [ ] **Step 5: Run and verify it passes**

Run: `python -m pytest tests/test_pixelfont.py -v`
Expected: PASS, 4 passed.

If `test_text_is_not_antialiased` fails, the default font is compositing with
alpha. Render onto a `pygame.SRCALPHA` surface and blit that instead, or call
`surface.set_colorkey((0, 0, 0))` on the rendered glyph surface.

- [ ] **Step 6: Commit**

```bash
git add src/render/palette.py src/render/pixelfont.py tests/test_pixelfont.py
git commit -m "feat: theme palette and non-antialiased pixel text"
```

---
## Task 9: Camera

**Files:**
- Create: `src/render/camera.py`, `tests/test_camera.py`

**Interfaces:**
- Consumes: numpy, `World`
- Produces:
  - `Camera(width, height, au_across=70.0)` with attributes `width`, `height`, `centre` (np array, AU), `log_zoom`, `follow_id`
  - `.zoom -> float` (pixels per AU)
  - `.world_to_screen(p) -> (float, float)`
  - `.world_to_screen_array(points) -> np.ndarray shape (n, 2)`
  - `.screen_to_world(sx, sy) -> np.ndarray shape (2,)`
  - `.zoom_by(steps, sx=None, sy=None) -> None`
  - `.pan_pixels(dx, dy) -> None`
  - `.update_follow(world) -> None`
  - `.frame_body(world, index, radii=8.0) -> None`

- [ ] **Step 1: Write the failing tests**

`tests/test_camera.py`:

```python
import numpy as np
import pytest

from src.render.camera import Camera


def test_centre_of_the_view_maps_to_the_camera_centre():
    cam = Camera(1728, 1117)
    cam.centre = np.array([3.0, -2.0])
    x, y = cam.world_to_screen(cam.centre)
    assert x == pytest.approx(1728 / 2)
    assert y == pytest.approx(1117 / 2)


def test_screen_to_world_inverts_world_to_screen():
    cam = Camera(1728, 1117)
    cam.centre = np.array([1.5, 0.25])
    point = np.array([4.2, -3.7])
    x, y = cam.world_to_screen(point)
    assert np.allclose(cam.screen_to_world(x, y), point)


def test_world_y_points_up_on_screen():
    """Screen y grows downward; world y grows upward.  Getting this backwards
    silently mirrors every orbit."""
    cam = Camera(800, 600)
    _, y_high = cam.world_to_screen(np.array([0.0, 1.0]))
    _, y_low = cam.world_to_screen(np.array([0.0, -1.0]))
    assert y_high < y_low


def test_default_view_spans_the_requested_width_in_au():
    cam = Camera(1400, 900, au_across=70.0)
    left = cam.screen_to_world(0, 450)
    right = cam.screen_to_world(1400, 450)
    assert (right[0] - left[0]) == pytest.approx(70.0, rel=1e-9)


def test_zoom_at_a_point_keeps_that_point_still():
    """Wheel-zoom must be anchored at the cursor, or the view slides away from
    whatever you were looking at."""
    cam = Camera(1728, 1117)
    sx, sy = 400.0, 300.0
    before = cam.screen_to_world(sx, sy)
    cam.zoom_by(3, sx, sy)
    after = cam.screen_to_world(sx, sy)
    assert np.allclose(before, after, atol=1e-9)


def test_zoom_in_increases_scale():
    cam = Camera(800, 600)
    z0 = cam.zoom
    cam.zoom_by(1)
    assert cam.zoom > z0
    cam.zoom_by(-1)
    assert cam.zoom == pytest.approx(z0, rel=1e-12)


def test_pan_moves_the_centre_by_the_right_number_of_au():
    cam = Camera(800, 600)
    start = cam.centre.copy()
    cam.pan_pixels(80, 0)
    assert cam.centre[0] == pytest.approx(start[0] - 80 / cam.zoom)


def test_follow_locks_the_centre_onto_a_body_by_id():
    from src import scenarios
    world = scenarios.solar_system()
    cam = Camera(800, 600)
    jupiter = next(b for b in world.bodies if b.name == "Jupiter")
    cam.follow_id = jupiter.id
    cam.update_follow(world)
    assert np.allclose(cam.centre, world.pos[world.index_of(jupiter.id)])


def test_follow_of_a_destroyed_body_is_dropped_not_crashed():
    from src import scenarios
    world = scenarios.solar_system()
    cam = Camera(800, 600)
    doomed = world.bodies[-1]
    cam.follow_id = doomed.id
    world.remove(world.index_of(doomed.id))
    cam.update_follow(world)          # must not raise
    assert cam.follow_id is None


def test_world_to_screen_array_matches_the_scalar_version():
    cam = Camera(800, 600)
    pts = np.array([[0.0, 0.0], [1.0, 2.0], [-3.0, 0.5]])
    batch = cam.world_to_screen_array(pts)
    for k, p in enumerate(pts):
        assert np.allclose(batch[k], cam.world_to_screen(p))
```

- [ ] **Step 2: Run and watch it fail**

Run: `python -m pytest tests/test_camera.py -v`
Expected: FAIL — `No module named 'src.render.camera'`.

- [ ] **Step 3: Implement camera.py**

```python
"""The mapping between AU and pixels.

Zoom is stored as its logarithm so that a wheel notch is a constant *ratio*
rather than a constant amount -- the only way a single control can span from
Neptune's orbit down to the surface of a moon.

Following stores a body **id**, never an index: World.remove swap-removes, so an
index held across a frame can silently start pointing at a different body.
"""

import math

import numpy as np

#: Ratio per wheel notch.
ZOOM_STEP = 1.18
#: Widest and closest views, in pixels per AU, applied as log bounds.
MIN_ZOOM = 1e-2
MAX_ZOOM = 1e12


class Camera:
    def __init__(self, width, height, au_across=70.0):
        self.width = width
        self.height = height
        self.centre = np.zeros(2, dtype=np.float64)
        self.log_zoom = math.log(width / au_across)
        self.follow_id = None

    # --- zoom -------------------------------------------------------------------

    @property
    def zoom(self):
        """Pixels per AU."""
        return math.exp(self.log_zoom)

    def zoom_by(self, steps, sx=None, sy=None):
        """Zoom by ``steps`` notches, keeping the world point under (sx, sy) fixed."""
        anchor = None
        if sx is not None and sy is not None:
            anchor = self.screen_to_world(sx, sy)

        self.log_zoom += steps * math.log(ZOOM_STEP)
        self.log_zoom = min(max(self.log_zoom, math.log(MIN_ZOOM)), math.log(MAX_ZOOM))

        if anchor is not None:
            # Re-centre so the anchored world point lands back under the cursor.
            moved = self.screen_to_world(sx, sy)
            self.centre += anchor - moved

    # --- panning and following ---------------------------------------------------

    def pan_pixels(self, dx, dy):
        """Drag the view by a screen delta.  Dragging right moves the world right."""
        self.centre[0] -= dx / self.zoom
        self.centre[1] += dy / self.zoom
        self.follow_id = None

    def update_follow(self, world):
        """Snap the centre onto the followed body, or forget it if it is gone."""
        if self.follow_id is None:
            return
        index = world.index_of(self.follow_id)
        if index is None:
            self.follow_id = None
            return
        self.centre[:] = world.pos[index]

    def frame_body(self, world, index, radii=8.0):
        """Centre on a body and zoom so it fills a sensible part of the view."""
        self.centre[:] = world.pos[index]
        span = max(world.radius[index] * radii, 1e-9)
        self.log_zoom = math.log(min(self.height, self.width) * 0.5 / span)

    # --- projection --------------------------------------------------------------

    def world_to_screen(self, point):
        zoom = self.zoom
        return (self.width * 0.5 + (float(point[0]) - self.centre[0]) * zoom,
                self.height * 0.5 - (float(point[1]) - self.centre[1]) * zoom)

    def world_to_screen_array(self, points):
        """Vectorised projection for trails and predicted orbits."""
        zoom = self.zoom
        out = np.empty((len(points), 2), dtype=np.float64)
        out[:, 0] = self.width * 0.5 + (points[:, 0] - self.centre[0]) * zoom
        out[:, 1] = self.height * 0.5 - (points[:, 1] - self.centre[1]) * zoom
        return out

    def screen_to_world(self, sx, sy):
        zoom = self.zoom
        return np.array([
            self.centre[0] + (sx - self.width * 0.5) / zoom,
            self.centre[1] - (sy - self.height * 0.5) / zoom,
        ])
```

- [ ] **Step 4: Run and verify it passes**

Run: `python -m pytest tests/test_camera.py -v`
Expected: PASS, 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/render/camera.py tests/test_camera.py
git commit -m "feat: camera with log zoom, cursor-anchored wheel, id-safe follow"
```

---

## Task 10: The size law

**Files:**
- Create: `src/render/scale.py`, `tests/test_scale.py`

**Interfaces:**
- Consumes: `src.units`
- Produces:
  - `display_radius_px(r_true, zoom, internal_height, real_size=False) -> float`
  - `display_radii_px(radii, zoom, internal_height, real_size=False) -> np.ndarray`
  - Constants `REF_HEIGHT = 360.0`, `FLOOR_BASE = 2.0`, `FLOOR_EXP = 0.35`, `FLOOR_MIN = 1.5`, `FLOOR_MAX = 14.0`

- [ ] **Step 1: Write the failing tests**

`tests/test_scale.py`:

```python
import numpy as np
import pytest

from src import units
from src.render import scale

WIDE_ZOOM = 1728 / 70.0        # whole solar system across a 1728px buffer
H = 1117


def _px(r, zoom=WIDE_ZOOM, real=False):
    return scale.display_radius_px(r, zoom, H, real_size=real)


def test_size_order_is_preserved_when_zoomed_out():
    """The whole point: Jupiter must look bigger than Earth, which must look
    bigger than the Moon, even though all three are sub-pixel at true scale."""
    sun = _px(units.SUN_RADIUS)
    jupiter = _px(units.JUPITER_RADIUS)
    earth = _px(units.EARTH_RADIUS)
    moon = _px(1737.4 / units.AU_KM)
    assert sun > jupiter > earth > moon


def test_nothing_is_smaller_than_a_pixel_and_a_half_unit():
    u = H / scale.REF_HEIGHT
    tiny = _px(1.0 / units.AU_KM)          # a 1 km rock
    assert tiny >= scale.FLOOR_MIN * u


def test_the_floor_is_capped_so_nothing_swallows_the_view():
    """A naive log exaggeration lets big bodies grow without bound until one disc
    covers the screen.  The floor is clamped, so it cannot."""
    u = H / scale.REF_HEIGHT
    # At a zoom where true size is negligible, the floor alone decides.
    assert _px(units.SUN_RADIUS * 1000.0, zoom=1e-6) == pytest.approx(scale.FLOOR_MAX * u)
    assert scale.FLOOR_MAX * u < 0.05 * H


def test_the_sun_stays_a_small_fraction_of_the_view():
    assert _px(units.SUN_RADIUS) < 0.05 * H


def test_true_scale_takes_over_continuously_when_zoomed_in():
    """No mode switch, no pop: as you zoom in, the true size overtakes the floor
    and from then on the drawing is honest."""
    r = units.EARTH_RADIUS
    previous = _px(r, zoom=1e2)
    for exponent in range(3, 12):
        current = _px(r, zoom=10.0 ** exponent)
        assert current >= previous
        previous = current
    # Far enough in, it is exactly true scale.
    huge = 1e11
    assert _px(r, zoom=huge) == pytest.approx(huge * r, rel=1e-12)


def test_real_size_mode_collapses_everything_to_a_dot():
    for r in (units.SUN_RADIUS, units.JUPITER_RADIUS, units.EARTH_RADIUS):
        assert _px(r, real=True) == pytest.approx(1.0)


def test_real_size_mode_is_still_honest_when_zoomed_in():
    r = units.EARTH_RADIUS
    assert _px(r, zoom=1e10, real=True) == pytest.approx(1e10 * r)


def test_floor_scales_with_resolution():
    """Constants tuned at 360px tall must not shrink to specks at 1117."""
    small = scale.display_radius_px(units.EARTH_RADIUS, WIDE_ZOOM, 360)
    large = scale.display_radius_px(units.EARTH_RADIUS, WIDE_ZOOM, 1117)
    assert large == pytest.approx(small * 1117 / 360, rel=1e-9)


def test_vectorised_matches_scalar():
    radii = np.array([units.SUN_RADIUS, units.EARTH_RADIUS, units.JUPITER_RADIUS])
    batch = scale.display_radii_px(radii, WIDE_ZOOM, H)
    for k, r in enumerate(radii):
        assert batch[k] == pytest.approx(_px(r))
```

- [ ] **Step 2: Run and watch it fail**

Run: `python -m pytest tests/test_scale.py -v`
Expected: FAIL — `No module named 'src.render.scale'`.

- [ ] **Step 3: Implement scale.py**

```python
"""How big to draw a body.

At true scale a solar-system view makes Earth about 0.001 pixels across, so some
cheat is compulsory.  This one is a floor rather than a multiplier:

    draw_px = max(zoom * r_true, floor_px(r_true))

The floor rises with real radius to the 0.35 power, so size *ordering* survives
-- the Sun reads bigger than Jupiter, which reads bigger than Earth -- while the
range compresses enormously.  Crucially the floor stays small in absolute terms,
so no body inflates far enough to swallow the orbits around it.  That is the
failure mode of a naive log exaggeration, which would draw Earth's disc a
hundred times wider than the Moon's entire orbit.

Because it is a floor, zooming in is continuous: at some point ``zoom * r_true``
overtakes it and the drawing simply becomes honest.  There is no mode switch and
nothing pops.

Real-size mode drops the floor to a single pixel.  The whole solar system
collapses to dots at once, which is the lesson, and it costs one branch.
"""

import numpy as np

from .. import units

#: The resolution the constants below were tuned at.  Everything scales from it.
REF_HEIGHT = 360.0

FLOOR_BASE = 2.0
FLOOR_EXP = 0.35
FLOOR_MIN = 1.5
FLOOR_MAX = 14.0


def _floor(r_true, unit):
    raw = FLOOR_BASE * unit * (r_true / units.EARTH_RADIUS) ** FLOOR_EXP
    return np.clip(raw, FLOOR_MIN * unit, FLOOR_MAX * unit)


def display_radius_px(r_true, zoom, internal_height, real_size=False):
    """Radius to draw, in internal-buffer pixels."""
    true_px = zoom * r_true
    if real_size:
        return max(true_px, 1.0)
    return float(max(true_px, _floor(r_true, internal_height / REF_HEIGHT)))


def display_radii_px(radii, zoom, internal_height, real_size=False):
    """Vectorised form for drawing every body in one pass."""
    true_px = zoom * np.asarray(radii, dtype=np.float64)
    if real_size:
        return np.maximum(true_px, 1.0)
    return np.maximum(true_px, _floor(np.asarray(radii, dtype=np.float64),
                                      internal_height / REF_HEIGHT))
```

- [ ] **Step 4: Run and verify it passes**

Run: `python -m pytest tests/test_scale.py -v`
Expected: PASS, 9 passed.

If `test_the_sun_stays_a_small_fraction_of_the_view` fails, `FLOOR_BASE` or
`FLOOR_MAX` is too large. Do not raise them to make planets more visible — that
is exactly the trap these two tests exist to catch.

- [ ] **Step 5: Commit**

```bash
git add src/render/scale.py tests/test_scale.py
git commit -m "feat: continuous display-radius law with resolution-relative floor"
```

---
## Task 11: Procedural body sprites

Bodies must be generated, never authored: the child creates them at arbitrary
sizes out of arbitrary materials, so there is no sprite sheet that could cover
the space.

**Files:**
- Create: `src/render/sprites.py`, `tests/test_sprites.py`

**Interfaces:**
- Consumes: pygame, `src.materials.Material`
- Produces:
  - `SpriteCache(limit=512)` with `.get(material, seed, px, lit_angle, is_star) -> pygame.Surface`, `.clear()`, `.__len__()`
  - `LIT_BUCKETS = 16`
  - `ring_surface(px, tones) -> pygame.Surface`

- [ ] **Step 1: Write the failing tests**

`tests/test_sprites.py`:

```python
import math
import os

import pygame
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    pygame.init()
    pygame.display.set_mode((320, 200))
    yield
    pygame.quit()


@pytest.fixture
def cache():
    from src.render.sprites import SpriteCache
    return SpriteCache(limit=32)


def test_sprite_is_square_and_sized_to_the_radius(cache):
    from src import materials
    surf = cache.get(materials.get("rock"), seed=1, px=10, lit_angle=0.0, is_star=False)
    assert surf.get_width() == surf.get_height()
    assert surf.get_width() == 21          # 2*px + 1


def test_sprite_silhouette_is_circular(cache):
    """Detail drawing must never spill outside the disc."""
    from src import materials
    px = 12
    surf = cache.get(materials.get("gas"), seed=3, px=px, lit_angle=0.0, is_star=False)
    centre = px
    for x in range(surf.get_width()):
        for y in range(surf.get_height()):
            alpha = surf.get_at((x, y))[3]
            distance = math.hypot(x - centre, y - centre)
            if distance > px + 1.5:
                assert alpha == 0, "pixel at %d,%d outside the disc" % (x, y)


def test_sprite_centre_is_opaque(cache):
    from src import materials
    surf = cache.get(materials.get("ice"), seed=2, px=8, lit_angle=1.0, is_star=False)
    assert surf.get_at((8, 8))[3] == 255


def test_identical_requests_return_the_same_object(cache):
    from src import materials
    rock = materials.get("rock")
    a = cache.get(rock, seed=5, px=9, lit_angle=0.3, is_star=False)
    b = cache.get(rock, seed=5, px=9, lit_angle=0.3, is_star=False)
    assert a is b


def test_lit_angle_is_bucketed_so_the_cache_survives_camera_motion(cache):
    """A continuous lighting angle would miss the cache every single frame."""
    from src import materials
    from src.render.sprites import LIT_BUCKETS
    rock = materials.get("rock")
    step = 2 * math.pi / LIT_BUCKETS
    a = cache.get(rock, seed=5, px=9, lit_angle=0.0, is_star=False)
    b = cache.get(rock, seed=5, px=9, lit_angle=step * 0.2, is_star=False)
    c = cache.get(rock, seed=5, px=9, lit_angle=step * 1.5, is_star=False)
    assert a is b
    assert a is not c


def test_different_materials_look_different(cache):
    from src import materials
    a = cache.get(materials.get("iron"), seed=1, px=14, lit_angle=0.0, is_star=False)
    b = cache.get(materials.get("water"), seed=1, px=14, lit_angle=0.0, is_star=False)
    assert pygame.image.tostring(a, "RGBA") != pygame.image.tostring(b, "RGBA")


def test_lighting_makes_one_side_brighter(cache):
    """Bodies shade away from the star, so the lit limb must be brighter."""
    from src import materials
    px = 16
    surf = cache.get(materials.get("rock"), seed=1, px=px, lit_angle=0.0, is_star=False)
    # lit_angle 0 means the light comes from +x, so the right limb is lit.
    right = sum(surf.get_at((px + int(px * 0.7), px))[:3])
    left = sum(surf.get_at((px - int(px * 0.7), px))[:3])
    assert right > left


def test_stars_are_larger_than_their_disc_because_of_the_glow(cache):
    from src import materials
    hydrogen = materials.get("hydrogen")
    plain = cache.get(hydrogen, seed=1, px=10, lit_angle=0.0, is_star=False)
    star = cache.get(hydrogen, seed=1, px=10, lit_angle=0.0, is_star=True)
    assert star.get_width() > plain.get_width()


def test_cache_evicts_and_stays_bounded(cache):
    from src import materials
    rock = materials.get("rock")
    for px in range(1, 60):
        cache.get(rock, seed=1, px=px, lit_angle=0.0, is_star=False)
    assert len(cache) <= 32


def test_tiny_bodies_still_produce_a_surface(cache):
    from src import materials
    for px in (0, 1, 2):
        surf = cache.get(materials.get("rock"), seed=1, px=px,
                         lit_angle=0.0, is_star=False)
        assert surf.get_width() >= 1
```

- [ ] **Step 2: Run and watch it fail**

Run: `python -m pytest tests/test_sprites.py -v`
Expected: FAIL — `No module named 'src.render.sprites'`.

- [ ] **Step 3: Implement sprites.py**

```python
"""Procedurally drawn pixel bodies.

Nothing here is authored art.  The child creates bodies at arbitrary radii out of
arbitrary materials, so there is no sprite sheet that could cover the space --
every disc is generated on demand and cached.

Shading is three offset circles rather than per-pixel lighting: a dark base, a
mid-tone circle pushed toward the light, and a small light-tone highlight pushed
further.  Three draw calls instead of tens of thousands of Python-level pixel
operations, which matters because a zoom can request a 300px body.

The lighting direction is bucketed to 16 compass points.  A continuous angle
would miss the cache on every frame the camera moves, which would mean
regenerating every sprite, every frame.
"""

import math
from collections import OrderedDict

import pygame

#: Lighting directions the cache distinguishes.
LIT_BUCKETS = 16

#: How far the mid-tone and highlight circles shift toward the light, as a
#: fraction of the radius.
MID_OFFSET = 0.22
HIGHLIGHT_OFFSET = 0.50
HIGHLIGHT_SCALE = 0.62

#: Below this many pixels there is no room for texture; draw a flat dot.
DETAIL_THRESHOLD = 4
#: Glow radius as a multiple of a star's disc radius.
GLOW_SCALE = 2.6


def _shade(tone, factor):
    return tuple(max(0, min(255, int(c * factor))) for c in tone)


def _draw_detail(surface, material, rng, px, centre):
    """Style-specific surface texture, drawn inside the disc."""
    dark, mid, light = material.tones
    style = material.style

    if style == "cratered":
        for _ in range(rng.integers(3, 9)):
            angle = rng.random() * 2 * math.pi
            distance = rng.random() ** 0.5 * px * 0.78
            radius = max(1, int(px * rng.uniform(0.08, 0.20)))
            x = int(centre + math.cos(angle) * distance)
            y = int(centre + math.sin(angle) * distance)
            pygame.draw.circle(surface, _shade(dark, 1.15), (x, y), radius)

    elif style == "banded":
        y = -px
        while y < px:
            thickness = max(1, int(px * rng.uniform(0.10, 0.22)))
            half = int(math.sqrt(max(0.0, px * px - y * y)))
            tone = _shade(mid, rng.uniform(0.78, 1.22))
            pygame.draw.rect(surface, tone,
                             (centre - half, centre + y, 2 * half, thickness))
            y += thickness + max(1, int(px * rng.uniform(0.04, 0.12)))

    elif style == "cracked":
        for _ in range(rng.integers(2, 6)):
            angle = rng.random() * 2 * math.pi
            length = px * rng.uniform(0.5, 1.4)
            x0 = centre + math.cos(angle) * px * 0.6
            y0 = centre + math.sin(angle) * px * 0.6
            x1 = x0 - math.cos(angle + 0.7) * length
            y1 = y0 - math.sin(angle + 0.7) * length
            pygame.draw.line(surface, light, (int(x0), int(y0)), (int(x1), int(y1)), 1)

    elif style == "turbulent":
        for _ in range(rng.integers(6, 14)):
            angle = rng.random() * 2 * math.pi
            distance = rng.random() ** 0.5 * px * 0.8
            radius = max(1, int(px * rng.uniform(0.10, 0.26)))
            x = int(centre + math.cos(angle) * distance)
            y = int(centre + math.sin(angle) * distance)
            tone = light if rng.random() < 0.5 else _shade(mid, 0.85)
            pygame.draw.circle(surface, tone, (x, y), radius)


def _mask_to_circle(surface, px, centre):
    """Clip everything to the disc, so no detail can spill past the silhouette."""
    mask = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    mask.fill((0, 0, 0, 0))
    pygame.draw.circle(mask, (255, 255, 255, 255), (centre, centre), px)
    surface.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)


def _make_body(material, seed, px, lit_angle, is_star):
    import numpy as np

    px = max(1, int(px))
    dark, mid, light = material.tones

    if is_star:
        size = int(px * GLOW_SCALE) * 2 + 1
        centre = size // 2
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        # Glow: concentric rings fading outward, blended additively at draw time.
        steps = max(4, px)
        for k in range(steps, 0, -1):
            radius = int(px + (px * (GLOW_SCALE - 1.0)) * (k / steps))
            alpha = int(80 * (1.0 - k / steps) ** 2)
            pygame.draw.circle(surface, light + (alpha,), (centre, centre), radius)
        pygame.draw.circle(surface, light, (centre, centre), px)
        pygame.draw.circle(surface, (255, 255, 255), (centre, centre),
                           max(1, int(px * 0.72)))
        return surface

    size = px * 2 + 1
    centre = px
    surface = pygame.Surface((size, size), pygame.SRCALPHA)

    pygame.draw.circle(surface, dark, (centre, centre), px)
    if px >= 2:
        dx = math.cos(lit_angle) * px * MID_OFFSET
        dy = -math.sin(lit_angle) * px * MID_OFFSET
        pygame.draw.circle(surface, mid, (int(centre + dx), int(centre + dy)), px)

    if px >= DETAIL_THRESHOLD:
        rng = np.random.default_rng(seed if seed else 1)
        _draw_detail(surface, material, rng, px, centre)
        hx = math.cos(lit_angle) * px * HIGHLIGHT_OFFSET
        hy = -math.sin(lit_angle) * px * HIGHLIGHT_OFFSET
        pygame.draw.circle(surface, light,
                           (int(centre + hx), int(centre + hy)),
                           max(1, int(px * (1.0 - HIGHLIGHT_SCALE))))

    _mask_to_circle(surface, px, centre)
    return surface


def ring_surface(px, tones):
    """A flat ring system, drawn behind and in front of the planet.  Saturn only."""
    dark, mid, light = tones
    outer = int(px * 2.3)
    size = outer * 2 + 1
    centre = outer
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    for radius, tone, alpha in ((outer, light, 150),
                                (int(px * 1.95), mid, 190),
                                (int(px * 1.6), dark, 120)):
        rect = pygame.Rect(centre - radius, centre - int(radius * 0.30),
                           radius * 2, max(2, int(radius * 0.60)))
        pygame.draw.ellipse(surface, tone + (alpha,), rect, max(1, px // 6))
    return surface


class SpriteCache:
    """LRU of rendered bodies, keyed by everything that changes the pixels."""

    def __init__(self, limit=512):
        self.limit = limit
        self._entries = OrderedDict()

    def __len__(self):
        return len(self._entries)

    def clear(self):
        self._entries.clear()

    def get(self, material, seed, px, lit_angle, is_star):
        px = max(1, int(px))
        bucket = int(round(lit_angle / (2 * math.pi) * LIT_BUCKETS)) % LIT_BUCKETS
        key = (material.key, seed, px, bucket, bool(is_star))
        hit = self._entries.get(key)
        if hit is not None:
            self._entries.move_to_end(key)
            return hit
        angle = bucket * 2 * math.pi / LIT_BUCKETS
        surface = _make_body(material, seed, px, angle, is_star)
        self._entries[key] = surface
        if len(self._entries) > self.limit:
            self._entries.popitem(last=False)
        return surface
```

- [ ] **Step 4: Run and verify it passes**

Run: `python -m pytest tests/test_sprites.py -v`
Expected: PASS, 10 passed.

If `test_sprite_silhouette_is_circular` fails, `_mask_to_circle` is not being
applied — check that it runs after `_draw_detail`, not before.

- [ ] **Step 5: Commit**

```bash
git add src/render/sprites.py tests/test_sprites.py
git commit -m "feat: procedural pixel body sprites with bucketed lighting and LRU cache"
```

---
## Task 12: World rendering — the first milestone you can look at

At the end of this task the solar system is on screen, running, and you can fly
around it. Everything before now has been invisible.

**Files:**
- Create: `src/render/starfield.py`, `src/render/draw.py`, `tests/test_draw.py`
- Modify: `src/app/main.py`

**Interfaces:**
- Consumes: `Camera`, `SpriteCache`, `scale`, `palette`, `PixelFont`, `Simulation`
- Produces:
  - `Starfield(width, height, count=300, seed=7)` with `.draw(surface, camera)`
  - `ViewSettings` dataclass: `show_trails=True`, `real_size=False`, `show_labels=True`
  - `WorldRenderer(cache, font)` with `.draw(surface, sim, camera, view, selected_id=None)`
  - `nearest_star_angle(world, index) -> float` — lighting direction for a body

- [ ] **Step 1: Write the failing tests**

`tests/test_draw.py`:

```python
import math
import os

import numpy as np
import pygame
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    pygame.init()
    pygame.display.set_mode((320, 200))
    yield
    pygame.quit()


@pytest.fixture
def parts():
    from src.render.camera import Camera
    from src.render.draw import ViewSettings, WorldRenderer
    from src.render.pixelfont import PixelFont
    from src.render.sprites import SpriteCache
    from src.sim import Simulation
    surface = pygame.Surface((640, 400))
    return {
        "surface": surface,
        "sim": Simulation(),
        "camera": Camera(640, 400),
        "renderer": WorldRenderer(SpriteCache(limit=256), PixelFont(12)),
        "view": ViewSettings(),
    }


def test_lighting_points_from_the_sun():
    """Bodies shade away from the star, so the angle must point sunward."""
    from src.render.draw import nearest_star_angle
    from src.sim import Simulation
    sim = Simulation()
    world = sim.world
    names = [b.name for b in world.bodies]
    sun, earth = names.index("Sun"), names.index("Earth")
    angle = nearest_star_angle(world, earth)
    rel = world.pos[sun] - world.pos[earth]
    assert angle == pytest.approx(math.atan2(rel[1], rel[0]))


def test_lighting_of_a_lone_body_does_not_crash():
    from src import materials, units
    from src.render.draw import nearest_star_angle
    from src.world import World
    world = World()
    world.add("lonely", materials.get("rock"), units.EARTH_MASS,
              units.EARTH_RADIUS, np.zeros(2), np.zeros(2))
    assert isinstance(nearest_star_angle(world, 0), float)


def test_drawing_the_solar_system_marks_the_surface(parts):
    from src.render import palette
    parts["surface"].fill(palette.SPACE)
    parts["renderer"].draw(parts["surface"], parts["sim"], parts["camera"],
                           parts["view"])
    assert pygame.transform.average_color(parts["surface"])[:3] != palette.SPACE[:3]


def test_drawing_an_empty_world_does_not_crash(parts):
    parts["sim"].world.clear()
    parts["renderer"].draw(parts["surface"], parts["sim"], parts["camera"],
                           parts["view"])


def test_bodies_far_off_screen_are_skipped(parts):
    """Culling is what keeps a 200-body world cheap when zoomed in."""
    parts["camera"].centre = np.array([1e6, 1e6])
    before = len(parts["renderer"].cache)
    parts["renderer"].draw(parts["surface"], parts["sim"], parts["camera"],
                           parts["view"])
    assert len(parts["renderer"].cache) == before, "off-screen bodies were rendered"


def test_trails_can_be_switched_off(parts):
    from src.render import palette
    for _ in range(30):
        parts["sim"].advance(5.0, budget_s=5.0)
        parts["sim"].update_trails()

    parts["view"].show_trails = True
    parts["surface"].fill(palette.SPACE)
    parts["renderer"].draw(parts["surface"], parts["sim"], parts["camera"],
                           parts["view"])
    with_trails = pygame.transform.average_color(parts["surface"])

    parts["view"].show_trails = False
    parts["surface"].fill(palette.SPACE)
    parts["renderer"].draw(parts["surface"], parts["sim"], parts["camera"],
                           parts["view"])
    without = pygame.transform.average_color(parts["surface"])

    assert with_trails != without


def test_selection_ring_changes_the_picture(parts):
    from src.render import palette
    earth = next(b for b in parts["sim"].world.bodies if b.name == "Earth")
    parts["surface"].fill(palette.SPACE)
    parts["renderer"].draw(parts["surface"], parts["sim"], parts["camera"],
                           parts["view"])
    plain = pygame.image.tostring(parts["surface"], "RGB")

    parts["surface"].fill(palette.SPACE)
    parts["renderer"].draw(parts["surface"], parts["sim"], parts["camera"],
                           parts["view"], selected_id=earth.id)
    selected = pygame.image.tostring(parts["surface"], "RGB")

    assert plain != selected


def test_selecting_a_dead_body_id_is_harmless(parts):
    parts["renderer"].draw(parts["surface"], parts["sim"], parts["camera"],
                           parts["view"], selected_id=999999)


def test_starfield_draws_stars():
    from src.render import palette
    from src.render.camera import Camera
    from src.render.starfield import Starfield
    surface = pygame.Surface((320, 200))
    surface.fill(palette.SPACE)
    Starfield(320, 200, count=200).draw(surface, Camera(320, 200))
    assert pygame.transform.average_color(surface)[:3] != palette.SPACE[:3]


def test_starfield_is_deterministic():
    from src.render.camera import Camera
    from src.render.starfield import Starfield
    a, b = pygame.Surface((320, 200)), pygame.Surface((320, 200))
    cam = Camera(320, 200)
    Starfield(320, 200, seed=42).draw(a, cam)
    Starfield(320, 200, seed=42).draw(b, cam)
    assert pygame.image.tostring(a, "RGB") == pygame.image.tostring(b, "RGB")
```

- [ ] **Step 2: Run and watch it fail**

Run: `python -m pytest tests/test_draw.py -v`
Expected: FAIL — `No module named 'src.render.starfield'`.

- [ ] **Step 3: Implement starfield.py**

```python
"""The background.

Real stars are so far away they would not parallax at all across a solar system,
but a very small factor gives a depth cue when panning without implying they are
nearby.  Positions come from a fixed seed so the sky does not reshuffle itself
every time the window is resized.
"""

import numpy as np
import pygame

PARALLAX = 0.02


class Starfield:
    def __init__(self, width, height, count=300, seed=7):
        rng = np.random.default_rng(seed)
        self.width = width
        self.height = height
        # Generated over three screen-widths so panning never runs out of sky.
        self._xy = rng.random((count, 2)) * (width * 3.0, height * 3.0)
        self._brightness = rng.integers(40, 190, count)

    def draw(self, surface, camera):
        offset_x = (-camera.centre[0] * camera.zoom * PARALLAX) % (self.width * 3.0)
        offset_y = (camera.centre[1] * camera.zoom * PARALLAX) % (self.height * 3.0)
        xs = (self._xy[:, 0] + offset_x) % (self.width * 3.0)
        ys = (self._xy[:, 1] + offset_y) % (self.height * 3.0)
        visible = (xs < self.width) & (ys < self.height)
        for x, y, value in zip(xs[visible].astype(int),
                               ys[visible].astype(int),
                               self._brightness[visible]):
            surface.set_at((int(x), int(y)), (int(value), int(value), int(value * 1.1)))
```

- [ ] **Step 4: Implement draw.py**

```python
"""Rendering the world: sky, trails, bodies, selection.

Two performance rules shape this file.

**Cull first.**  A body whose disc cannot touch the viewport is skipped before
its sprite is ever requested, which is what keeps the sprite cache from
thrashing when the camera is zoomed into a moon.

**Trails are drawn in chunks, not segments.**  A trail holds up to 512 points;
fading each segment individually would mean ~500 draw calls per body and roughly
10ms a frame across the system.  Eight chunks, each a step dimmer, costs eight
calls and looks the same at this resolution.
"""

import math
from dataclasses import dataclass

import numpy as np
import pygame

from ..body import KIND_STAR
from . import palette, scale
from .sprites import ring_surface

#: Trail fade steps.  See the module docstring for why this is not per-segment.
TRAIL_CHUNKS = 8
#: A body must be within this many pixels of the viewport to be drawn.
CULL_MARGIN = 64
#: Labels are suppressed below this drawn radius to stop the view turning to soup.
LABEL_MIN_PX = 3


@dataclass
class ViewSettings:
    show_trails: bool = True
    real_size: bool = False
    show_labels: bool = True


def nearest_star_angle(world, index):
    """Direction from ``index`` toward the star lighting it, in radians.

    Falls back to a fixed angle when there is no star, so a debris-only world
    still shades consistently instead of flickering.
    """
    best = None
    best_d2 = float("inf")
    for k in range(world.n):
        if k == index or world.bodies[k].kind != KIND_STAR:
            continue
        rel = world.pos[k] - world.pos[index]
        d2 = float(rel[0] * rel[0] + rel[1] * rel[1])
        if d2 < best_d2:
            best_d2, best = d2, rel
    if best is None:
        return 0.0
    return float(math.atan2(best[1], best[0]))


class WorldRenderer:
    def __init__(self, cache, font):
        self.cache = cache
        self.font = font

    def draw(self, surface, sim, camera, view, selected_id=None):
        world = sim.world
        width, height = surface.get_size()

        if view.show_trails:
            self._draw_trails(surface, world, camera, width, height)

        radii = scale.display_radii_px(world.radii, camera.zoom, height,
                                       real_size=view.real_size)
        screen = camera.world_to_screen_array(world.positions)

        for i in range(world.n):
            px = radii[i]
            x, y = screen[i]
            if (x < -px - CULL_MARGIN or x > width + px + CULL_MARGIN
                    or y < -px - CULL_MARGIN or y > height + px + CULL_MARGIN):
                continue
            body = world.bodies[i]
            is_star = body.kind == KIND_STAR

            if px < 2.0:
                surface.set_at((int(x), int(y)), body.material.tones[2])
            else:
                if body.rings and px >= 4:
                    rings = ring_surface(int(px), body.material.tones)
                    rect = rings.get_rect(center=(int(x), int(y)))
                    surface.blit(rings, rect)
                sprite = self.cache.get(body.material, body.seed, int(px),
                                        nearest_star_angle(world, i), is_star)
                rect = sprite.get_rect(center=(int(x), int(y)))
                surface.blit(sprite, rect,
                             special_flags=pygame.BLEND_ADD if is_star else 0)

            if selected_id is not None and body.id == selected_id:
                pygame.draw.circle(surface, palette.SELECT, (int(x), int(y)),
                                   int(max(px, 3) + 5), 1)

            if view.show_labels and px >= LABEL_MIN_PX:
                self.font.draw(surface, int(x + px + 4), int(y - self.font.height // 2),
                               body.name, palette.TEXT_DIM)

    def _draw_trails(self, surface, world, camera, width, height):
        for i in range(world.n):
            points = world.bodies[i].trail.ordered()
            if len(points) < 4:
                continue
            projected = camera.world_to_screen_array(points)
            # Cheap reject: a trail entirely off-screen costs nothing.
            if (projected[:, 0].max() < 0 or projected[:, 0].min() > width
                    or projected[:, 1].max() < 0 or projected[:, 1].min() > height):
                continue
            base = world.bodies[i].material.tones[1]
            per = max(2, len(projected) // TRAIL_CHUNKS)
            for chunk in range(TRAIL_CHUNKS):
                start = chunk * per
                stop = min(len(projected), start + per + 1)
                if stop - start < 2:
                    continue
                fade = 0.25 + 0.75 * (chunk + 1) / TRAIL_CHUNKS
                colour = tuple(int(palette.TRAIL_BASE[k] * (1 - fade) + base[k] * fade)
                               for k in range(3))
                pygame.draw.lines(surface, colour, False,
                                  projected[start:stop].astype(int).tolist(), 1)
```

- [ ] **Step 5: Run and verify it passes**

Run: `python -m pytest tests/test_draw.py -v`
Expected: PASS, 10 passed.

- [ ] **Step 6: Replace main.py with a real frame loop**

```python
"""Entry point and frame loop."""

import sys

import pygame

from ..render.camera import Camera
from ..render.draw import ViewSettings, WorldRenderer
from ..render.pixelfont import PixelFont
from ..render.sprites import SpriteCache
from ..render.starfield import Starfield
from ..sim import Simulation
from .config import font_px, open_display


def main():
    pygame.init()
    ctx = open_display()
    clock = pygame.time.Clock()

    sim = Simulation()
    camera = Camera(ctx.width, ctx.height)
    font = PixelFont(font_px(ctx.s))
    renderer = WorldRenderer(SpriteCache(), font)
    stars = Starfield(ctx.width, ctx.height)
    view = ViewSettings()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                camera.zoom_by(event.y, mx, my)

        frame_dt = clock.tick(60) / 1000.0
        sim.advance(30.0 * frame_dt)
        sim.update_trails()
        camera.update_follow(sim.world)

        ctx.surface.fill((6, 8, 16))
        stars.draw(ctx.surface, camera)
        renderer.draw(ctx.surface, sim, camera, view)
        pygame.display.flip()

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7: Look at it**

Run: `python -m src.app.main`

You should see the solar system, in pixels, orbiting at roughly a month per
second, with trails building behind each planet. The wheel zooms. Escape quits.
**This is the milestone — stop and actually look at it before continuing.**

- [ ] **Step 8: Commit**

```bash
git add src/render/starfield.py src/render/draw.py src/app/main.py tests/test_draw.py
git commit -m "feat: world rendering — the solar system is on screen"
```

---
## Task 13: Application state and time controls

**Files:**
- Create: `src/app/state.py`, `tests/test_state.py`

**Interfaces:**
- Consumes: `Simulation`, `Camera`, `ViewSettings`
- Produces:
  - `TIME_SCALES` (tuple of days per real second) and `TIME_LABELS` (parallel tuple of strings)
  - `AppState(sim, camera, view)` with attributes `selected_id`, `time_index`, `running`, `help_visible`, `tool_name`
  - `.days_per_second -> float`, `.time_label -> str`
  - `.nudge_time(delta) -> None`
  - `.toggle_pause() -> None`
  - `.selected_index -> int | None`
  - `.select(body_id) -> None`
  - `.cycle_selection(step=1) -> None`
  - `.pick_body_at(sx, sy) -> int | None`
  - `.follow_selected() -> None`
  - `.delete_selected() -> None`
  - `.elapsed_label -> str`

- [ ] **Step 1: Write the failing tests**

`tests/test_state.py`:

```python
import numpy as np
import pytest

from src.render.camera import Camera
from src.render.draw import ViewSettings
from src.sim import Simulation


@pytest.fixture
def state():
    from src.app.state import AppState
    sim = Simulation()
    return AppState(sim, Camera(1000, 700), ViewSettings())


def test_time_ladder_is_monotonic_and_labelled():
    from src.app.state import TIME_LABELS, TIME_SCALES
    assert len(TIME_SCALES) == len(TIME_LABELS)
    assert list(TIME_SCALES) == sorted(TIME_SCALES)
    assert TIME_SCALES[0] < 1.0 < TIME_SCALES[-1]


def test_time_scale_clamps_at_both_ends(state):
    from src.app.state import TIME_SCALES
    for _ in range(50):
        state.nudge_time(-1)
    assert state.days_per_second == TIME_SCALES[0]
    for _ in range(50):
        state.nudge_time(1)
    assert state.days_per_second == TIME_SCALES[-1]


def test_pause_toggles_the_simulation(state):
    assert state.sim.paused is False
    state.toggle_pause()
    assert state.sim.paused is True
    state.toggle_pause()
    assert state.sim.paused is False


def test_selection_survives_bodies_being_removed(state):
    """Selection stores an id, so a merge elsewhere must not silently reselect
    a different planet."""
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    state.select(earth.id)
    state.sim.world.remove(0)          # remove the Sun; rows shuffle
    assert state.sim.world.bodies[state.selected_index].name == "Earth"


def test_selecting_a_removed_body_reports_none(state):
    doomed = state.sim.world.bodies[-1]
    state.select(doomed.id)
    state.sim.world.remove(state.sim.world.index_of(doomed.id))
    assert state.selected_index is None


def test_clicking_on_a_planet_picks_it(state):
    world = state.sim.world
    jupiter = next(b for b in world.bodies if b.name == "Jupiter")
    index = world.index_of(jupiter.id)
    state.camera.centre = world.pos[index].copy()
    sx, sy = state.camera.world_to_screen(world.pos[index])
    assert state.pick_body_at(sx, sy) == index


def test_clicking_empty_space_picks_nothing(state):
    assert state.pick_body_at(3.0, 3.0) is None


def test_cycle_selection_visits_different_bodies(state):
    state.cycle_selection()
    first = state.selected_id
    state.cycle_selection()
    assert state.selected_id != first


def test_follow_selected_sets_the_camera(state):
    mars = next(b for b in state.sim.world.bodies if b.name == "Mars")
    state.select(mars.id)
    state.follow_selected()
    assert state.camera.follow_id == mars.id


def test_delete_selected_removes_it_and_clears_selection(state):
    mars = next(b for b in state.sim.world.bodies if b.name == "Mars")
    before = state.sim.world.n
    state.select(mars.id)
    state.delete_selected()
    assert state.sim.world.n == before - 1
    assert state.selected_id is None
    assert all(b.name != "Mars" for b in state.sim.world.bodies)


def test_elapsed_label_reports_years_and_days(state):
    state.sim.world.time = 400.0
    label = state.elapsed_label
    assert "Year" in label and "Day" in label
```

- [ ] **Step 2: Run and watch it fail**

Run: `python -m pytest tests/test_state.py -v`
Expected: FAIL — `No module named 'src.app.state'`.

- [ ] **Step 3: Implement state.py**

```python
"""Everything the application knows that is not the world itself.

Selection and camera-follow store body **ids**, never indices, because
World.remove swap-removes: an index held from one frame to the next can quietly
start referring to a different planet.  Every read goes through index_of.
"""

import math

from .. import units
from ..render import scale

#: Days of simulation per second of real time, coarsest last.
TIME_SCALES = (
    1.0 / 24.0, 0.25, 1.0, 3.0, 7.0, 30.0, 90.0,
    units.DAYS_PER_YEAR, 3.0 * units.DAYS_PER_YEAR, 10.0 * units.DAYS_PER_YEAR,
)
TIME_LABELS = (
    "1 hr/s", "6 hr/s", "1 day/s", "3 day/s", "1 week/s", "1 month/s",
    "3 month/s", "1 yr/s", "3 yr/s", "10 yr/s",
)

DEFAULT_TIME_INDEX = 5          # 1 month per second: fast enough to see motion

#: Click tolerance in pixels, so a 2px Mercury is still catchable.
PICK_MARGIN = 8


class AppState:
    def __init__(self, sim, camera, view):
        self.sim = sim
        self.camera = camera
        self.view = view
        self.selected_id = None
        self.time_index = DEFAULT_TIME_INDEX
        self.running = True
        self.help_visible = False
        self.tool_name = "Look"

    # --- time -------------------------------------------------------------------

    @property
    def days_per_second(self):
        return TIME_SCALES[self.time_index]

    @property
    def time_label(self):
        return TIME_LABELS[self.time_index]

    def nudge_time(self, delta):
        self.time_index = max(0, min(len(TIME_SCALES) - 1, self.time_index + delta))

    def toggle_pause(self):
        self.sim.paused = not self.sim.paused

    @property
    def elapsed_label(self):
        days = self.sim.world.time
        years = days / units.DAYS_PER_YEAR
        return "Year %.1f  Day %d" % (years, int(days))

    # --- selection --------------------------------------------------------------

    @property
    def selected_index(self):
        if self.selected_id is None:
            return None
        return self.sim.world.index_of(self.selected_id)

    @property
    def selected_body(self):
        index = self.selected_index
        return None if index is None else self.sim.world.bodies[index]

    def select(self, body_id):
        self.selected_id = body_id

    def cycle_selection(self, step=1):
        world = self.sim.world
        if world.n == 0:
            self.selected_id = None
            return
        current = self.selected_index
        nxt = 0 if current is None else (current + step) % world.n
        self.selected_id = world.bodies[nxt].id

    def pick_body_at(self, sx, sy, internal_height=None):
        """Index of the body under a screen point, or None.

        Picks the *smallest* match so a moon in front of its planet is still
        selectable rather than always losing to the bigger disc behind it.
        """
        world = self.sim.world
        if world.n == 0:
            return None
        height = internal_height or self.camera.height
        radii = scale.display_radii_px(world.radii, self.camera.zoom, height,
                                       real_size=self.view.real_size)
        screen = self.camera.world_to_screen_array(world.positions)
        best = None
        best_px = float("inf")
        for i in range(world.n):
            reach = max(radii[i], 2.0) + PICK_MARGIN
            if math.hypot(screen[i][0] - sx, screen[i][1] - sy) <= reach:
                if radii[i] < best_px:
                    best_px, best = radii[i], i
        return best

    def follow_selected(self):
        if self.selected_id is not None:
            self.camera.follow_id = self.selected_id

    def delete_selected(self):
        index = self.selected_index
        if index is None:
            return
        self.sim.world.remove(index)
        self.selected_id = None
        self.sim.refresh()
```

- [ ] **Step 4: Run and verify it passes**

Run: `python -m pytest tests/test_state.py -v`
Expected: PASS, 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/app/state.py tests/test_state.py
git commit -m "feat: application state, time ladder, id-safe selection and picking"
```

---

## Task 14: The HUD

**Files:**
- Create: `src/render/hud.py`, `tests/test_hud.py`

**Interfaces:**
- Consumes: `palette`, `PixelFont`, `AppState`, `src.units`, `src.orbits`
- Produces:
  - `format_mass(m) -> str`, `format_length(r) -> str`, `format_speed(v) -> str`
  - `orbit_facts(world, index) -> dict | None` with keys `attractor`, `a`, `e`, `period_label`, `periapsis`, `apoapsis`, `speed`
  - `Hud(font)` with `.draw(surface, state)`, `.draw_help(surface)`
  - `KEY_HELP` — tuple of `(keys, description)` pairs

- [ ] **Step 1: Write the failing tests**

`tests/test_hud.py`:

```python
import os

import pygame
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    pygame.init()
    pygame.display.set_mode((320, 200))
    yield
    pygame.quit()


@pytest.fixture
def state():
    from src.app.state import AppState
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.sim import Simulation
    return AppState(Simulation(), Camera(1000, 700), ViewSettings())


def test_mass_is_shown_in_whichever_unit_is_readable():
    from src import units
    from src.render.hud import format_mass
    assert "Earth" in format_mass(units.EARTH_MASS)
    assert "Jupiter" in format_mass(units.JUPITER_MASS)
    assert "Sun" in format_mass(1.0)


def test_length_switches_between_km_and_au():
    from src import units
    from src.render.hud import format_length
    assert "km" in format_length(units.EARTH_RADIUS)
    assert "AU" in format_length(5.2)


def test_speed_is_in_km_per_second():
    from src import units
    from src.render.hud import format_speed
    text = format_speed(30.0 / units.AUD_TO_KMS)
    assert "km/s" in text and "30" in text


def test_orbit_facts_name_the_right_attractor(state):
    from src.render.hud import orbit_facts
    world = state.sim.world
    names = [b.name for b in world.bodies]
    facts = orbit_facts(world, names.index("Moon"))
    assert facts["attractor"] == "Earth"


def test_orbit_facts_get_earths_year_right(state):
    from src.render.hud import orbit_facts
    world = state.sim.world
    names = [b.name for b in world.bodies]
    facts = orbit_facts(world, names.index("Earth"))
    assert facts["a"] == pytest.approx(1.0, rel=0.02)
    assert "1.0 yr" in facts["period_label"] or "1 yr" in facts["period_label"]


def test_orbit_facts_of_a_lone_body_are_none():
    from src import materials, units
    from src.render.hud import orbit_facts
    from src.world import World
    import numpy as np
    world = World()
    world.add("lonely", materials.get("rock"), units.EARTH_MASS,
              units.EARTH_RADIUS, np.zeros(2), np.zeros(2))
    assert orbit_facts(world, 0) is None


def test_hud_draws_without_a_selection(state):
    from src.render import palette
    from src.render.hud import Hud
    from src.render.pixelfont import PixelFont
    surface = pygame.Surface((1000, 700))
    surface.fill(palette.SPACE)
    Hud(PixelFont(12)).draw(surface, state)
    assert pygame.transform.average_color(surface)[:3] != palette.SPACE[:3]


def test_hud_draws_the_inspector_when_something_is_selected(state):
    from src.render import palette
    from src.render.hud import Hud
    from src.render.pixelfont import PixelFont
    hud = Hud(PixelFont(12))

    plain = pygame.Surface((1000, 700))
    plain.fill(palette.SPACE)
    hud.draw(plain, state)

    state.select(next(b for b in state.sim.world.bodies if b.name == "Saturn").id)
    with_panel = pygame.Surface((1000, 700))
    with_panel.fill(palette.SPACE)
    hud.draw(with_panel, state)

    assert pygame.image.tostring(plain, "RGB") != pygame.image.tostring(with_panel, "RGB")


def test_hud_survives_the_selected_body_being_destroyed(state):
    from src.render.hud import Hud
    from src.render.pixelfont import PixelFont
    doomed = state.sim.world.bodies[-1]
    state.select(doomed.id)
    state.sim.world.remove(state.sim.world.index_of(doomed.id))
    Hud(PixelFont(12)).draw(pygame.Surface((1000, 700)), state)


def test_help_overlay_lists_every_binding():
    from src.render.hud import KEY_HELP, Hud
    from src.render.pixelfont import PixelFont
    assert len(KEY_HELP) >= 12
    surface = pygame.Surface((1000, 700))
    Hud(PixelFont(12)).draw_help(surface)
```

- [ ] **Step 2: Run and watch it fail**

Run: `python -m pytest tests/test_hud.py -v`
Expected: FAIL — `No module named 'src.render.hud'`.

- [ ] **Step 3: Implement hud.py**

```python
"""Numbers on screen.

The audience is an eight-year-old sitting next to an adult, so this shows real
figures in real units rather than icons: Earth masses, kilometres, g's, km/s.
Every number is derived from the simulation state at the moment it is drawn --
nothing is cached or precomputed, so a body edited mid-drag updates live.
"""

import numpy as np
import pygame

from .. import orbits, units
from ..body import KIND_STAR
from . import palette

PAD = 6
PANEL_WIDTH_CHARS = 34

KEY_HELP = (
    ("Space", "pause and resume"),
    ("[  ]", "slower / faster"),
    ("Wheel", "zoom at the cursor"),
    ("Drag", "pan the view"),
    ("Click", "select a body"),
    ("Tab", "cycle through bodies"),
    ("F", "follow the selected body"),
    ("1 2 3", "Look / Throw / Edit"),
    ("0", "stop the selected body dead"),
    ("T", "orbit trails on and off"),
    ("R", "real sizes (everything is a dot)"),
    ("L", "name labels"),
    ("G", "pixel grid size"),
    ("Del", "delete the selected body"),
    ("F5", "start the solar system over"),
    ("Cmd+F", "fullscreen"),
    ("?", "this help"),
    ("Esc", "deselect, or close this help"),
)


# --- formatting -------------------------------------------------------------------

def format_mass(mass):
    """Mass in whichever familiar unit keeps the number readable."""
    if mass >= 0.02:
        return "%.3g Suns" % mass
    if mass >= 0.3 * units.JUPITER_MASS:
        return "%.3g Jupiters" % units.to_jupiter_masses(mass)
    if mass >= 1e-4 * units.EARTH_MASS:
        return "%.3g Earths" % units.to_earth_masses(mass)
    return "%.3g kg" % (mass * units.MSUN_KG)


def format_length(length):
    if length >= 0.02:
        return "%.4g AU" % length
    return "%.4g km" % units.to_km(length)


def format_speed(speed):
    return "%.4g km/s" % units.to_kms(speed)


def orbit_facts(world, index):
    """Osculating orbit of ``index`` about whatever is pulling hardest on it."""
    attractor = orbits.dominant_attractor(world, index)
    if attractor is None:
        return None
    mu = units.G * (world.mass[attractor] + world.mass[index])
    rel_pos = world.pos[index] - world.pos[attractor]
    rel_vel = world.vel[index] - world.vel[attractor]
    el = orbits.elements_from_state(mu, rel_pos, rel_vel)

    if not el["bound"]:
        period_label = "escaping"
    elif el["period"] >= 2.0 * units.DAYS_PER_YEAR:
        period_label = "%.1f yr" % units.to_years(el["period"])
    else:
        period_label = "%.1f days" % el["period"]

    return {
        "attractor": world.bodies[attractor].name,
        "a": el["a"],
        "e": el["e"],
        "period_label": period_label,
        "periapsis": el["periapsis"],
        "apoapsis": el["apoapsis"],
        "speed": float(np.hypot(rel_vel[0], rel_vel[1])),
    }


# --- drawing ----------------------------------------------------------------------

class Hud:
    def __init__(self, font):
        self.font = font
        self.line = font.height + 2

    def _panel(self, surface, rect):
        box = pygame.Surface(rect.size, pygame.SRCALPHA)
        box.fill(palette.PANEL + (232,))
        surface.blit(box, rect.topleft)
        pygame.draw.rect(surface, palette.PANEL_EDGE, rect, 1)

    def draw(self, surface, state):
        self._draw_top_bar(surface, state)
        self._draw_inspector(surface, state)
        self._draw_hint(surface, state)
        if state.help_visible:
            self.draw_help(surface)

    def _draw_top_bar(self, surface, state):
        width = surface.get_width()
        rect = pygame.Rect(0, 0, width, self.line + PAD)
        self._panel(surface, rect)

        x = PAD
        x += self.font.draw(surface, x, PAD // 2, state.elapsed_label, palette.TEXT) + 18
        x += self.font.draw(surface, x, PAD // 2,
                            "PAUSED" if state.sim.paused else state.time_label,
                            palette.WARN if state.sim.paused else palette.ACCENT) + 18

        if state.sim.last_ratio < 0.95 and not state.sim.paused:
            x += self.font.draw(surface, x, PAD // 2,
                                "running at %d%%" % int(state.sim.last_ratio * 100),
                                palette.WARN) + 18

        x += self.font.draw(surface, x, PAD // 2,
                            "%d bodies" % state.sim.world.n, palette.TEXT_DIM) + 18
        self.font.draw(surface, x, PAD // 2, state.tool_name, palette.SELECT)

    def _draw_inspector(self, surface, state):
        index = state.selected_index
        if index is None:
            return
        world = state.sim.world
        body = world.bodies[index]
        mass = float(world.mass[index])
        radius = float(world.radius[index])
        density = units.density_from_mass_radius(mass, radius)

        rows = [
            (body.name, palette.SELECT),
            ("%s  -  %s" % (body.kind, body.material.name), palette.TEXT_DIM),
            ("", palette.TEXT),
            ("Mass       %s" % format_mass(mass), palette.TEXT),
            ("Radius     %s" % format_length(radius), palette.TEXT),
            ("           %.3g Earths" % units.to_earth_radii(radius), palette.TEXT_DIM),
            ("Density    %.3g g/cm3" % units.to_gcm3(density), palette.TEXT),
            ("Gravity    %.3g g" % units.surface_gravity_g(mass, radius), palette.TEXT),
            ("Escape     %s" % format_speed(units.escape_velocity(mass, radius)),
             palette.TEXT),
        ]

        facts = orbit_facts(world, index)
        if facts is not None:
            rows += [
                ("", palette.TEXT),
                ("Orbits     %s" % facts["attractor"], palette.ACCENT),
                ("Distance   %s" % format_length(facts["a"]), palette.TEXT),
                ("Stretch    %.3f" % facts["e"], palette.TEXT),
                ("One lap    %s" % facts["period_label"], palette.TEXT),
                ("Closest    %s" % format_length(facts["periapsis"]), palette.TEXT),
            ]
            if facts["apoapsis"] is not None:
                rows.append(("Farthest   %s" % format_length(facts["apoapsis"]),
                             palette.TEXT))
            rows.append(("Speed      %s" % format_speed(facts["speed"]), palette.TEXT))

        if body.kind != KIND_STAR:
            short = units.STAR_MASS_THRESHOLD / mass
            rows += [
                ("", palette.TEXT),
                ("%.0fx heavier to ignite" % short, palette.WARN),
            ]

        rows += [("", palette.TEXT), (body.material.blurb, palette.TEXT_DIM)]

        width = self.font.width("X") * PANEL_WIDTH_CHARS + PAD * 2
        height = len(rows) * self.line + PAD * 2
        rect = pygame.Rect(surface.get_width() - width - PAD, self.line + PAD * 3,
                           width, height)
        self._panel(surface, rect)
        for k, (text, colour) in enumerate(rows):
            self.font.draw(surface, rect.x + PAD, rect.y + PAD + k * self.line,
                           text, colour)

    def _draw_hint(self, surface, state):
        width, height = surface.get_size()
        rect = pygame.Rect(0, height - self.line - PAD, width, self.line + PAD)
        self._panel(surface, rect)
        self.font.draw(surface, PAD, height - self.line - PAD // 2,
                       state.hint, palette.TEXT_DIM)

    def draw_help(self, surface):
        width = self.font.width("X") * 46 + PAD * 2
        height = (len(KEY_HELP) + 2) * self.line + PAD * 2
        rect = pygame.Rect((surface.get_width() - width) // 2,
                           (surface.get_height() - height) // 2, width, height)
        self._panel(surface, rect)
        self.font.draw(surface, rect.x + PAD, rect.y + PAD, "Controls", palette.SELECT)
        for k, (keys, description) in enumerate(KEY_HELP):
            y = rect.y + PAD + (k + 2) * self.line
            self.font.draw(surface, rect.x + PAD, y, keys, palette.ACCENT)
            self.font.draw(surface, rect.x + PAD + self.font.width("X") * 9, y,
                           description, palette.TEXT)
```

- [ ] **Step 4: Add the `hint` property to AppState**

`_draw_hint` reads `state.hint`, which Task 15 will hand over to the active tool.
For now add to `src/app/state.py`:

```python
    @property
    def hint(self):
        return "Click a body to inspect it  -  Wheel zooms  -  ? for controls"
```

- [ ] **Step 5: Run and verify it passes**

Run: `python -m pytest tests/test_hud.py -v`
Expected: PASS, 10 passed.

- [ ] **Step 6: Commit**

```bash
git add src/render/hud.py src/app/state.py tests/test_hud.py
git commit -m "feat: HUD with real units, live orbit facts and a help overlay"
```

---
## Task 15: Tool framework and SelectTool

**Files:**
- Create: `src/app/tools.py`, `src/app/input.py`, `tests/test_tools_select.py`
- Modify: `src/app/state.py` (delegate `hint` to the tool), `src/app/main.py`

**Interfaces:**
- Consumes: `AppState`, `Camera`, `Simulation`
- Produces:
  - `Tool` base class: attributes `name`, `hint`; methods `on_event(event, state) -> bool`, `draw_overlay(surface, state, font)`, `on_exit(state)`
  - `SelectTool(Tool)` with `name = "Look"`
  - `TOOLS` — dict mapping `pygame.K_1/K_2/K_3` to tool classes (Task 16 and 17 extend it)
  - `handle_event(event, state, ctx) -> None` in `input.py`
  - `AppState.tool` attribute holding the live Tool instance, `AppState.set_tool(cls)`

- [ ] **Step 1: Write the failing tests**

`tests/test_tools_select.py`:

```python
import os

import pygame
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    pygame.init()
    pygame.display.set_mode((320, 200))
    yield
    pygame.quit()


@pytest.fixture
def state():
    from src.app.state import AppState
    from src.app.tools import SelectTool
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.sim import Simulation
    st = AppState(Simulation(), Camera(1000, 700), ViewSettings())
    st.set_tool(SelectTool)
    return st


def _click(x, y, button=1):
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(x, y), button=button)


def _release(x, y, button=1):
    return pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(x, y), button=button)


def _motion(x, y, rel, buttons=(1, 0, 0)):
    return pygame.event.Event(pygame.MOUSEMOTION, pos=(x, y), rel=rel, buttons=buttons)


def test_hint_comes_from_the_active_tool(state):
    assert state.hint == state.tool.hint
    assert state.tool_name == "Look"


def test_clicking_a_body_selects_it(state):
    world = state.sim.world
    jupiter = next(b for b in world.bodies if b.name == "Jupiter")
    index = world.index_of(jupiter.id)
    state.camera.centre = world.pos[index].copy()
    sx, sy = state.camera.world_to_screen(world.pos[index])
    state.tool.on_event(_click(int(sx), int(sy)), state)
    state.tool.on_event(_release(int(sx), int(sy)), state)
    assert state.selected_id == jupiter.id


def test_clicking_empty_space_deselects(state):
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    state.select(earth.id)
    state.tool.on_event(_click(2, 2), state)
    state.tool.on_event(_release(2, 2), state)
    assert state.selected_id is None


def test_dragging_pans_instead_of_selecting(state):
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    state.select(earth.id)
    start = state.camera.centre.copy()
    state.tool.on_event(_click(500, 350), state)
    state.tool.on_event(_motion(560, 350, (60, 0)), state)
    state.tool.on_event(_release(560, 350), state)
    assert state.camera.centre[0] != start[0]
    assert state.selected_id == earth.id, "a drag must not change the selection"


def test_double_click_frames_the_body(state):
    world = state.sim.world
    saturn = next(b for b in world.bodies if b.name == "Saturn")
    index = world.index_of(saturn.id)
    state.camera.centre = world.pos[index].copy()
    sx, sy = state.camera.world_to_screen(world.pos[index])
    zoom_before = state.camera.zoom
    event = _click(int(sx), int(sy))
    state.tool.on_event(event, state)
    state.tool.on_event(_release(int(sx), int(sy)), state)
    state.tool.on_event(event, state)          # second click, no motion between
    state.tool.on_event(_release(int(sx), int(sy)), state)
    assert state.camera.zoom > zoom_before


def test_number_keys_switch_tools(state):
    from src.app.input import handle_event
    from src.app.tools import SelectTool
    handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_1, mod=0), state, None)
    assert isinstance(state.tool, SelectTool)


def test_bracket_keys_change_the_time_scale(state):
    from src.app.input import handle_event
    before = state.days_per_second
    handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHTBRACKET, mod=0),
                 state, None)
    assert state.days_per_second > before
    handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFTBRACKET, mod=0),
                 state, None)
    assert state.days_per_second == before


def test_space_pauses(state):
    from src.app.input import handle_event
    handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE, mod=0),
                 state, None)
    assert state.sim.paused is True


def test_toggles_flip_view_settings(state):
    from src.app.input import handle_event
    for key, attribute in ((pygame.K_t, "show_trails"),
                           (pygame.K_r, "real_size"),
                           (pygame.K_l, "show_labels")):
        before = getattr(state.view, attribute)
        handle_event(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0), state, None)
        assert getattr(state.view, attribute) is not before


def test_zero_key_stops_the_selected_body(state):
    import numpy as np
    from src.app.input import handle_event
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    state.select(earth.id)
    handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_0, mod=0), state, None)
    index = state.sim.world.index_of(earth.id)
    assert np.allclose(state.sim.world.vel[index], 0.0)


def test_escape_closes_help_before_deselecting(state):
    from src.app.input import handle_event
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    state.select(earth.id)
    state.help_visible = True
    escape = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0)
    handle_event(escape, state, None)
    assert state.help_visible is False
    assert state.selected_id == earth.id
    handle_event(escape, state, None)
    assert state.selected_id is None


def test_delete_removes_the_selected_body(state):
    from src.app.input import handle_event
    mars = next(b for b in state.sim.world.bodies if b.name == "Mars")
    state.select(mars.id)
    before = state.sim.world.n
    handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DELETE, mod=0),
                 state, None)
    assert state.sim.world.n == before - 1
```

- [ ] **Step 2: Run and watch it fail**

Run: `python -m pytest tests/test_tools_select.py -v`
Expected: FAIL — `No module named 'src.app.tools'`.

- [ ] **Step 3: Implement tools.py**

```python
"""Interaction tools.

Each tool owns its own event handling and its own overlay drawing, so adding one
is a new class rather than another branch in a growing event handler.  Exactly
one tool is active; the HUD's hint line is whatever it says it is.
"""

import pygame

#: A press-and-release within this many pixels counts as a click, not a drag.
CLICK_SLOP = 4
#: Two clicks within this many milliseconds on the same body zoom to it.
DOUBLE_CLICK_MS = 400


class Tool:
    name = "Tool"
    hint = ""

    def on_event(self, event, state):
        """Return True if the event was consumed."""
        return False

    def draw_overlay(self, surface, state, font):
        pass

    def on_exit(self, state):
        pass


class SelectTool(Tool):
    name = "Look"
    hint = ("Click a body to inspect it  -  drag to pan  -  wheel to zoom  -  "
            "double-click to fly to it  -  ? for controls")

    def __init__(self):
        self._press = None
        self._dragged = False
        self._last_click = (None, 0)

    def on_event(self, event, state):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._press = event.pos
            self._dragged = False
            return True

        if event.type == pygame.MOUSEMOTION and self._press is not None:
            if event.buttons[0]:
                dx, dy = event.rel
                if abs(event.pos[0] - self._press[0]) > CLICK_SLOP or \
                        abs(event.pos[1] - self._press[1]) > CLICK_SLOP:
                    self._dragged = True
                if self._dragged:
                    state.camera.pan_pixels(dx, dy)
                return True

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            pressed, self._press = self._press, None
            if pressed is None or self._dragged:
                return True
            self._click(event.pos, state)
            return True

        return False

    def _click(self, pos, state):
        index = state.pick_body_at(pos[0], pos[1])
        if index is None:
            state.selected_id = None
            self._last_click = (None, 0)
            return

        body_id = state.sim.world.bodies[index].id
        now = pygame.time.get_ticks()
        previous_id, previous_time = self._last_click
        if previous_id == body_id and now - previous_time < DOUBLE_CLICK_MS:
            state.camera.frame_body(state.sim.world, index)
            state.camera.follow_id = body_id
        state.selected_id = body_id
        self._last_click = (body_id, now)


#: Filled in by Tasks 16 and 17.
TOOLS = {
    pygame.K_1: SelectTool,
}
```

- [ ] **Step 4: Implement input.py**

```python
"""Keyboard and mouse routing.

The active tool sees every mouse event first and can consume it; global keys are
handled here.  Keeping the split at "tool first, globals second" means a tool can
capture a drag without the camera also panning underneath it.
"""

import pygame

from .tools import TOOLS


def handle_event(event, state, ctx):
    """Route one pygame event.  ``ctx`` may be None in tests."""
    if event.type == pygame.QUIT:
        state.running = False
        return

    if state.tool is not None and state.tool.on_event(event, state):
        return

    if event.type == pygame.MOUSEWHEEL:
        mx, my = pygame.mouse.get_pos()
        state.camera.zoom_by(event.y, mx, my)
        return

    if event.type != pygame.KEYDOWN:
        return

    key = event.key
    command = event.mod & (pygame.KMOD_META | pygame.KMOD_CTRL)

    if command and key == pygame.K_q:
        state.running = False
    elif command and key == pygame.K_f:
        if ctx is not None:
            ctx.toggle_fullscreen()
    elif key in TOOLS:
        state.set_tool(TOOLS[key])
    elif key == pygame.K_SPACE:
        state.toggle_pause()
    elif key == pygame.K_RIGHTBRACKET:
        state.nudge_time(1)
    elif key == pygame.K_LEFTBRACKET:
        state.nudge_time(-1)
    elif key == pygame.K_TAB:
        state.cycle_selection(-1 if event.mod & pygame.KMOD_SHIFT else 1)
    elif key == pygame.K_f:
        state.follow_selected()
    elif key == pygame.K_t:
        state.view.show_trails = not state.view.show_trails
    elif key == pygame.K_r:
        state.view.real_size = not state.view.real_size
    elif key == pygame.K_l:
        state.view.show_labels = not state.view.show_labels
    elif key == pygame.K_g:
        state.cycle_grid(ctx)
    elif key == pygame.K_0:
        state.stop_selected()
    elif key in (pygame.K_DELETE, pygame.K_BACKSPACE):
        state.delete_selected()
    elif key == pygame.K_F5:
        state.reset()
    elif key in (pygame.K_SLASH, pygame.K_QUESTION):
        state.help_visible = not state.help_visible
    elif key == pygame.K_ESCAPE:
        if state.help_visible:
            state.help_visible = False
        else:
            state.selected_id = None
```

- [ ] **Step 5: Extend AppState**

Replace the placeholder `hint` property added in Task 14 and add the new methods
to `src/app/state.py`:

```python
    # --- tools ------------------------------------------------------------------

    def set_tool(self, tool_cls):
        if self.tool is not None:
            self.tool.on_exit(self)
        self.tool = tool_cls()
        self.tool_name = self.tool.name

    @property
    def hint(self):
        return self.tool.hint if self.tool is not None else ""

    # --- global commands --------------------------------------------------------

    def stop_selected(self):
        """Zero the selected body's velocity.  Stop the Earth; watch it fall."""
        index = self.selected_index
        if index is None:
            return
        self.sim.world.vel[index] = 0.0
        self.sim.refresh()

    def reset(self):
        self.sim.reset()
        self.selected_id = None
        self.camera.follow_id = None

    def cycle_grid(self, ctx):
        """Step the pixel grid.  Sprites are sized in pixels, so the cache dies."""
        if ctx is None:
            return
        from .config import GRID_CHOICES
        nxt = GRID_CHOICES[(GRID_CHOICES.index(ctx.s) + 1) % len(GRID_CHOICES)]
        ctx.set_grid(nxt)
        self.on_resize(ctx)

    def on_resize(self, ctx):
        self.camera.width = ctx.width
        self.camera.height = ctx.height
```

and in `AppState.__init__`, add before `self.tool_name`:

```python
        self.tool = None
```

Note `tool_name` is set by `set_tool`; keep its initial value as `"Look"`.

- [ ] **Step 6: Run and verify it passes**

Run: `python -m pytest tests/test_tools_select.py -v`
Expected: PASS, 12 passed.

- [ ] **Step 7: Wire it into main.py**

Replace the event loop and drawing section of `src/app/main.py`:

```python
"""Entry point and frame loop."""

import sys

import pygame

from ..render import palette
from ..render.camera import Camera
from ..render.draw import ViewSettings, WorldRenderer
from ..render.hud import Hud
from ..render.pixelfont import PixelFont
from ..render.sprites import SpriteCache
from ..render.starfield import Starfield
from ..sim import Simulation
from .config import font_px, open_display
from .input import handle_event
from .state import AppState
from .tools import SelectTool


class App:
    def __init__(self):
        pygame.init()
        self.ctx = open_display()
        self.clock = pygame.time.Clock()
        self.cache = SpriteCache()
        self._build_render_stack()
        self.state = AppState(Simulation(), self.camera, ViewSettings())
        self.state.set_tool(SelectTool)

    def _build_render_stack(self):
        self.font = PixelFont(font_px(self.ctx.s))
        self.camera = getattr(self, "camera", None) or Camera(self.ctx.width,
                                                              self.ctx.height)
        self.camera.width, self.camera.height = self.ctx.width, self.ctx.height
        self.renderer = WorldRenderer(self.cache, self.font)
        self.hud = Hud(self.font)
        self.stars = Starfield(self.ctx.width, self.ctx.height)
        self.cache.clear()

    def run(self):
        state = self.state
        while state.running:
            grid = self.ctx.s
            for event in pygame.event.get():
                handle_event(event, state, self.ctx)
            if self.ctx.s != grid:
                self._build_render_stack()

            frame_dt = self.clock.tick(60) / 1000.0
            state.sim.advance(state.days_per_second * frame_dt)
            state.sim.update_trails()
            state.camera.update_follow(state.sim.world)

            surface = self.ctx.surface
            surface.fill(palette.SPACE)
            self.stars.draw(surface, state.camera)
            self.renderer.draw(surface, state.sim, state.camera, state.view,
                               selected_id=state.selected_id)
            state.tool.draw_overlay(surface, state, self.font)
            self.hud.draw(surface, state)
            pygame.display.flip()

        pygame.quit()
        return 0


def main():
    return App().run()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 8: Play with it**

Run: `python -m src.app.main`

Pause, speed up, click Jupiter, press `F` to ride it, `Tab` around the system,
`R` to watch everything collapse to dots, `?` for the controls.
**Milestone: it is now a working planetarium.**

- [ ] **Step 9: Commit**

```bash
git add src/app tests/test_tools_select.py
git commit -m "feat: tool framework, input routing, and a fully navigable planetarium"
```

---
## Task 16: LaunchTool and the live orbit preview

The best feature in the project. Because `orbits.py` already converts a state
vector to elements, the orbit a thrown body *will* follow can be drawn
algebraically — no integration, no guessing — and re-solved every frame while the
child drags. Pull harder and watch the ellipse stretch until it opens into a
hyperbola and the rock leaves forever.

**Files:**
- Create: `tests/test_tools_launch.py`
- Modify: `src/orbits.py`, `src/app/tools.py`

**Interfaces:**
- Consumes: `orbits`, `materials`, `units`, `Camera`, `AppState`
- Produces:
  - `orbits.conic_points(mu, rel_pos, rel_vel, samples=128) -> (np.ndarray shape (k,2), bool bound)` — points **relative to the attractor**
  - `PRESETS` — tuple of `LaunchPreset(key, label, material_key, radius_km, mass)` (mass `None` means derive from the material's density)
  - `LaunchTool(Tool)` with `name = "Throw"`, `.preset_index`
  - `preset_body(preset) -> (mass, radius)` in simulation units

- [ ] **Step 1: Write the failing tests**

`tests/test_tools_launch.py`:

```python
import math
import os

import numpy as np
import pygame
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    pygame.init()
    pygame.display.set_mode((320, 200))
    yield
    pygame.quit()


@pytest.fixture
def state():
    from src.app.state import AppState
    from src.app.tools import LaunchTool
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.sim import Simulation
    st = AppState(Simulation(), Camera(1000, 700), ViewSettings())
    st.set_tool(LaunchTool)
    return st


# --- the conic maths --------------------------------------------------------------

def test_circular_orbit_traces_a_circle():
    from src import orbits, units
    mu = units.G
    pos, vel = orbits.state_from_elements(mu, 1.0, 0.0, 0.0, 0.0)
    points, bound = orbits.conic_points(mu, pos, vel, samples=64)
    assert bound is True
    radii = np.hypot(points[:, 0], points[:, 1])
    assert np.allclose(radii, 1.0, rtol=1e-9)


def test_ellipse_spans_periapsis_to_apoapsis():
    from src import orbits, units
    mu = units.G
    a, e = 2.0, 0.4
    pos, vel = orbits.state_from_elements(mu, a, e, 0.0, 0.0)
    points, bound = orbits.conic_points(mu, pos, vel, samples=256)
    radii = np.hypot(points[:, 0], points[:, 1])
    assert bound is True
    assert radii.min() == pytest.approx(a * (1 - e), rel=1e-6)
    assert radii.max() == pytest.approx(a * (1 + e), rel=1e-6)


def test_escape_velocity_throw_is_reported_unbound():
    from src import orbits, units
    mu = units.G
    pos = np.array([1.0, 0.0])
    speed = math.sqrt(2.0 * mu / 1.0) * 1.3        # comfortably above escape
    vel = np.array([0.0, speed])
    points, bound = orbits.conic_points(mu, pos, vel, samples=64)
    assert bound is False
    assert len(points) >= 8
    assert np.isfinite(points).all()


def test_conic_of_a_body_at_rest_does_not_crash():
    from src import orbits, units
    points, bound = orbits.conic_points(units.G, np.array([1.0, 0.0]),
                                        np.zeros(2), samples=32)
    assert np.isfinite(points).all()


# --- the tool ---------------------------------------------------------------------

def test_presets_cover_the_promised_range():
    from src.app.tools import PRESETS
    labels = [p.label for p in PRESETS]
    assert len(PRESETS) == 5
    assert any("Asteroid" in x for x in labels)
    assert any("Star" in x for x in labels)


def test_preset_bodies_have_sane_masses():
    from src import units
    from src.app.tools import PRESETS, preset_body
    for preset in PRESETS:
        mass, radius = preset_body(preset)
        assert mass > 0 and radius > 0
        density = units.to_gcm3(units.density_from_mass_radius(mass, radius))
        assert 0.05 < density < 25.0, "%s has density %.3g g/cm3" % (preset.label,
                                                                     density)


def test_the_rocky_preset_is_about_an_earth():
    from src import units
    from src.app.tools import PRESETS, preset_body
    rocky = next(p for p in PRESETS if p.label == "Rocky planet")
    mass, radius = preset_body(rocky)
    assert radius == pytest.approx(units.EARTH_RADIUS, rel=1e-9)
    assert units.to_earth_masses(mass) == pytest.approx(1.0, rel=0.5)


def test_number_keys_choose_a_preset(state):
    state.tool.on_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_4, mod=0),
                        state)
    assert state.tool.preset_index == 3


def test_drag_and_release_adds_a_body(state):
    before = state.sim.world.n
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(500, 300),
                                           button=1), state)
    state.tool.on_event(pygame.event.Event(pygame.MOUSEMOTION, pos=(560, 300),
                                           rel=(60, 0), buttons=(1, 0, 0)), state)
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(560, 300),
                                           button=1), state)
    assert state.sim.world.n == before + 1


def test_the_launched_body_is_selected_so_you_can_watch_it(state):
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(500, 300),
                                           button=1), state)
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(520, 300),
                                           button=1), state)
    assert state.selected_id == state.sim.world.bodies[-1].id


def test_launch_velocity_follows_the_drag_direction(state):
    """Drag right, fly right."""
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(500, 300),
                                           button=1), state)
    state.tool.on_event(pygame.event.Event(pygame.MOUSEMOTION, pos=(600, 300),
                                           rel=(100, 0), buttons=(1, 0, 0)), state)
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(600, 300),
                                           button=1), state)
    assert state.sim.world.vel[state.sim.world.n - 1][0] > 0


def test_shift_snaps_to_a_circular_orbit(state):
    from src import orbits, units
    world = state.sim.world
    # Place well clear of the planets so the Sun is unambiguously the attractor.
    state.camera.centre = np.zeros(2)
    state.camera.log_zoom = math.log(1000 / 8.0)
    sx, sy = state.camera.world_to_screen(np.array([2.5, 0.0]))

    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                           pos=(int(sx), int(sy)), button=1), state)
    state.tool.on_event(pygame.event.Event(pygame.MOUSEMOTION,
                                           pos=(int(sx) + 40, int(sy)),
                                           rel=(40, 0), buttons=(1, 0, 0)), state)
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONUP,
                                           pos=(int(sx) + 40, int(sy)), button=1),
                        state)
    # Relaunch with shift held.
    index = world.n - 1
    world.remove(index)

    state.tool.shift = True
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                           pos=(int(sx), int(sy)), button=1), state)
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONUP,
                                           pos=(int(sx) + 40, int(sy)), button=1),
                        state)
    index = world.n - 1
    attractor = orbits.dominant_attractor(world, index)
    mu = units.G * (world.mass[attractor] + world.mass[index])
    el = orbits.elements_from_state(mu, world.pos[index] - world.pos[attractor],
                                    world.vel[index] - world.vel[attractor])
    assert el["e"] < 0.02, "shift should have produced a near-circular orbit"


def test_overlay_draws_while_dragging(state):
    from src.render.pixelfont import PixelFont
    from src.render import palette
    surface = pygame.Surface((1000, 700))
    surface.fill(palette.SPACE)
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(500, 300),
                                           button=1), state)
    state.tool.on_event(pygame.event.Event(pygame.MOUSEMOTION, pos=(600, 320),
                                           rel=(100, 20), buttons=(1, 0, 0)), state)
    state.tool.draw_overlay(surface, state, PixelFont(12))
    assert pygame.transform.average_color(surface)[:3] != palette.SPACE[:3]


def test_launching_into_a_full_world_is_refused_quietly(state):
    from src import materials, units
    world = state.sim.world
    while world.n < world.capacity:
        world.add("filler", materials.get("rock"), 1e-15, units.EARTH_RADIUS * 0.01,
                  np.array([500.0 + world.n, 0.0]), np.zeros(2))
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(500, 300),
                                           button=1), state)
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(520, 300),
                                           button=1), state)
    assert world.n == world.capacity
```

- [ ] **Step 2: Run and watch it fail**

Run: `python -m pytest tests/test_tools_launch.py -v`
Expected: FAIL — `module 'src.orbits' has no attribute 'conic_points'`.

- [ ] **Step 3: Add `conic_points` to `src/orbits.py`**

```python
def conic_points(mu, rel_pos, rel_vel, samples=128):
    """The path a body on this state vector will follow, relative to its attractor.

    Returns ``(points, bound)``.  This is algebra, not integration: the whole
    trajectory comes out of the orbital elements in closed form, which is what
    makes it cheap enough to redraw every frame while someone is aiming.

    For a hyperbola the true anomaly is only valid within +-acos(-1/e), so the
    sampling range is narrowed accordingly and the arc is drawn open-ended.
    """
    el = elements_from_state(mu, rel_pos, rel_vel)
    e = el["e"]
    bound = el["bound"]

    if bound:
        a = el["a"]
        p = a * (1.0 - e * e)
        nus = np.linspace(-math.pi, math.pi, samples)
    else:
        r = math.hypot(float(rel_pos[0]), float(rel_pos[1]))
        p = el["periapsis"] * (1.0 + e) if e > 0 else r
        if e > 1.0:
            limit = math.acos(min(1.0, max(-1.0, -1.0 / e))) * 0.985
        else:
            limit = math.pi * 0.985
        nus = np.linspace(-limit, limit, samples)

    if p <= 0.0 or not math.isfinite(p):
        return np.zeros((0, 2)), bound

    denominator = 1.0 + e * np.cos(nus)
    good = denominator > 1e-9
    nus = nus[good]
    radii = p / denominator[good]

    lon = el["lon_peri"]
    angles = nus + lon
    points = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    return points, bound
```

- [ ] **Step 4: Implement LaunchTool in `src/app/tools.py`**

```python
import math
from dataclasses import dataclass

import numpy as np

from .. import materials, orbits, units
from ..body import KIND_PLANET, KIND_STAR


@dataclass(frozen=True)
class LaunchPreset:
    key: int                 # 1..5, the number key that selects it
    label: str
    material_key: str
    radius_km: float
    #: Solar masses, or None to derive from the material's real density.
    mass: float = None


PRESETS = (
    LaunchPreset(1, "Asteroid", "rock", 50.0),
    LaunchPreset(2, "Rocky planet", "rock", 6371.0),
    LaunchPreset(3, "Ice planet", "ice", 2500.0),
    LaunchPreset(4, "Gas giant", "gas", 69911.0),
    LaunchPreset(5, "Star", "hydrogen", 0.3 * 695700.0, mass=0.3),
)

#: Drag pixels per AU/day of launch speed, at the current zoom.  Chosen so a
#: drag of a few dozen pixels lands near orbital speed at solar-system zoom.
DRAG_SCALE = 0.0025


def preset_body(preset):
    """(mass, radius) in simulation units for a launch preset."""
    radius = preset.radius_km / units.AU_KM
    if preset.mass is not None:
        return preset.mass, radius
    density = materials.get(preset.material_key).density
    return units.FOUR_THIRDS_PI * density * radius ** 3, radius


class LaunchTool(Tool):
    name = "Throw"
    hint = ("1-5 pick what to throw  -  press and drag to aim, release to launch  -  "
            "hold Shift for a perfect circular orbit")

    def __init__(self):
        self.preset_index = 0
        self.shift = False
        self._origin = None          # world position, set on press
        self._cursor = None          # screen position while dragging

    # --- events -----------------------------------------------------------------

    def on_event(self, event, state):
        import pygame

        if event.type == pygame.KEYDOWN:
            if pygame.K_1 <= event.key <= pygame.K_5 and event.key != pygame.K_1:
                self.preset_index = event.key - pygame.K_1
                return True
            if event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                self.shift = True
                return True
        if event.type == pygame.KEYUP and event.key in (pygame.K_LSHIFT,
                                                        pygame.K_RSHIFT):
            self.shift = False
            return True

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._origin = state.camera.screen_to_world(*event.pos)
            self._cursor = event.pos
            return True

        if event.type == pygame.MOUSEMOTION and self._origin is not None:
            self._cursor = event.pos
            return True

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 \
                and self._origin is not None:
            self._cursor = event.pos
            self._launch(state)
            self._origin = None
            self._cursor = None
            return True

        return False

    # --- physics ----------------------------------------------------------------

    def _drag_velocity(self, state):
        """Velocity implied by the current drag, in AU/day."""
        if self._origin is None or self._cursor is None:
            return np.zeros(2)
        start = state.camera.world_to_screen(self._origin)
        dx = self._cursor[0] - start[0]
        dy = self._cursor[1] - start[1]
        return np.array([dx, -dy]) * DRAG_SCALE / max(state.camera.zoom, 1e-9)

    def _tentative(self, state):
        """(position, velocity) the body would have if released right now."""
        pos = self._origin
        vel = self._drag_velocity(state)
        if self.shift:
            world = state.sim.world
            index = self._nearest_attractor(world, pos)
            if index is not None:
                rel = pos - world.pos[index]
                r = float(np.hypot(rel[0], rel[1]))
                if r > 0.0:
                    mu = units.G * world.mass[index]
                    tangent = np.array([-rel[1], rel[0]]) / r
                    # Match the drag's sense of rotation.
                    if float(np.dot(tangent, vel)) < 0.0:
                        tangent = -tangent
                    vel = world.vel[index] + tangent * math.sqrt(mu / r)
        return pos, vel

    @staticmethod
    def _nearest_attractor(world, pos):
        """Index of whatever pulls hardest at a point that is not yet a body."""
        if world.n == 0:
            return None
        d = world.positions - pos
        r2 = np.einsum("ij,ij->i", d, d)
        r2[r2 <= 0.0] = np.inf
        return int(np.argmax(world.masses / r2))

    def _launch(self, state):
        world = state.sim.world
        if world.n >= world.capacity:
            return
        preset = PRESETS[self.preset_index]
        mass, radius = preset_body(preset)
        pos, vel = self._tentative(state)

        kind = KIND_STAR if mass >= units.STAR_MASS_THRESHOLD else KIND_PLANET
        body = world.add(preset.label, materials.get(preset.material_key),
                         mass, radius, pos.copy(), vel,
                         kind=kind, seed=int(world.time * 1000) % (1 << 30) + world.n)
        if body is None:
            return
        state.selected_id = body.id
        state.sim.refresh()

    # --- overlay ----------------------------------------------------------------

    def draw_overlay(self, surface, state, font):
        import pygame

        from ..render import palette

        if self._origin is None or self._cursor is None:
            return

        camera = state.camera
        world = state.sim.world
        pos, vel = self._tentative(state)

        start = camera.world_to_screen(pos)
        pygame.draw.line(surface, palette.TEXT, start, self._cursor, 1)
        pygame.draw.circle(surface, palette.SELECT,
                           (int(start[0]), int(start[1])), 4, 1)

        index = self._nearest_attractor(world, pos)
        if index is not None:
            mu = units.G * (world.mass[index] + 0.0)
            points, bound = orbits.conic_points(
                mu, pos - world.pos[index], vel - world.vel[index], samples=160)
            if len(points) >= 2:
                projected = camera.world_to_screen_array(points + world.pos[index])
                colour = palette.AIM_GOOD if bound else palette.AIM_ESCAPE
                pygame.draw.lines(surface, colour, bound,
                                  projected.astype(int).tolist(), 1)

        speed = float(np.hypot(vel[0], vel[1]))
        label = "%.4g km/s" % units.to_kms(speed)
        if self.shift:
            label += "  (circular)"
        font.draw(surface, self._cursor[0] + 8, self._cursor[1] + 8,
                  label, palette.TEXT)
        font.draw(surface, self._cursor[0] + 8, self._cursor[1] + 8 + font.height,
                  PRESETS[self.preset_index].label, palette.TEXT_DIM)
```

- [ ] **Step 5: Register the tool**

At the bottom of `tools.py`, replace the `TOOLS` dict:

```python
TOOLS = {
    pygame.K_1: SelectTool,
    pygame.K_2: LaunchTool,
}
```

`LaunchTool` deliberately ignores `K_1` in its own key handler so that pressing
`1` while throwing switches back to the Look tool rather than selecting preset 1.
The asteroid preset is the default, so nothing is lost.

- [ ] **Step 6: Run and verify it passes**

Run: `python -m pytest tests/test_tools_launch.py -v`
Expected: PASS, 14 passed.

If `test_shift_snaps_to_a_circular_orbit` fails with an eccentricity near 1,
`_nearest_attractor` is picking a planet rather than the Sun — check that the
launch point really is at 2.5 AU and clear of Mars and Jupiter.

- [ ] **Step 7: Throw something at Jupiter**

Run: `python -m src.app.main`, press `2`, then `4` for a gas giant, and drag.
Watch the predicted ellipse swing around as you move the mouse; pull further and
it turns red and opens into a hyperbola.

- [ ] **Step 8: Commit**

```bash
git add src/orbits.py src/app/tools.py tests/test_tools_launch.py
git commit -m "feat: LaunchTool with exact algebraic orbit preview and circular snap"
```

---
## Task 17: EditTool

**The invariant: mass and density are edited; radius is always derived.** There
is no code path that can produce an inconsistent mass/radius/density triple, so
the physics stays honest no matter what the child does to Jupiter.

The rules live in `src/edits.py` — physics, not pixels — so they are testable
headless and the tool is a thin shell over them.

**Files:**
- Create: `src/edits.py`, `tests/test_edits.py`, `tests/test_tools_edit.py`
- Modify: `src/app/tools.py`, `tests/test_layering.py`

**Interfaces:**
- Consumes: `units`, `materials`, `World`
- Produces:
  - `MIN_MASS = 1e-16`, `MAX_MASS = 50.0` (solar masses)
  - `scrub_mass(mass, notches) -> float` — logarithmic, clamped
  - `scrub_density_gcm3(gcm3, notches) -> float` — linear, clamped
  - `set_mass(world, index, mass) -> None`
  - `set_density(world, index, density) -> None`
  - `set_material(world, index, material) -> None`
  - `cycle_material(world, index, step=1) -> None`
  - `body_density(world, index) -> float`
  - `ignition_ratio(world, index) -> float`
  - `EDIT_FIELDS = ("mass", "density", "material")`
  - `EditTool(Tool)` with `name = "Edit"`, `.field_index`

- [ ] **Step 1: Add `src.edits` to the layering test**

In `tests/test_layering.py`, append `"src.edits"` to `PHYSICS_MODULES`. The rule
only means something if new physics modules join it.

- [ ] **Step 2: Write the failing tests**

`tests/test_edits.py`:

```python
import numpy as np
import pytest

from src import edits, materials, units
from src.body import KIND_PLANET, KIND_STAR
from src.sim import Simulation


@pytest.fixture
def sim():
    return Simulation()


def _index(sim, name):
    return next(i for i, b in enumerate(sim.world.bodies) if b.name == name)


def test_mass_scrub_is_logarithmic():
    """Mass spans eight orders of magnitude; a linear scrub would be useless."""
    doubled = edits.scrub_mass(1e-6, 1)
    assert doubled > 1e-6
    ratio = doubled / 1e-6
    assert edits.scrub_mass(1e-3, 1) / 1e-3 == pytest.approx(ratio, rel=1e-9)


def test_mass_scrub_is_reversible():
    start = units.EARTH_MASS
    assert edits.scrub_mass(edits.scrub_mass(start, 5), -5) == pytest.approx(start)


def test_mass_scrub_clamps():
    assert edits.scrub_mass(1.0, 10_000) == pytest.approx(edits.MAX_MASS)
    assert edits.scrub_mass(1.0, -10_000) == pytest.approx(edits.MIN_MASS)


def test_density_scrub_clamps_to_the_material_range():
    assert edits.scrub_density_gcm3(1.0, 10_000) == pytest.approx(
        materials.MAX_DENSITY_GCM3)
    assert edits.scrub_density_gcm3(1.0, -10_000) == pytest.approx(
        materials.MIN_DENSITY_GCM3)


def test_setting_mass_keeps_density_and_grows_the_radius(sim):
    i = _index(sim, "Earth")
    before_density = edits.body_density(sim.world, i)
    before_radius = float(sim.world.radius[i])
    edits.set_mass(sim.world, i, units.EARTH_MASS * 8.0)
    assert edits.body_density(sim.world, i) == pytest.approx(before_density, rel=1e-9)
    # Eight times the mass at constant density is exactly twice the radius.
    assert sim.world.radius[i] == pytest.approx(before_radius * 2.0, rel=1e-9)


def test_squeezing_a_body_denser_really_does_shrink_it(sim):
    i = _index(sim, "Earth")
    before = float(sim.world.radius[i])
    edits.set_density(sim.world, i, edits.body_density(sim.world, i) * 8.0)
    assert sim.world.radius[i] == pytest.approx(before / 2.0, rel=1e-9)
    assert sim.world.mass[i] == pytest.approx(units.EARTH_MASS, rel=1e-9)


def test_changing_material_snaps_density_to_that_substance(sim):
    i = _index(sim, "Earth")
    edits.set_material(sim.world, i, materials.get("iron"))
    assert sim.world.bodies[i].material.key == "iron"
    assert edits.body_density(sim.world, i) == pytest.approx(
        materials.get("iron").density, rel=1e-9)


def test_cycling_material_wraps_around(sim):
    i = _index(sim, "Earth")
    seen = set()
    for _ in range(len(materials.MATERIALS) + 1):
        edits.cycle_material(sim.world, i)
        seen.add(sim.world.bodies[i].material.key)
    assert seen == {m.key for m in materials.MATERIALS}


def test_crossing_the_ignition_threshold_lights_a_star(sim):
    i = _index(sim, "Jupiter")
    assert sim.world.bodies[i].kind != KIND_STAR
    edits.set_mass(sim.world, i, units.STAR_MASS_THRESHOLD * 1.2)
    assert sim.world.bodies[i].kind == KIND_STAR


def test_dropping_below_the_threshold_puts_the_star_out(sim):
    i = _index(sim, "Sun")
    edits.set_mass(sim.world, i, units.EARTH_MASS)
    assert sim.world.bodies[i].kind == KIND_PLANET


def test_ignition_ratio_says_how_far_from_a_star_it_is(sim):
    i = _index(sim, "Jupiter")
    ratio = edits.ignition_ratio(sim.world, i)
    assert ratio == pytest.approx(units.STAR_MASS_THRESHOLD / units.JUPITER_MASS,
                                  rel=0.01)
    assert ratio > 1.0


def test_editing_mass_changes_the_forces_on_everything_else(sim):
    """Mass edits change the field for every body at once; stale accelerations
    would leave the system a step behind."""
    i = _index(sim, "Jupiter")
    before = sim.world.acc[_index(sim, "Earth")].copy()
    edits.set_mass(sim.world, i, units.JUPITER_MASS * 500.0)
    sim.refresh()
    after = sim.world.acc[_index(sim, "Earth")]
    assert not np.allclose(before, after)
```

`tests/test_tools_edit.py`:

```python
import os

import pygame
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    pygame.init()
    pygame.display.set_mode((320, 200))
    yield
    pygame.quit()


@pytest.fixture
def state():
    from src.app.state import AppState
    from src.app.tools import EditTool
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.sim import Simulation
    st = AppState(Simulation(), Camera(1000, 700), ViewSettings())
    st.set_tool(EditTool)
    st.select(next(b for b in st.sim.world.bodies if b.name == "Earth").id)
    return st


def _key(key):
    return pygame.event.Event(pygame.KEYDOWN, key=key, mod=0)


def test_up_and_down_move_between_fields(state):
    from src.app.tools import EDIT_FIELDS
    assert state.tool.field_index == 0
    state.tool.on_event(_key(pygame.K_DOWN), state)
    assert state.tool.field_index == 1
    for _ in range(10):
        state.tool.on_event(_key(pygame.K_DOWN), state)
    assert state.tool.field_index == len(EDIT_FIELDS) - 1


def test_right_arrow_increases_the_selected_field(state):
    index = state.selected_index
    before = float(state.sim.world.mass[index])
    state.tool.on_event(_key(pygame.K_RIGHT), state)
    assert state.sim.world.mass[state.selected_index] > before


def test_left_arrow_decreases_it(state):
    index = state.selected_index
    before = float(state.sim.world.mass[index])
    state.tool.on_event(_key(pygame.K_LEFT), state)
    assert state.sim.world.mass[state.selected_index] < before


def test_editing_density_changes_the_radius(state):
    before = float(state.sim.world.radius[state.selected_index])
    state.tool.on_event(_key(pygame.K_DOWN), state)      # to density
    state.tool.on_event(_key(pygame.K_RIGHT), state)
    assert state.sim.world.radius[state.selected_index] != before


def test_editing_material_changes_it(state):
    before = state.sim.world.bodies[state.selected_index].material.key
    state.tool.on_event(_key(pygame.K_DOWN), state)
    state.tool.on_event(_key(pygame.K_DOWN), state)      # to material
    state.tool.on_event(_key(pygame.K_RIGHT), state)
    assert state.sim.world.bodies[state.selected_index].material.key != before


def test_dragging_horizontally_scrubs(state):
    before = float(state.sim.world.mass[state.selected_index])
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(500, 300),
                                           button=1), state)
    state.tool.on_event(pygame.event.Event(pygame.MOUSEMOTION, pos=(560, 300),
                                           rel=(60, 0), buttons=(1, 0, 0)), state)
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(560, 300),
                                           button=1), state)
    assert state.sim.world.mass[state.selected_index] > before


def test_editing_with_nothing_selected_is_harmless(state):
    state.selected_id = None
    state.tool.on_event(_key(pygame.K_RIGHT), state)


def test_editing_a_destroyed_body_is_harmless(state):
    index = state.selected_index
    state.sim.world.remove(index)
    state.tool.on_event(_key(pygame.K_RIGHT), state)


def test_overlay_names_the_active_field(state):
    from src.render import palette
    from src.render.pixelfont import PixelFont
    surface = pygame.Surface((1000, 700))
    surface.fill(palette.SPACE)
    state.tool.draw_overlay(surface, state, PixelFont(12))
    assert pygame.transform.average_color(surface)[:3] != palette.SPACE[:3]
```

- [ ] **Step 3: Run and watch them fail**

Run: `python -m pytest tests/test_edits.py tests/test_tools_edit.py -v`
Expected: FAIL — `No module named 'src.edits'`.

- [ ] **Step 4: Implement `src/edits.py`**

```python
"""Editing a body that already exists.

**The invariant of this module: mass and density are what you set; radius is
always derived.**  ``materials.py`` was written for exactly this -- squeeze a
body denser and it really does shrink -- and honouring it means there is no code
path that can produce an inconsistent mass/radius/density triple, whatever the
child does to Jupiter.

Kind follows mass: cross ``units.STAR_MASS_THRESHOLD`` and the body ignites;
drop back below and it goes out.

Imports numpy and nothing from pygame -- ``tests/test_layering.py`` asserts it.
"""

import math

from . import materials, units
from .body import KIND_PLANET, KIND_STAR

#: A body may not be lighter than a small asteroid or heavier than a large star.
MIN_MASS = 1e-16
MAX_MASS = 50.0

#: Ratio per scrub notch for mass, and g/cm^3 per notch for density.
MASS_NOTCH = 1.06
DENSITY_NOTCH = 0.15

EDIT_FIELDS = ("mass", "density", "material")


# --- scrubbing --------------------------------------------------------------------

def scrub_mass(mass, notches):
    """Logarithmic scrub: every notch is the same *ratio*, not the same amount."""
    scaled = max(mass, MIN_MASS) * (MASS_NOTCH ** notches)
    return float(min(max(scaled, MIN_MASS), MAX_MASS))


def scrub_density_gcm3(gcm3, notches):
    """Linear scrub, clamped to the range real matter occupies."""
    value = gcm3 + notches * DENSITY_NOTCH
    return float(min(max(value, materials.MIN_DENSITY_GCM3),
                     materials.MAX_DENSITY_GCM3))


# --- reading ----------------------------------------------------------------------

def body_density(world, index):
    return units.density_from_mass_radius(float(world.mass[index]),
                                          float(world.radius[index]))


def ignition_ratio(world, index):
    """How many times heavier this body must become before it lights up."""
    mass = float(world.mass[index])
    if mass <= 0.0:
        return float("inf")
    return units.STAR_MASS_THRESHOLD / mass


# --- writing ----------------------------------------------------------------------

def _refresh_kind(world, index):
    body = world.bodies[index]
    if float(world.mass[index]) >= units.STAR_MASS_THRESHOLD:
        body.kind = KIND_STAR
    elif body.kind == KIND_STAR:
        body.kind = KIND_PLANET


def set_mass(world, index, mass):
    """Set mass, holding density constant, so the radius follows."""
    density = body_density(world, index)
    mass = float(min(max(mass, MIN_MASS), MAX_MASS))
    world.mass[index] = mass
    world.radius[index] = units.radius_from_mass_density(mass, density)
    _refresh_kind(world, index)


def set_density(world, index, density):
    """Set density, holding mass constant, so the radius follows."""
    lo = units.from_gcm3(materials.MIN_DENSITY_GCM3)
    hi = units.from_gcm3(materials.MAX_DENSITY_GCM3)
    density = float(min(max(density, lo), hi))
    world.radius[index] = units.radius_from_mass_density(
        float(world.mass[index]), density)
    # Keep the composition label honest about what it now is.
    world.bodies[index].material = materials.nearest_by_density(density)


def set_material(world, index, material):
    """Adopt a substance, snapping density to that substance's real value."""
    world.bodies[index].material = material
    world.radius[index] = units.radius_from_mass_density(
        float(world.mass[index]), material.density)


def cycle_material(world, index, step=1):
    keys = [m.key for m in materials.MATERIALS]
    current = keys.index(world.bodies[index].material.key)
    set_material(world, index, materials.MATERIALS[(current + step) % len(keys)])
```

- [ ] **Step 5: Implement EditTool in `src/app/tools.py`**

```python
from .. import edits

EDIT_FIELDS = edits.EDIT_FIELDS

#: Screen pixels of horizontal drag per scrub notch.
DRAG_PIXELS_PER_NOTCH = 6.0


class EditTool(Tool):
    name = "Edit"
    hint = ("Up/Down pick what to change  -  Left/Right or drag to change it  -  "
            "0 stops it dead  -  make it 80x heavier and it becomes a star")

    def __init__(self):
        self.field_index = 0
        self._drag_x = None

    def on_event(self, event, state):
        import pygame

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DOWN:
                self.field_index = min(len(EDIT_FIELDS) - 1, self.field_index + 1)
                return True
            if event.key == pygame.K_UP:
                self.field_index = max(0, self.field_index - 1)
                return True
            if event.key == pygame.K_RIGHT:
                self._scrub(state, 1)
                return True
            if event.key == pygame.K_LEFT:
                self._scrub(state, -1)
                return True

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._drag_x = event.pos[0]
            return True

        if event.type == pygame.MOUSEMOTION and self._drag_x is not None:
            travelled = event.pos[0] - self._drag_x
            notches = int(travelled / DRAG_PIXELS_PER_NOTCH)
            if notches:
                self._scrub(state, notches)
                self._drag_x += notches * DRAG_PIXELS_PER_NOTCH
            return True

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._drag_x = None
            return True

        return False

    def _scrub(self, state, notches):
        index = state.selected_index
        if index is None:
            return
        world = state.sim.world
        field = EDIT_FIELDS[self.field_index]

        if field == "mass":
            edits.set_mass(world, index,
                           edits.scrub_mass(float(world.mass[index]), notches))
        elif field == "density":
            current = units.to_gcm3(edits.body_density(world, index))
            edits.set_density(world, index,
                              units.from_gcm3(edits.scrub_density_gcm3(current,
                                                                       notches)))
        else:
            edits.cycle_material(world, index, 1 if notches > 0 else -1)

        state.sim.refresh()

    def draw_overlay(self, surface, state, font):
        from ..render import palette

        index = state.selected_index
        if index is None:
            font.draw(surface, 12, surface.get_height() // 2,
                      "Click a body first, then edit it", palette.WARN)
            return

        world = state.sim.world
        values = {
            "mass": "%.4g Earths" % units.to_earth_masses(float(world.mass[index])),
            "density": "%.3g g/cm3" % units.to_gcm3(edits.body_density(world, index)),
            "material": world.bodies[index].material.name,
        }
        y = surface.get_height() // 2 - len(EDIT_FIELDS) * font.height
        for k, field in enumerate(EDIT_FIELDS):
            active = k == self.field_index
            colour = palette.SELECT if active else palette.TEXT_DIM
            marker = "> " if active else "  "
            font.draw(surface, 12, y + k * (font.height + 3),
                      "%s%-9s %s" % (marker, field, values[field]), colour)
```

- [ ] **Step 6: Register the tool**

```python
TOOLS = {
    pygame.K_1: SelectTool,
    pygame.K_2: LaunchTool,
    pygame.K_3: EditTool,
}
```

- [ ] **Step 7: Run and verify**

Run: `python -m pytest tests/test_edits.py tests/test_tools_edit.py -v`
Expected: PASS, 12 + 9 passed.

- [ ] **Step 8: Run the whole suite**

Run: `python selftest.py`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/edits.py src/app/tools.py tests/test_edits.py tests/test_tools_edit.py tests/test_layering.py
git commit -m "feat: EditTool with derived radius, live ignition and material cycling"
```

---

## Task 18: Ship it

**Files:**
- Create: `tests/test_smoke.py`, `run.command`, `README.md`

**Interfaces:**
- Consumes: everything
- Produces: a double-clickable app and a README

- [ ] **Step 1: Write the smoke test**

`tests/test_smoke.py`:

```python
"""Boot the real application headless and run frames through it.

Unit tests cover each piece; this is the one that catches the wiring -- a bad
argument order in main.py, a tool that crashes on its first overlay, a HUD that
divides by zero when the last body is destroyed.
"""

import os

import pygame
import pytest

os.environ["SDL_VIDEODRIVER"] = "dummy"


@pytest.fixture
def app():
    from src.app.main import App
    instance = App()
    yield instance
    pygame.quit()


def test_a_hundred_frames_run_without_raising(app):
    for _ in range(100):
        app.step_once()


def test_frames_run_with_every_tool_active(app):
    from src.app.tools import EditTool, LaunchTool, SelectTool
    for tool in (SelectTool, LaunchTool, EditTool):
        app.state.set_tool(tool)
        app.state.select(app.state.sim.world.bodies[3].id)
        for _ in range(10):
            app.step_once()


def test_frames_run_with_the_help_overlay_open(app):
    app.state.help_visible = True
    for _ in range(5):
        app.step_once()


def test_frames_run_in_real_size_mode(app):
    app.state.view.real_size = True
    for _ in range(5):
        app.step_once()


def test_frames_run_after_everything_is_destroyed(app):
    app.state.sim.world.clear()
    for _ in range(5):
        app.step_once()


def test_frames_run_at_maximum_time_scale(app):
    from src.app.state import TIME_SCALES
    app.state.time_index = len(TIME_SCALES) - 1
    for _ in range(10):
        app.step_once()
    assert app.state.sim.world.time > 0
```

- [ ] **Step 2: Run and watch it fail**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: FAIL — `App has no attribute 'step_once'`.

- [ ] **Step 3: Split one frame out of the loop in `src/app/main.py`**

Replace `run` with:

```python
    def step_once(self):
        """One frame: events, physics, drawing.  Extracted so the smoke test can
        drive the real application without owning the loop."""
        grid = self.ctx.s
        for event in pygame.event.get():
            handle_event(event, self.state, self.ctx)
        if self.ctx.s != grid:
            self._build_render_stack()

        state = self.state
        frame_dt = self.clock.tick(60) / 1000.0
        state.sim.advance(state.days_per_second * frame_dt)
        state.sim.update_trails()
        state.camera.update_follow(state.sim.world)

        surface = self.ctx.surface
        surface.fill(palette.SPACE)
        self.stars.draw(surface, state.camera)
        self.renderer.draw(surface, state.sim, state.camera, state.view,
                           selected_id=state.selected_id)
        state.tool.draw_overlay(surface, state, self.font)
        self.hud.draw(surface, state)
        pygame.display.flip()

    def run(self):
        while self.state.running:
            self.step_once()
        pygame.quit()
        return 0
```

- [ ] **Step 4: Run and verify it passes**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: PASS, 6 passed.

- [ ] **Step 5: Write the launcher**

`run.command`:

```bash
#!/bin/bash
# Double-click me in Finder.
cd "$(dirname "$0")" || exit 1

if [ ! -d .venv ]; then
    echo "First run: setting up..."
    python3 -m venv .venv || exit 1
    ./.venv/bin/pip install --quiet --upgrade pip
    ./.venv/bin/pip install --quiet -r requirements.txt || exit 1
fi

exec ./.venv/bin/python -m src.app.main
```

Then: `chmod +x run.command`

- [ ] **Step 6: Write the README**

`README.md`:

````markdown
# Solar System Sandbox

A 2D pixel-art solar system you can break.

The planets are not animated. Each one starts from its real J2000 orbital
elements and moves because gravity is being integrated, which is why Neptune's
year comes out at 165 years and why throwing something heavy at Jupiter changes
everything else's orbit too.

## Running it

Double-click `run.command`. First launch sets up a virtual environment; after
that it opens straight away.

From a terminal instead:

```bash
pip install -r requirements.txt
python -m src.app.main
```

## Controls

| | |
|---|---|
| `Space` | pause and resume |
| `[` `]` | slower / faster (1 hour per second up to 10 years per second) |
| Wheel | zoom at the cursor |
| Drag | pan |
| Click | select a body; double-click flies to it |
| `Tab` | cycle through bodies |
| `F` | follow the selected body |
| `1` `2` `3` | Look / Throw / Edit |
| `0` | stop the selected body dead |
| `T` `R` `L` | trails / real sizes / labels |
| `G` | pixel grid size |
| `Del` | delete the selected body |
| `F5` | start over |
| `?` | controls |
| `Cmd+F` | fullscreen |

## Things worth trying

- Select Earth, press `0`, and watch it fall into the Sun.
- Press `R`. Everything in the solar system becomes a single dot at once — that
  is what the real scale looks like.
- Throw tool, pick a gas giant, and drag. The green ellipse is the exact orbit
  it will follow, worked out algebraically as you aim. Pull further and it turns
  red: that one is leaving forever.
- Hold `Shift` while aiming to snap to a perfect circular orbit.
- Edit tool on Jupiter. Make it about 80 times heavier and it ignites into a
  second star, and every shadow in the system swings around.
- Throw something fast enough at a planet and it will not merge — it shatters
  into fragments that fly apart, then slowly pull themselves back together.

## Tests

```bash
python selftest.py
```

The physics layer never imports pygame, which is what lets the orbital mechanics
be tested without a display. One of the tests enforces that.

## How it is put together

| | |
|---|---|
| `src/` | physics: units, gravity, orbits, collisions. Pure numpy. |
| `src/render/` | camera, sprites, HUD. Everything that touches pixels. |
| `src/app/` | frame loop, input, one class per tool. |

Design notes are in `docs/superpowers/specs/`.
````

- [ ] **Step 7: Run everything one last time**

Run: `python selftest.py`
Expected: PASS, all green.

Run: `./run.command`
Expected: the game opens fullscreen.

- [ ] **Step 8: Commit**

```bash
git add tests/test_smoke.py run.command README.md src/app/main.py
git commit -m "feat: smoke test, double-clickable launcher, README"
```

---

## Self-review notes

Checked against the spec:

- Every spec section maps to a task. Display/resolution -> 7; size law -> 10;
  sprites, trails, starfield -> 11, 12; simulation loop and budget -> 6;
  collisions -> 3, 4, 5; interaction -> 15, 16, 17; HUD -> 14; testing -> spread
  across all tasks plus 18; delivery -> 18.
- Two deliberate deviations from the spec, both recorded where they occur:
  **Task 8** uses pygame's built-in font with antialiasing off rather than an
  embedded 5x7 bitmap font, behind a narrow interface that can be swapped later.
  **Task 17** puts editing rules in a new `src/edits.py` rather than inside the
  tool, so they stay headless-testable.
- The one open risk in the spec — whether SDL exposes the true Retina backing
  store on macOS — is Task 7 Step 7, with both outcomes and their remedies
  written out.
