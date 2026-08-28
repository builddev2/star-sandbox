# Black Holes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a black hole body kind the child can throw, collapse a star into, and edit — with real Schwarzschild/Kerr physics, an accretion-disk sprite, and an interface that never offers a control that means nothing.

**Architecture:** A fourth `kind` alongside planet, star and debris. Physics gets new formulas in `units.py` and a branch beside existing logic in `edits.py`, `collisions.py` and `destruction.py` — no rewrites. The render layer gains one helper, `scale.visual_radii(world)`, that solves both the drawing and the aiming problem at once, and one new sprite function. The app layer extends the Throw presets, the Edit panel and the inspector.

**Tech Stack:** Python 3.11, numpy, pygame 2.6, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-27-black-holes-design.md`

## Global Constraints

Copied verbatim from the spec and from `CLAUDE.md`. **Every task's requirements implicitly include this section.**

- **Simulation units are AU, solar masses, days.** `src/units.py` is the single source of truth for `G`, `c` and every conversion factor. No other module may define one.
- **`src/*.py` must never import pygame.** `tests/test_layering.py` asserts it in a subprocess. `spin` and `disk` live on `Body`, not in World's arrays, because the force kernel never reads them.
- **Never hold an array index across a frame.** `World.remove` is a swap-remove. Store `body.id` and resolve through `World.index_of(id)`.
- **Every mutation of mass, radius or membership must be followed by `Simulation.refresh()`.**
- **Mass is set; radius is always derived.** For an ordinary body by density; for a black hole by `r_s = 2GM/c^2`. There is no code path allowed to produce an inconsistent triple.
- **`World.contact_radius` is not `World.radius`.** Every collision *outcome* uses true `radius`. Only "did these two touch" uses `contact_radius`. Any mutation of `radius` must reset `contact_radius` to match.
- **Every action has a button, and every button carries its hotkey.** A control names an `AppState` or tool method the key already called. `tests/test_ui.py` reads `hud.KEY_HELP` and fails if a documented key has no button.
- **Nothing may cache a size across frames.** The window is resizable.
- **Icons are drawn from lines, not typed as glyphs.**
- **Tests that touch pygame set `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")` at module top, before importing pygame.**
- `World.MAX_BODIES` is 200 and must not be raised.
- Angles are radians everywhere except the source data tables in `src/scenarios.py`.

**Commands:**

```bash
python selftest.py                                  # whole suite (553 tests, ~50s)
python -m pytest tests/test_black_hole.py -v         # one file
python -m pytest tests/test_black_hole.py::test_name -v
./.venv/bin/python                                   # if system python lacks pygame/numpy
```

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/units.py` | modify | All black hole formulas and constants. The only place `c` is written down. |
| `src/body.py` | modify | `KIND_BLACK_HOLE`, `spin`/`disk` fields, creation defaults |
| `src/materials.py` | modify | The `SINGULARITY` material, kept out of the cycle-able tuple |
| `src/edits.py` | modify | Collapse, the black-hole mass law, spin/disk scrubs, per-kind field lists |
| `src/collisions.py` | modify | The hole takes the `i` slot, always accretes, never sheds ejecta; `v_impact` clamp |
| `src/destruction.py` | modify | Black holes raise tides; stars become victims of black holes only |
| `src/scenarios.py` | modify | `black_hole()` starting arrangement |
| `src/render/scale.py` | modify | `visual_radii(world)` plus the two optional arguments |
| `src/render/sprites.py` | modify | `_make_black_hole` and `SpriteCache.get_black_hole` |
| `src/render/effects.py` | modify | The `swallow` outcome |
| `src/render/draw.py` | modify | Route black holes to the new sprite; pass the floor radii |
| `src/render/widgets.py` | modify | The `blackhole` icon |
| `src/render/hud.py` | modify | Black-hole inspector rows; `C` in `KEY_HELP` |
| `src/app/state.py` | modify | Pass floor radii when picking; scenario selection |
| `src/app/main.py` | modify | Pass aim radii to the contact refresh |
| `src/app/tools.py` | modify | Throw preset 6; the Edit panel's per-kind fields and Collapse |
| `src/app/ui.py` | modify | `edit.collapse` and `scenario.cycle` dispatch |
| `tests/test_black_hole.py` | create | Physics: formulas, collapse, collisions, tides |
| `tests/test_black_hole_render.py` | create | Sprite, visual radii, swallow effect |

---

### Task 1: Black hole formulas in `units.py`

**Files:**
- Modify: `src/units.py` (append after `star_temperature`, before `circular_speed`)
- Test: `tests/test_black_hole.py` (create)

**Interfaces:**
- Consumes: nothing
- Produces: `units.C`, `units.C_KMS`, `units.MAX_SPIN`, `units.TOV_LIMIT_MASS`, `units.COLLAPSE_MASS_THRESHOLD`, `units.SHADOW_FACTOR`, `units.PHOTON_SPHERE_FACTOR`, `units.schwarzschild_radius(mass)`, `units.shadow_radius(mass)`, `units.photon_sphere_radius(mass)`, `units.isco_radius(mass, spin=0.0)`, `units.accretion_efficiency(spin=0.0)`, `units.hawking_temperature_k(mass)`, `units.evaporation_years(mass)`, `units.apparent_radius(mass)`. All masses in solar masses, all radii in AU.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_black_hole.py`:

```python
"""Black hole physics: formulas, collapse, collisions and tides.

Pure numpy -- no pygame anywhere in this file, so it exercises the same
layer tests/test_layering.py protects.
"""

import math

import pytest

from src import units


def test_speed_of_light_in_simulation_units():
    """c is derived from the conversion factors, never written down twice."""
    assert units.C == pytest.approx(173.1446326742, abs=1e-9)


def test_schwarzschild_radius_matches_the_textbook_value():
    """2GM/c^2 for one solar mass is 2.953 km."""
    r_s = units.schwarzschild_radius(1.0)
    assert units.to_km(r_s) == pytest.approx(2.95325, rel=1e-5)


def test_schwarzschild_radius_is_linear_in_mass():
    assert units.schwarzschild_radius(10.0) == pytest.approx(
        10.0 * units.schwarzschild_radius(1.0))


def test_escape_velocity_at_the_horizon_is_exactly_c():
    """Falls out of the existing formula with no special case, which is a
    consistency check on the whole unit system."""
    for mass in (1.0, 10.0, 150.0):
        r_s = units.schwarzschild_radius(mass)
        assert units.escape_velocity(mass, r_s) == pytest.approx(units.C, rel=1e-12)


def test_density_scales_as_inverse_mass_squared():
    """Bigger black holes are less dense -- the counterintuitive fact."""
    light = units.density_from_mass_radius(1.0, units.schwarzschild_radius(1.0))
    heavy = units.density_from_mass_radius(100.0, units.schwarzschild_radius(100.0))
    assert heavy / light == pytest.approx(1.0 / 100.0 ** 2, rel=1e-9)


def test_shadow_is_larger_than_the_photon_sphere_is_larger_than_the_horizon():
    r_s = units.schwarzschild_radius(10.0)
    assert units.photon_sphere_radius(10.0) == pytest.approx(1.5 * r_s)
    assert units.shadow_radius(10.0) == pytest.approx(math.sqrt(27.0) / 2.0 * r_s)
    assert units.shadow_radius(10.0) > units.photon_sphere_radius(10.0) > r_s


@pytest.mark.parametrize("spin,gm_c2", [
    (0.0, 6.0000),
    (0.5, 4.2330),
    (0.9, 2.3209),
    (0.998, 1.2370),
])
def test_isco_matches_bardeen_press_teukolsky(spin, gm_c2):
    """Prograde Kerr ISCO, in units of GM/c^2 -- which is half of r_s."""
    mass = 7.0
    expected = gm_c2 * 0.5 * units.schwarzschild_radius(mass)
    assert units.isco_radius(mass, spin) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("spin,efficiency", [
    (0.0, 0.0572),
    (0.998, 0.3210),
])
def test_accretion_efficiency_rises_with_spin(spin, efficiency):
    assert units.accretion_efficiency(spin) == pytest.approx(efficiency, abs=5e-4)


def test_hawking_temperature_is_absurdly_cold_for_a_stellar_hole():
    """6e-8 K -- colder than the CMB, so such a hole grows rather than evaporates."""
    assert units.hawking_temperature_k(1.0) == pytest.approx(6.169e-8, rel=1e-3)
    assert units.hawking_temperature_k(10.0) == pytest.approx(6.169e-9, rel=1e-3)


def test_evaporation_time_scales_as_mass_cubed():
    assert units.evaporation_years(10.0) / units.evaporation_years(1.0) == \
        pytest.approx(1000.0, rel=1e-9)


def test_apparent_radius_is_big_enough_to_draw():
    """The horizon is invisible; the apparent radius stands in for the disk.

    A 10 solar mass hole must read at least as large as the Sun does, or it
    draws smaller than a fragment of ejecta.
    """
    assert units.apparent_radius(10.0) > units.SUN_RADIUS
    assert units.apparent_radius(3.0) > units.SUN_RADIUS
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_black_hole.py -v`
Expected: FAIL — `AttributeError: module 'src.units' has no attribute 'C'`

- [ ] **Step 3: Implement the formulas**

In `src/units.py`, add `C_KMS` and `C` to the **Length/Time derived** section (after `AUD2_TO_MS2`):

```python
#: The speed of light.  Every other module asks for it here; the whole point
#: of this file is that a constant is written down once.
C_KMS = 299792.458
#: ... and in simulation units, AU per day: 173.1446 AU/day.
C = C_KMS * SEC_PER_DAY / AU_KM
```

Then append this block after `star_temperature`:

```python
# --- Black holes --------------------------------------------------------------------
#
# A black hole is completely described by mass, angular momentum and charge (the
# "no-hair" theorem).  Astrophysical holes carry no net charge -- the surrounding
# plasma neutralises any imbalance almost at once -- so there are two real
# parameters here, and everything below is derived from them.

#: Dimensionless spin ceiling.  Not 1: radiation from the accretion disk torques
#: a hole back down before it can reach the extremal limit (Thorne 1974).
MAX_SPIN = 0.998

#: Tolman-Oppenheimer-Volkoff limit: the heaviest a neutron star can be before it
#: collapses.  Below this nature makes neutron stars, not black holes.  Reported
#: in the inspector, deliberately not enforced -- see the design doc.
TOV_LIMIT_MASS = 3.0

#: Past this mass a star does not exist; the core collapses directly to a black
#: hole with no supernova at all (above the pair-instability window).  It is
#: also ``edits.MAX_MASS``, so the mass scrub tops out exactly where nature does.
COLLAPSE_MASS_THRESHOLD = 150.0

#: The dark disc a black hole casts is not the horizon -- light bending magnifies
#: it to sqrt(27)/2 horizon radii, about 2.6x.
SHADOW_FACTOR = math.sqrt(27.0) / 2.0
#: Where photons orbit in circles, at 1.5 horizon radii.
PHOTON_SPHERE_FACTOR = 1.5

#: Hawking temperature of a one-solar-mass hole, kelvin.  Sixty nanokelvin --
#: far below the 2.7 K microwave background, which is why a stellar-mass hole
#: absorbs more than it radiates and grows instead of evaporating.
HAWKING_T_SUN_K = 6.169e-8
#: Evaporation time of a one-solar-mass hole, years.  Scales as M^3.
EVAPORATION_SUN_YEARS = 2.1e67

#: Main-sequence mass-radius exponent, used only to give a black hole a drawable
#: apparent size: the disk is roughly as big as the star that made it.
APPARENT_RADIUS_EXP = 0.8


def schwarzschild_radius(mass):
    """Event horizon radius 2GM/c^2, in AU.  2.953 km per solar mass."""
    return 2.0 * G * max(float(mass), 0.0) / (C * C)


def shadow_radius(mass):
    """Radius of the dark disc an observer actually sees, in AU."""
    return SHADOW_FACTOR * schwarzschild_radius(mass)


def photon_sphere_radius(mass):
    """Radius at which light orbits in a circle, in AU."""
    return PHOTON_SPHERE_FACTOR * schwarzschild_radius(mass)


def isco_radius(mass, spin=0.0):
    """Innermost stable circular orbit (prograde), in AU.

    Bardeen, Press & Teukolsky (1972).  This is the inner edge of an accretion
    disk: inside it there is no stable orbit and matter plunges.  Spin drags it
    from 3 horizon radii down to 0.62, which is why a fast-spinning hole
    converts 32% of what it eats into light against a still one's 5.7%.
    """
    a = min(abs(float(spin)), MAX_SPIN)
    z1 = 1.0 + (1.0 - a * a) ** (1.0 / 3.0) * (
        (1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0))
    z2 = math.sqrt(3.0 * a * a + z1 * z1)
    r_gm = 3.0 + z2 - math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))
    # r_gm is in units of GM/c^2, which is half a Schwarzschild radius.
    return 0.5 * r_gm * schwarzschild_radius(mass)


def accretion_efficiency(spin=0.0):
    """Fraction of infalling rest mass radiated away, 0..1.

    One minus the specific binding energy at the ISCO.  5.7% for a still hole,
    32% at the Thorne limit -- against 0.7% for hydrogen fusion, which is why
    accreting black holes outshine the galaxies they sit in.
    """
    a = min(abs(float(spin)), MAX_SPIN)
    z1 = 1.0 + (1.0 - a * a) ** (1.0 / 3.0) * (
        (1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0))
    z2 = math.sqrt(3.0 * a * a + z1 * z1)
    r_gm = 3.0 + z2 - math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))
    return 1.0 - math.sqrt(1.0 - 2.0 / (3.0 * r_gm))


def hawking_temperature_k(mass):
    """Temperature of the hole's own radiation, kelvin.  Scales as 1/M."""
    mass = float(mass)
    return HAWKING_T_SUN_K / mass if mass > 0.0 else math.inf


def evaporation_years(mass):
    """How long the hole takes to evaporate, years.  Scales as M^3."""
    return EVAPORATION_SUN_YEARS * max(float(mass), 0.0) ** 3


def apparent_radius(mass):
    """The size a black hole is *drawn* from, in AU -- not a physical radius.

    The horizon is seven orders of magnitude below anything else in the sim, so
    a hole would draw as a single pixel and, worse, be impossible to aim at
    (see render/scale.py).  This stands in for the accretion disk, using the
    main-sequence mass-radius relation on the grounds that the disk is about as
    big as the star that made the hole.  **Physics must never read this.**
    """
    return SUN_RADIUS * max(float(mass), 0.0) ** APPARENT_RADIUS_EXP
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_black_hole.py -v`
Expected: PASS (16 tests)

- [ ] **Step 5: Run the full suite to check nothing regressed**

Run: `python selftest.py`
Expected: all existing tests still pass — this task adds only new names.

- [ ] **Step 6: Commit**

```bash
git add src/units.py tests/test_black_hole.py
git commit -m "feat: black hole formulas in units.py

Schwarzschild radius, shadow, photon sphere, Kerr ISCO and accretion
efficiency (Bardeen, Press & Teukolsky 1972), Hawking temperature and
evaporation time.

c is derived from the existing conversion factors rather than written
down again, and escape_velocity(M, r_s) then returns exactly c with no
special case -- a consistency check on the whole unit system.

apparent_radius is explicitly not physics: it is the size the render
layer draws a hole from, since the horizon is seven orders of magnitude
below anything else in the simulation."
```

---

### Task 2: The black hole kind and the singularity material

**Files:**
- Modify: `src/body.py`
- Modify: `src/materials.py`
- Test: `tests/test_black_hole.py` (append)

**Interfaces:**
- Consumes: `units` from Task 1
- Produces: `body.KIND_BLACK_HOLE == "black hole"`, `Body.spin` (float), `Body.disk` (float), `Body.is_black_hole` (property), `body.DEFAULT_BLACK_HOLE_SPIN == 0.7`, `body.DEFAULT_BLACK_HOLE_DISK == 0.6`, `materials.SINGULARITY` (a `Material` with key `"singularity"`), reachable via `materials.get("singularity")` but **absent from** `materials.MATERIALS`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_black_hole.py`:

```python
# --- the kind and the material -------------------------------------------------

def test_black_hole_kind_exists_and_bodies_report_it():
    from src.body import KIND_BLACK_HOLE, Body
    from src import materials
    b = Body(1, "Hole", materials.SINGULARITY, kind=KIND_BLACK_HOLE)
    assert b.is_black_hole
    assert not b.is_star


def test_ordinary_bodies_are_not_black_holes():
    from src.body import Body
    from src import materials
    assert not Body(1, "Earth", materials.get("rock")).is_black_hole


def test_new_bodies_start_with_no_spin_and_no_disk():
    from src.body import Body
    from src import materials
    b = Body(1, "Earth", materials.get("rock"))
    assert b.spin == 0.0
    assert b.disk == 0.0


def test_singularity_is_reachable_by_key_but_not_cycleable():
    """It must not appear in MATERIALS, or the Edit tool's material cycle and
    nearest_by_density could both land on it."""
    from src import materials
    assert materials.get("singularity") is materials.SINGULARITY
    assert materials.SINGULARITY not in materials.MATERIALS
    assert all(m.key != "singularity" for m in materials.MATERIALS)


def test_nearest_by_density_can_never_return_the_singularity():
    from src import materials
    for density_gcm3 in (0.01, 1.0, 25.0, 1e12):
        chosen = materials.nearest_by_density(units.from_gcm3(density_gcm3))
        assert chosen.key != "singularity"


def test_singularity_never_boils_and_never_shatters():
    from src import materials
    assert materials.SINGULARITY.vapour_temperature_k == math.inf
    assert materials.SINGULARITY.toughness > 1e6
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_black_hole.py -k "kind or singularity or spin or nearest" -v`
Expected: FAIL — `ImportError: cannot import name 'KIND_BLACK_HOLE'`

- [ ] **Step 3: Implement in `src/body.py`**

Add the kind constant beside the others:

```python
KIND_PLANET = "planet"
KIND_STAR = "star"
KIND_DEBRIS = "debris"
KIND_BLACK_HOLE = "black hole"
```

Add the creation defaults below `TRAIL_ANGLE_STEP`:

```python
#: What a freshly made black hole looks like.  A hole with no spin and no disk
#: is a black disc and nothing else -- correct for a starved one, but a poor
#: first impression, so anything the child creates arrives already feeding.
DEFAULT_BLACK_HOLE_SPIN = 0.7
DEFAULT_BLACK_HOLE_DISK = 0.6
```

Extend `Body.__slots__` and `__init__`:

```python
    __slots__ = ("id", "name", "kind", "material", "trail", "born_day",
                 "event_id", "no_collide_until", "seed", "rings", "spin", "disk")
```

At the end of `__init__`, after `self.rings = False`:

```python
        #: Dimensionless spin, 0 .. units.MAX_SPIN.  Black holes only; it moves
        #: the accretion disk's inner edge and turns the jets on.
        self.spin = 0.0
        #: Feeding rate, 0 .. 1, driving how bright and how wide the accretion
        #: disk is drawn.  0 is a starved hole: shadow and photon ring only.
        self.disk = 0.0
```

And the property beside `is_star`:

```python
    @property
    def is_black_hole(self):
        return self.kind == KIND_BLACK_HOLE
```

- [ ] **Step 4: Implement in `src/materials.py`**

Add after the `MATERIALS` tuple, **before** `BY_KEY`:

```python
#: A black hole is not made of anything, but every code path that draws, labels
#: or trails a body wants a Material.  This one exists to satisfy them, and its
#: tones are the accretion disk's colours so that trails and labels read hot.
#:
#: Deliberately **not** in MATERIALS: the Edit tool cycles that tuple and
#: nearest_by_density searches it, and neither may ever land here.  A black
#: hole's radius comes from its Schwarzschild radius, so density_gcm3 is never
#: read -- it is present only because the dataclass requires it.
SINGULARITY = Material(
    "singularity", "Singularity", 1.0,
    ((0x40, 0x08, 0x04), (0xE0, 0x70, 0x18), (0xFF, 0xF0, 0xD0)),
    "disk",
    "Not made of anything. Just gravity, wound too tight to let go.",
    toughness=1e9, vapour_temperature_k=float("inf"),
)
```

Then, immediately after `BY_KEY = {m.key: m for m in MATERIALS}`:

```python
# Reachable by key -- the Throw preset and collapse both ask for it by name --
# without being reachable by cycling or by density search.
BY_KEY[SINGULARITY.key] = SINGULARITY
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_black_hole.py -v`
Expected: PASS (22 tests)

- [ ] **Step 6: Run the full suite**

Run: `python selftest.py`
Expected: all pass. `Body.__slots__` grew, which nothing pickles or iterates.

- [ ] **Step 7: Commit**

```bash
git add src/body.py src/materials.py tests/test_black_hole.py
git commit -m "feat: black hole kind and the singularity material

spin and disk live on Body beside rings and seed rather than in World's
arrays, because the O(n^2) force kernel never reads them.

SINGULARITY is reachable through materials.get but kept out of the
MATERIALS tuple, so neither the Edit tool's material cycle nor
nearest_by_density can ever land on it."
```

---

### Task 3: Collapse, and the black hole mass law

**Files:**
- Modify: `src/edits.py`
- Test: `tests/test_black_hole.py` (append)

**Interfaces:**
- Consumes: `units` (Task 1), `body.KIND_BLACK_HOLE`, `body.DEFAULT_BLACK_HOLE_SPIN`, `body.DEFAULT_BLACK_HOLE_DISK`, `materials.SINGULARITY` (Task 2)
- Produces: `edits.MAX_MASS == 150.0`, `edits.SPIN_NOTCH == 0.02`, `edits.DISK_NOTCH == 0.05`, `edits.PLANET_EDIT_FIELDS`, `edits.BLACK_HOLE_EDIT_FIELDS`, `edits.edit_fields(body)`, `edits.collapse(world, index)`, `edits.set_spin(world, index, spin)`, `edits.set_disk(world, index, disk)`, `edits.scrub_spin(spin, notches)`, `edits.scrub_disk(disk, notches)`. `edits.EDIT_FIELDS` is retained as an alias of `PLANET_EDIT_FIELDS` so existing callers keep working.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_black_hole.py`:

```python
# --- collapse and editing ------------------------------------------------------

@pytest.fixture
def world_with_sun():
    from src import materials, units as u
    from src.body import KIND_STAR
    from src.world import World
    import numpy as np
    w = World()
    w.add("Sun", materials.get("hydrogen"), 1.0, u.SUN_RADIUS,
          np.zeros(2), np.zeros(2), kind=KIND_STAR)
    return w


def test_collapse_turns_a_star_into_a_black_hole(world_with_sun):
    from src import edits
    from src.body import KIND_BLACK_HOLE
    edits.collapse(world_with_sun, 0)
    body = world_with_sun.bodies[0]
    assert body.kind == KIND_BLACK_HOLE
    assert body.material.key == "singularity"
    assert body.disk > 0.0


def test_collapse_conserves_mass_exactly(world_with_sun):
    """Collapsing the Sun gives a 1 solar mass hole, which nature could not
    make -- but inventing two extra solar masses to respect the TOV limit is
    worse than allowing it."""
    from src import edits
    edits.collapse(world_with_sun, 0)
    assert float(world_with_sun.mass[0]) == pytest.approx(1.0)


def test_collapse_sets_the_radius_to_the_horizon(world_with_sun):
    from src import edits
    edits.collapse(world_with_sun, 0)
    assert float(world_with_sun.radius[0]) == pytest.approx(
        units.schwarzschild_radius(1.0))
    assert float(world_with_sun.contact_radius[0]) == pytest.approx(
        float(world_with_sun.radius[0]))


def test_collapse_clears_the_trail_and_the_rings(world_with_sun):
    from src import edits
    world_with_sun.bodies[0].rings = True
    world_with_sun.bodies[0].trail.maybe_append((1.0, 2.0), None)
    edits.collapse(world_with_sun, 0)
    assert not world_with_sun.bodies[0].rings
    assert world_with_sun.bodies[0].trail.count == 0


def test_collapse_is_idempotent(world_with_sun):
    from src import edits
    edits.collapse(world_with_sun, 0)
    world_with_sun.bodies[0].spin = 0.1
    edits.collapse(world_with_sun, 0)
    assert world_with_sun.bodies[0].spin == 0.1     # not reset a second time


def test_a_star_scrubbed_past_the_threshold_collapses(world_with_sun):
    from src import edits
    from src.body import KIND_BLACK_HOLE
    edits.set_mass(world_with_sun, 0, units.COLLAPSE_MASS_THRESHOLD)
    assert world_with_sun.bodies[0].kind == KIND_BLACK_HOLE


def test_a_star_below_the_threshold_stays_a_star(world_with_sun):
    from src import edits
    from src.body import KIND_STAR
    edits.set_mass(world_with_sun, 0, units.COLLAPSE_MASS_THRESHOLD * 0.99)
    assert world_with_sun.bodies[0].kind == KIND_STAR


def test_collapse_is_one_way(world_with_sun):
    """Losing mass leaves a smaller black hole, never a star again."""
    from src import edits
    from src.body import KIND_BLACK_HOLE
    edits.collapse(world_with_sun, 0)
    edits.set_mass(world_with_sun, 0, 1e-6)
    assert world_with_sun.bodies[0].kind == KIND_BLACK_HOLE


def test_black_hole_radius_follows_the_horizon_not_the_density(world_with_sun):
    from src import edits
    edits.collapse(world_with_sun, 0)
    edits.set_mass(world_with_sun, 0, 20.0)
    assert float(world_with_sun.radius[0]) == pytest.approx(
        units.schwarzschild_radius(20.0))
    assert float(world_with_sun.contact_radius[0]) == pytest.approx(
        float(world_with_sun.radius[0]))


def test_max_mass_is_the_collapse_threshold():
    from src import edits
    assert edits.MAX_MASS == units.COLLAPSE_MASS_THRESHOLD


def test_spin_scrub_is_clamped_to_the_thorne_limit():
    from src import edits
    assert edits.scrub_spin(0.99, +100) == pytest.approx(units.MAX_SPIN)
    assert edits.scrub_spin(0.01, -100) == 0.0


def test_disk_scrub_is_clamped_to_zero_and_one():
    from src import edits
    assert edits.scrub_disk(0.9, +100) == 1.0
    assert edits.scrub_disk(0.1, -100) == 0.0


def test_edit_fields_depend_on_the_kind(world_with_sun):
    from src import edits
    star = world_with_sun.bodies[0]
    assert edits.edit_fields(star) == ("mass", "density", "material")
    edits.collapse(world_with_sun, 0)
    assert edits.edit_fields(star) == ("mass", "spin", "disk")


def test_cycle_material_refuses_to_touch_a_black_hole(world_with_sun):
    """SINGULARITY is not in MATERIALS, so cycling would raise ValueError."""
    from src import edits
    edits.collapse(world_with_sun, 0)
    edits.cycle_material(world_with_sun, 0, 1)
    assert world_with_sun.bodies[0].material.key == "singularity"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_black_hole.py -k "collapse or scrub_spin or edit_fields" -v`
Expected: FAIL — `AttributeError: module 'src.edits' has no attribute 'collapse'`

- [ ] **Step 3: Implement**

In `src/edits.py`, change the imports and constants:

```python
from . import materials, units
from .body import (DEFAULT_BLACK_HOLE_DISK, DEFAULT_BLACK_HOLE_SPIN,
                   KIND_BLACK_HOLE, KIND_PLANET, KIND_STAR)

#: A body may not be lighter than a small asteroid, nor heavier than the point
#: where a star stops being possible -- past units.COLLAPSE_MASS_THRESHOLD the
#: core collapses instead, so the scrub tops out exactly where nature does.
MIN_MASS = 1e-16
MAX_MASS = units.COLLAPSE_MASS_THRESHOLD

#: Ratio per scrub notch for mass, and g/cm^3 per notch for density.
MASS_NOTCH = 1.06
DENSITY_NOTCH = 0.15
#: Linear per-notch steps for the two black hole controls.  Spin's range is
#: under 1.0 and disk's is exactly 1.0, so neither wants a ratio scrub.
SPIN_NOTCH = 0.02
DISK_NOTCH = 0.05

PLANET_EDIT_FIELDS = ("mass", "density", "material")
BLACK_HOLE_EDIT_FIELDS = ("mass", "spin", "disk")
#: Retained for callers that predate black holes and only ever meant the
#: ordinary case.
EDIT_FIELDS = PLANET_EDIT_FIELDS


def edit_fields(body):
    """The editable fields for this body -- a black hole has no density or
    composition to speak of, so it is offered spin and accretion instead."""
    return BLACK_HOLE_EDIT_FIELDS if body.kind == KIND_BLACK_HOLE \
        else PLANET_EDIT_FIELDS
```

Add the scrubs beside the existing ones:

```python
def scrub_spin(spin, notches):
    """Linear scrub, clamped to the Thorne limit."""
    value = float(spin) + notches * SPIN_NOTCH
    return float(min(max(value, 0.0), units.MAX_SPIN))


def scrub_disk(disk, notches):
    """Linear scrub over the whole feeding range, 0 (starved) to 1."""
    value = float(disk) + notches * DISK_NOTCH
    return float(min(max(value, 0.0), 1.0))
```

Replace `_refresh_kind` and `set_mass`, and add the writers:

```python
def _refresh_kind(world, index):
    """Kind follows mass -- except that collapse is one-way."""
    body = world.bodies[index]
    if body.kind == KIND_BLACK_HOLE:
        # Nothing un-collapses.  A hole that loses mass is a smaller hole.
        return
    mass = float(world.mass[index])
    if mass >= units.COLLAPSE_MASS_THRESHOLD:
        collapse(world, index)
    elif mass >= units.STAR_MASS_THRESHOLD:
        body.kind = KIND_STAR
    elif body.kind == KIND_STAR:
        body.kind = KIND_PLANET


def set_mass(world, index, mass):
    """Set mass so the radius follows.

    For an ordinary body density is held constant and the radius derived from
    it; for a black hole there is no density to hold -- the radius *is* the
    Schwarzschild radius, and it is set by ``_refresh_kind`` reaching
    ``collapse`` or by the branch below.
    """
    body = world.bodies[index]
    mass = float(min(max(float(mass), MIN_MASS), MAX_MASS))

    if body.kind == KIND_BLACK_HOLE:
        world.mass[index] = mass
        world.radius[index] = units.schwarzschild_radius(mass)
        world.contact_radius[index] = world.radius[index]
        return

    density = body_density(world, index)
    world.mass[index] = mass
    world.radius[index] = units.radius_from_mass_density(mass, density)
    _refresh_kind(world, index)


def set_spin(world, index, spin):
    world.bodies[index].spin = float(min(max(float(spin), 0.0), units.MAX_SPIN))


def set_disk(world, index, disk):
    world.bodies[index].disk = float(min(max(float(disk), 0.0), 1.0))


def collapse(world, index):
    """Collapse a body into a black hole.  One-way, and mass is conserved.

    Collapsing the Sun leaves a one-solar-mass hole, which is below the
    Tolman-Oppenheimer-Volkoff limit and so something nature does not make.
    Allowing it is the lesser evil: refusing removes the most interesting
    thing the child can do here, and rounding the mass up to 3 suns invents
    matter out of nothing.
    """
    body = world.bodies[index]
    if body.kind == KIND_BLACK_HOLE:
        return
    body.kind = KIND_BLACK_HOLE
    body.material = materials.SINGULARITY
    body.rings = False
    body.spin = DEFAULT_BLACK_HOLE_SPIN
    body.disk = DEFAULT_BLACK_HOLE_DISK
    # The trail was drawn in the old material's colour and at the old size.
    body.trail.clear()
    world.radius[index] = units.schwarzschild_radius(float(world.mass[index]))
    world.contact_radius[index] = world.radius[index]
```

Guard `cycle_material` (SINGULARITY is not in `MATERIALS`, so `.index()` would raise):

```python
def cycle_material(world, index, step=1):
    body = world.bodies[index]
    if body.kind == KIND_BLACK_HOLE:
        return          # a hole is not made of anything to cycle through
    keys = [m.key for m in materials.MATERIALS]
    current = keys.index(body.material.key)
    set_material(world, index, materials.MATERIALS[(current + step) % len(keys)])
```

Also guard `ignition_ratio`, which the Edit tool's status line calls:

```python
def ignition_ratio(world, index):
    """How many times heavier this body must become before it lights up.

    Zero for a black hole: it is never going to ignite, and the caller shows a
    different line entirely.
    """
    if world.bodies[index].kind == KIND_BLACK_HOLE:
        return 0.0
    mass = float(world.mass[index])
    if mass <= 0.0:
        return float("inf")
    return units.STAR_MASS_THRESHOLD / mass
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_black_hole.py -v`
Expected: PASS (36 tests)

- [ ] **Step 5: Run the existing edit tests**

Run: `python -m pytest tests/test_edits.py tests/test_tools_edit.py -v`
Expected: PASS. `MAX_MASS` rose from 50 to 150; if a test asserts the old ceiling, update it to `units.COLLAPSE_MASS_THRESHOLD` rather than hard-coding 150.

- [ ] **Step 6: Run the full suite**

Run: `python selftest.py`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/edits.py tests/test_black_hole.py tests/test_edits.py
git commit -m "feat: collapse a body into a black hole

MAX_MASS rises from 50 to units.COLLAPSE_MASS_THRESHOLD (150), so the
mass scrub now walks rock -> brown dwarf -> star -> black hole and every
rung is a real number.  You cannot make a 200 solar mass star because
nature cannot either; it collapses instead.

Collapse conserves mass exactly and is one-way: _refresh_kind returns
early for a hole, so losing mass leaves a smaller hole rather than a
star.  cycle_material and ignition_ratio are guarded, since SINGULARITY
is deliberately absent from the MATERIALS tuple."
```

---

### Task 4: A black hole always wins, and always swallows

**Files:**
- Modify: `src/collisions.py`
- Test: `tests/test_black_hole.py` (append)

**Interfaces:**
- Consumes: `units.C`, `units.schwarzschild_radius` (Task 1), `body.KIND_BLACK_HOLE` (Task 2)
- Produces: `collisions.SWALLOW == "swallow"`. `find_collisions` puts a black hole in `Collision.i` regardless of mass and clamps `v_impact` to `units.C`; `classify` returns `ACCRETE` for any pair containing one; `merge` derives a black hole survivor's radius from its horizon; `accrete` sheds no ejecta from one; `_build_event` reports `SWALLOW`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_black_hole.py`:

```python
# --- collisions ----------------------------------------------------------------

def _pair(hole_mass, other_mass, other_radius, separation, speed):
    """A black hole and one other body, closing head-on."""
    import numpy as np
    from src import edits, materials
    from src.body import KIND_STAR
    from src.world import World
    w = World()
    w.add("Hole", materials.get("hydrogen"), hole_mass, units.SUN_RADIUS,
          np.zeros(2), np.zeros(2), kind=KIND_STAR)
    edits.collapse(w, 0)
    w.add("Other", materials.get("rock"), other_mass, other_radius,
          np.array([separation, 0.0]), np.array([-speed, 0.0]))
    w.contact_radius[:w.n] = w.radius[:w.n]
    return w


def test_a_light_black_hole_takes_the_target_slot_against_a_heavy_star():
    """Without this a 3 solar mass hole is absorbed *by* a 10 solar mass star,
    because find_collisions sorts the pair by mass."""
    import numpy as np
    from src import collisions, edits, materials
    from src.body import KIND_BLACK_HOLE, KIND_STAR
    from src.world import World
    w = World()
    w.add("Hole", materials.get("hydrogen"), 3.0, units.SUN_RADIUS,
          np.zeros(2), np.zeros(2), kind=KIND_STAR)
    edits.collapse(w, 0)
    w.add("Star", materials.get("hydrogen"), 10.0, units.SUN_RADIUS,
          np.array([1e-3, 0.0]), np.zeros(2), kind=KIND_STAR)
    w.contact_radius[:w.n] = np.array([1e-2, 1e-2])
    prev = w.pos.copy()
    hits = collisions.find_collisions(w, prev)
    assert hits, "the pair must be detected as touching"
    assert w.bodies[hits[0].i].kind == KIND_BLACK_HOLE


def test_two_black_holes_are_still_ordered_by_mass():
    import numpy as np
    from src import collisions, edits, materials
    from src.body import KIND_STAR
    from src.world import World
    w = World()
    for k, mass in enumerate((3.0, 12.0)):
        w.add("Hole%d" % k, materials.get("hydrogen"), mass, units.SUN_RADIUS,
              np.array([k * 1e-3, 0.0]), np.zeros(2), kind=KIND_STAR)
        edits.collapse(w, k)
    w.contact_radius[:w.n] = np.array([1e-2, 1e-2])
    hits = collisions.find_collisions(w, w.pos.copy())
    assert hits
    assert float(w.mass[hits[0].i]) == pytest.approx(12.0)


def test_impact_speed_never_exceeds_the_speed_of_light():
    """A hole's radius is near zero, so the mutual escape velocity in
    find_collisions runs away and ImpactEvent.energy with it."""
    from src import collisions
    w = _pair(hole_mass=100.0, other_mass=1e-9, other_radius=1e-9,
              separation=1e-6, speed=0.0)
    w.contact_radius[:w.n] = 1e-5
    hits = collisions.find_collisions(w, w.pos.copy())
    assert hits
    assert hits[0].v_impact <= units.C


def test_a_black_hole_always_accretes_never_grazes_or_shatters():
    from src import collisions
    from src.units import EARTH_RADIUS
    # A fast, glancing hit that would be a graze against any ordinary target.
    w = _pair(hole_mass=10.0, other_mass=1e-4, other_radius=EARTH_RADIUS,
              separation=1e-4, speed=5.0)
    w.contact_radius[:w.n] = 1e-3
    hits = collisions.find_collisions(w, w.pos.copy())
    assert hits
    assert collisions.classify(w, hits[0]) == collisions.ACCRETE


def test_swallowing_grows_the_horizon(world_with_sun):
    from src import collisions, edits
    from src.integrator import Integrator
    import numpy as np
    w = world_with_sun
    edits.collapse(w, 0)
    w.add("Rock", __import__("src.materials", fromlist=["x"]).get("rock"),
          1.0, units.EARTH_RADIUS, np.array([1e-6, 0.0]), np.zeros(2))
    w.contact_radius[:w.n] = 1e-4
    resolver = collisions.CollisionResolver(seed=1)
    count, event = resolver.resolve(w, Integrator(w.capacity), w.pos.copy())
    assert count == 1
    assert w.n == 1                                   # nothing came back out
    assert float(w.radius[0]) == pytest.approx(units.schwarzschild_radius(2.0))
    assert float(w.contact_radius[0]) == pytest.approx(float(w.radius[0]))
    assert event.outcome == collisions.SWALLOW


def test_a_black_hole_stays_a_black_hole_after_eating(world_with_sun):
    from src import collisions, edits, materials
    from src.body import KIND_BLACK_HOLE
    from src.integrator import Integrator
    import numpy as np
    w = world_with_sun
    edits.collapse(w, 0)
    w.add("Rock", materials.get("rock"), 1e-6, units.EARTH_RADIUS,
          np.array([1e-6, 0.0]), np.zeros(2))
    w.contact_radius[:w.n] = 1e-4
    collisions.CollisionResolver(1).resolve(w, Integrator(w.capacity), w.pos.copy())
    assert w.bodies[0].kind == KIND_BLACK_HOLE
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_black_hole.py -k "slot or swallow or accretes or light_speed or ordered" -v`
Expected: FAIL — `AttributeError: module 'src.collisions' has no attribute 'SWALLOW'`

- [ ] **Step 3: Implement**

In `src/collisions.py`, extend the import and add the outcome constant beside the others:

```python
from .body import KIND_BLACK_HOLE, KIND_DEBRIS, KIND_PLANET, KIND_STAR
```

```python
ACCRETE = "accrete"
GRAZE = "graze"
DISRUPT = "disrupt"
#: Not a fourth outcome -- the physics of a swallow *is* accretion.  This is
#: the label ImpactEvent carries so the effects layer can draw a collapse
#: inward rather than an explosion outward.
SWALLOW = "swallow"
```

In `find_collisions`, replace the ordering line inside the loop:

```python
    for a, b in zip(rows.tolist(), cols.tolist()):
        ba, bb = bodies[a], bodies[b]
        if (ba.event_id and ba.event_id == bb.event_id
                and now < ba.no_collide_until):
            continue
        # A black hole is always the target, whatever the masses say: a light
        # hole striking a heavy star would otherwise be absorbed *by the star*.
        a_hole = ba.kind == KIND_BLACK_HOLE
        b_hole = bb.kind == KIND_BLACK_HOLE
        if a_hole != b_hole:
            big, small = (a, b) if a_hole else (b, a)
        else:
            big, small = (a, b) if mass[a] >= mass[b] else (b, a)
        dv = vel[small] - vel[big]
        v_rel = float(np.hypot(dv[0], dv[1]))
        sep = float(radii[big] + radii[small])
        v_esc = math.sqrt(2.0 * units.G * float(mass[big] + mass[small]) / sep) \
            if sep > 0.0 else 0.0
        # A horizon is essentially a point, so sep collapses and the mutual
        # escape velocity runs away -- and impact_energy with it, squared.
        v_impact = min(math.hypot(v_rel, v_esc), units.C)
        miss = math.sqrt(max(float(dist2[a, b]), 0.0))
        impact_b = min(miss / sep, 1.0) if sep > 0.0 else 0.0
        out.append(Collision(big, small, float(s[a, b]), v_rel, v_impact, impact_b))
```

In `classify`, short-circuit at the top:

```python
def classify(world, c):
    """Which of the three outcomes this impact is.

    ...existing docstring...
    """
    # Nothing comes back out of a horizon: no graze, no fragments, no ejecta.
    if (world.bodies[c.i].kind == KIND_BLACK_HOLE
            or world.bodies[c.j].kind == KIND_BLACK_HOLE):
        return ACCRETE
    r_i, r_j = float(world.radius[c.i]), float(world.radius[c.j])
    ...
```

In `merge`, replace the radius and kind block:

```python
    survivor = world.bodies[i]
    world.pos[i] = centre
    world.vel[i] = momentum / total
    world.mass[i] = total

    if survivor.kind == KIND_BLACK_HOLE:
        # Radius follows the horizon, not a mass-weighted density.  The mass
        # ceiling does not apply here: eating is how every hole above the
        # stellar range was actually made.
        world.radius[i] = units.schwarzschild_radius(total)
    else:
        world.radius[i] = units.radius_from_mass_density(total, density)
        if total >= units.STAR_MASS_THRESHOLD:
            survivor.kind = KIND_STAR
        elif survivor.kind == KIND_DEBRIS and total >= units.EARTH_MASS * 0.01:
            survivor.kind = KIND_PLANET

    # Reset rather than leave stale: contact_radius must never sit below the
    # true radius it is stretched from, and the render layer only refreshes
    # it once a frame.
    world.contact_radius[i] = world.radius[i]

    world.remove(j)
    return world.index_of(survivor.id)
```

In `accrete`, bail out before the ejecta block:

```python
        idx = merge(world, c)
        if world.bodies[idx].kind == KIND_BLACK_HOLE:
            return idx          # a horizon excavates no crater
        room = world.capacity - world.n
```

In `_build_event`, relabel:

```python
    @staticmethod
    def _build_event(world, c, outcome):
        """Snapshot a collision as an :class:`ImpactEvent`, before anything
        about the pair changes."""
        pos = world.pos[c.j]
        vel = world.vel[c.j] - world.vel[c.i]
        target = world.bodies[c.i]
        if target.kind == KIND_BLACK_HOLE:
            outcome = SWALLOW
        return ImpactEvent(
            ...unchanged...
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_black_hole.py -v`
Expected: PASS (43 tests)

- [ ] **Step 5: Run every collision test**

Run: `python -m pytest tests/test_collision_detect.py tests/test_collision_merge.py tests/test_collision_graze.py tests/test_collision_shatter.py tests/test_contact_radius.py -v`
Expected: PASS — every change is a branch that ordinary bodies never enter.

- [ ] **Step 6: Run the full suite**

Run: `python selftest.py`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/collisions.py tests/test_black_hole.py
git commit -m "feat: a black hole always wins and always swallows

Three branches beside existing logic, not rewrites:

- find_collisions puts a hole in the i slot regardless of mass.  Without
  it a 3 solar mass hole striking a 10 solar mass star is absorbed by
  the star, because the pair is otherwise ordered by mass alone.
- v_impact is clamped to c.  A horizon is essentially a point, so sep
  collapses in the mutual-escape-velocity term and impact_energy runs
  away as its square, which the effects layer then reads.
- classify short-circuits to ACCRETE, merge derives the survivor's
  radius from its horizon, and accrete sheds no ejecta.  Nothing comes
  back out."
```

---

### Task 5: Tides from a black hole, and stellar disruption

**Files:**
- Modify: `src/destruction.py`
- Test: `tests/test_black_hole.py` (append)

**Interfaces:**
- Consumes: `body.KIND_BLACK_HOLE` (Task 2)
- Produces: `destruction._bodies_and_attractors(world)` returning `(victims, stars, attractors)`, replacing `_bodies_and_stars`. `destruction.apply` is unchanged in signature.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_black_hole.py`:

```python
# --- tides ---------------------------------------------------------------------

def _hole_and_victim(hole_mass, distance, victim_kind=None):
    import numpy as np
    from src import edits, materials
    from src.body import KIND_STAR
    from src.world import World
    w = World()
    w.add("Hole", materials.get("hydrogen"), hole_mass, units.SUN_RADIUS,
          np.zeros(2), np.zeros(2), kind=KIND_STAR)
    edits.collapse(w, 0)
    kwargs = {"kind": victim_kind} if victim_kind else {}
    w.add("Victim", materials.get("rock"), units.EARTH_MASS, units.EARTH_RADIUS,
          np.array([distance, 0.0]), np.zeros(2), **kwargs)
    w.contact_radius[:w.n] = w.radius[:w.n]
    return w


def test_a_black_hole_shreds_a_planet_that_comes_close():
    """Earth's tidal radius from a 10 solar mass hole is 0.00636 AU, some
    32,000 horizon radii out -- a big, slow, visible event."""
    from src import collisions, destruction
    w = _hole_and_victim(10.0, distance=0.003)
    count, event = destruction.apply(w, collisions.CollisionResolver(1), 0.01)
    assert count == 1
    assert event.outcome == "tidal"
    assert w.n > 2                          # replaced by a spray of debris


def test_a_planet_safely_outside_the_tidal_radius_survives():
    from src import collisions, destruction
    w = _hole_and_victim(10.0, distance=0.5)
    count, _ = destruction.apply(w, collisions.CollisionResolver(1), 0.01)
    assert count == 0
    assert w.n == 2


def test_a_black_hole_boils_nothing():
    """Hawking radiation from a stellar hole is 6 nanokelvin -- colder than
    the microwave background.  Only the tide destroys anything."""
    from src import collisions, destruction
    w = _hole_and_victim(10.0, distance=0.5)
    before = float(w.mass[1])
    destruction.apply(w, collisions.CollisionResolver(1), 10.0)
    assert float(w.mass[1]) == pytest.approx(before)


def test_a_star_is_shredded_by_a_black_hole():
    """A tidal disruption event -- the most spectacular thing a hole does."""
    import numpy as np
    from src import collisions, destruction, edits, materials
    from src.body import KIND_STAR
    from src.world import World
    w = World()
    w.add("Hole", materials.get("hydrogen"), 50.0, units.SUN_RADIUS,
          np.zeros(2), np.zeros(2), kind=KIND_STAR)
    edits.collapse(w, 0)
    w.add("Star", materials.get("hydrogen"), 1.0, units.SUN_RADIUS,
          np.array([0.005, 0.0]), np.zeros(2), kind=KIND_STAR)
    w.contact_radius[:w.n] = w.radius[:w.n]
    count, event = destruction.apply(w, collisions.CollisionResolver(1), 0.01)
    assert count == 1
    assert event.outcome == "tidal"


def test_a_star_is_not_shredded_by_another_star():
    """Guarded: two ordinary stars would have to be almost touching, and
    letting them shred each other would break every binary scenario."""
    import numpy as np
    from src import collisions, destruction, materials
    from src.body import KIND_STAR
    from src.world import World
    w = World()
    for k in range(2):
        w.add("S%d" % k, materials.get("hydrogen"), 1.0, units.SUN_RADIUS,
              np.array([k * 0.01, 0.0]), np.zeros(2), kind=KIND_STAR)
    w.contact_radius[:w.n] = w.radius[:w.n]
    count, _ = destruction.apply(w, collisions.CollisionResolver(1), 0.01)
    assert count == 0


def test_a_black_hole_is_never_a_victim():
    """Nothing tears a horizon apart -- and the exclusion is in the victim
    list, not merely in the arithmetic.

    A hole's own surface gravity is the largest possible, so the tide on one
    never approaches its strength at any separation (1.7e-15 here).  A purely
    dynamical assertion would therefore pass even with the exclusion deleted,
    which is why this checks the list itself.
    """
    import numpy as np
    from src import collisions, destruction, edits, materials
    from src.body import KIND_STAR
    from src.world import World
    w = World()
    w.add("Big", materials.get("hydrogen"), 100.0, units.SUN_RADIUS,
          np.zeros(2), np.zeros(2), kind=KIND_STAR)
    w.add("Hole", materials.get("hydrogen"), 3.0, units.SUN_RADIUS,
          np.array([0.02, 0.0]), np.zeros(2), kind=KIND_STAR)
    edits.collapse(w, 1)
    w.contact_radius[:w.n] = w.radius[:w.n]

    victims, stars, attractors = destruction._bodies_and_attractors(w)
    assert 1 not in victims, "a black hole must never be a victim"
    assert 1 in attractors, "but it must still raise tides"
    assert 1 not in stars, "and it radiates nothing"

    count, _ = destruction.apply(w, collisions.CollisionResolver(1), 0.01)
    assert count == 0
    assert w.n == 2
```

**The separation matters, and the earlier draft of this plan got it wrong.**
`units.SUN_RADIUS` is 0.00465 AU, so the 1e-4 AU this test originally used put
the hole *inside* the star, where it shreds it at a tidal ratio of 6034 --
triggering this task's own star-as-victim rule and failing on `count == 0`.
0.02 AU puts the ratio on the star at 7.5e-4, comfortably clear.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_black_hole.py -k "shred or boils or victim or safely" -v`
Expected: FAIL — the tidal loop only runs over stars, so `count == 0`.

- [ ] **Step 3: Implement**

In `src/destruction.py`, change the import and replace `_bodies_and_stars`:

```python
from .body import KIND_BLACK_HOLE, KIND_STAR
```

```python
def _bodies_and_attractors(world):
    """(victims, stars, attractors) for this world.

    **Attractors** raise tides: stars and black holes both. **Stars** alone
    radiate -- a stellar-mass hole's Hawking temperature is 6 nanokelvin,
    colder than the microwave background, so it boils nothing.

    A black hole is never a victim: nothing tears a horizon apart. A star may
    be a victim, but only of a hole -- a tidal disruption event is the most
    spectacular thing in this simulation, while two ordinary stars would have
    to be nearly touching, and letting them shred each other would wreck every
    binary arrangement.
    """
    n = world.n
    kinds = [world.bodies[k].kind for k in range(n)]
    stars = [k for k in range(n) if kinds[k] == KIND_STAR]
    attractors = [k for k in range(n)
                  if kinds[k] in (KIND_STAR, KIND_BLACK_HOLE)]
    victims = [i for i in range(n) if kinds[i] != KIND_BLACK_HOLE]
    return victims, stars, attractors
```

Replace the body of `apply` down to the thermal loop:

```python
def apply(world, resolver, dt):
    """Tidal disruption and thermal ablation near a star or a black hole.

    Returns ``(count, event)`` -- the number of bodies destroyed (0 or 1)
    this call, and an :class:`ImpactEvent` describing it, or ``None``.
    """
    victims, stars, attractors = _bodies_and_attractors(world)
    if not victims or not attractors:
        return 0, None

    worst_ratio, worst_i, worst_k = 0.0, None, None
    for i in victims:
        star_victim = world.bodies[i].kind == KIND_STAR
        for k in attractors:
            if k == i:
                continue
            # Only a black hole may shred a star.
            if star_victim and world.bodies[k].kind != KIND_BLACK_HOLE:
                continue
            ratio = tidal_stress_ratio(world, i, k)
            if ratio > worst_ratio:
                worst_ratio, worst_i, worst_k = ratio, i, k
    if worst_ratio >= 1.0:
        event = _event(world, worst_i, "tidal")
        resolver.tidal_disrupt(world, worst_i, worst_k)
        return 1, event

    vaporised = None
    for i in victims:
        if world.bodies[i].kind == KIND_STAR:
            continue                    # a star does not boil in its own light
        hottest = max((equilibrium_temperature(world, i, k) for k in stars),
                     default=0.0)
        ...rest unchanged...
```

Note the `if k == i: continue` guard: `victims` now includes stars, which are also attractors, so a star could otherwise raise a tide on itself and divide by zero.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_black_hole.py -v`
Expected: PASS (49 tests)

- [ ] **Step 5: Run the destruction tests**

Run: `python -m pytest tests/test_destruction.py -v`
Expected: PASS. If a test imports `_bodies_and_stars` by name, update it to `_bodies_and_attractors` and unpack three values.

- [ ] **Step 6: Run the full suite**

Run: `python selftest.py`
Expected: all pass. Pay attention to `tests/test_sim.py` and `tests/test_smoke.py`, which run the real solar system — no black hole exists there, so the new branches must not fire.

- [ ] **Step 7: Commit**

```bash
git add src/destruction.py tests/test_black_hole.py tests/test_destruction.py
git commit -m "feat: black holes raise tides, and shred stars

_bodies_and_stars becomes _bodies_and_attractors, splitting the two
roles that were conflated: attractors raise tides (stars and holes),
stars alone radiate.  A stellar-mass hole is 6 nanokelvin, colder than
the microwave background, so it boils nothing.

Stars join the victim list, guarded so only a hole can shred one --
two ordinary stars would have to be nearly touching, and letting them
disrupt each other would wreck every binary arrangement."
```

---

### Task 6: `scale.visual_radii` — the drawing and aiming fix

**Files:**
- Modify: `src/render/scale.py`
- Test: `tests/test_black_hole_render.py` (create)

**Interfaces:**
- Consumes: `units.apparent_radius` (Task 1), `body.KIND_BLACK_HOLE` (Task 2)
- Produces: `scale.visual_radii(world)` returning a float64 array of length `world.n`; `scale.display_radii_px(radii, zoom, internal_height, real_size=False, floor_radii=None)`; `scale.contact_radii_au(radii, zoom, internal_height, real_size=False, aim_radii=None)`. Both new arguments default to `None`, meaning "use the true radii", so every existing caller is unchanged.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_black_hole_render.py`:

```python
"""Black hole rendering: visual radii, the sprite, and the swallow effect."""

import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    pygame.init()
    pygame.display.set_mode((320, 200))
    yield
    pygame.quit()


@pytest.fixture
def hole_world():
    """A 10 solar mass black hole and the Earth, for comparison."""
    from src import edits, materials, units
    from src.body import KIND_STAR
    from src.world import World
    w = World()
    w.add("Hole", materials.get("hydrogen"), 10.0, units.SUN_RADIUS,
          np.zeros(2), np.zeros(2), kind=KIND_STAR)
    edits.collapse(w, 0)
    w.add("Earth", materials.get("rock"), units.EARTH_MASS, units.EARTH_RADIUS,
          np.array([1.0, 0.0]), np.zeros(2))
    return w


# --- visual radii ---------------------------------------------------------------

def test_visual_radii_replaces_only_the_black_hole(hole_world):
    from src import units
    from src.render import scale
    visual = scale.visual_radii(hole_world)
    assert visual[0] == pytest.approx(units.apparent_radius(10.0))
    assert visual[1] == pytest.approx(units.EARTH_RADIUS)


def test_visual_radii_does_not_mutate_the_world(hole_world):
    from src import units
    from src.render import scale
    scale.visual_radii(hole_world)
    assert float(hole_world.radius[0]) == pytest.approx(
        units.schwarzschild_radius(10.0))


def test_without_the_floor_fix_a_black_hole_draws_as_one_pixel(hole_world):
    """The problem this task exists to solve, pinned so it cannot come back."""
    from src.render import scale
    naive = scale.display_radii_px(hole_world.radii, zoom=64.0,
                                   internal_height=360.0)
    assert naive[0] == pytest.approx(1.0)


def test_with_the_floor_fix_a_black_hole_draws_as_large_as_a_star(hole_world):
    from src.render import scale
    fixed = scale.display_radii_px(hole_world.radii, zoom=64.0,
                                   internal_height=360.0,
                                   floor_radii=scale.visual_radii(hole_world))
    assert fixed[0] == pytest.approx(scale.FLOOR_MAX)


def test_real_size_mode_collapses_a_black_hole_to_a_dot(hole_world):
    """Pressing R must still show the honest truth: the horizon is nothing."""
    from src.render import scale
    real = scale.display_radii_px(hole_world.radii, zoom=64.0,
                                  internal_height=360.0, real_size=True,
                                  floor_radii=scale.visual_radii(hole_world))
    assert real[0] == pytest.approx(1.0)


def test_the_floor_argument_leaves_ordinary_bodies_alone(hole_world):
    from src.render import scale
    plain = scale.display_radii_px(hole_world.radii, 64.0, 360.0)
    fixed = scale.display_radii_px(hole_world.radii, 64.0, 360.0,
                                   floor_radii=scale.visual_radii(hole_world))
    assert fixed[1] == pytest.approx(plain[1])


# --- aiming ---------------------------------------------------------------------

def test_without_the_aim_fix_a_black_hole_is_impossible_to_hit(hole_world):
    """8x the true horizon is 1.6e-6 AU: about 0.0001 pixels at system zoom."""
    from src.render import scale
    naive = scale.contact_radii_au(hole_world.radii, zoom=64.0,
                                   internal_height=360.0)
    assert naive[0] * 64.0 < 0.001          # sub-thousandth of a pixel


def test_with_the_aim_fix_a_black_hole_is_hittable(hole_world):
    from src.render import scale
    fixed = scale.contact_radii_au(hole_world.radii, zoom=64.0,
                                   internal_height=360.0,
                                   aim_radii=scale.visual_radii(hole_world))
    assert fixed[0] * 64.0 >= scale.FLOOR_MAX - 1e-9


def test_contact_radius_never_falls_below_the_true_radius(hole_world):
    from src.render import scale
    for zoom in (1e-3, 1.0, 64.0, 1e9):
        fixed = scale.contact_radii_au(hole_world.radii, zoom, 360.0,
                                       aim_radii=scale.visual_radii(hole_world))
        assert np.all(fixed >= hole_world.radii - 1e-30)


def test_the_aim_argument_leaves_ordinary_bodies_alone(hole_world):
    from src.render import scale
    plain = scale.contact_radii_au(hole_world.radii, 64.0, 360.0)
    fixed = scale.contact_radii_au(hole_world.radii, 64.0, 360.0,
                                   aim_radii=scale.visual_radii(hole_world))
    assert fixed[1] == pytest.approx(plain[1])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_black_hole_render.py -v`
Expected: FAIL — `AttributeError: module 'src.render.scale' has no attribute 'visual_radii'`

Note: `test_without_the_floor_fix_...` and `test_without_the_aim_fix_...` should **pass** even now — they pin the bug, not the fix.

- [ ] **Step 3: Implement**

In `src/render/scale.py`, add the import at the top:

```python
import numpy as np

from .. import units
from ..body import KIND_BLACK_HOLE
```

Add `visual_radii` after the module docstring's constants, before `display_radius_px`:

```python
def visual_radii(world):
    """Per-body radius the *eye* should be served by, in AU.

    Equal to the true radius for everything except a black hole, whose horizon
    is seven orders of magnitude smaller than anything else in the simulation.
    Two things break because of that, and both are fixed here:

    **It would draw as one pixel.** A 10 solar mass hole's horizon is 0.0046
    Earth radii, which puts ``_floor`` at 0.31 px and so clamps up to
    FLOOR_MIN. The Sun draws at 6. The most dangerous object in the sandbox
    would be the least visible thing on screen.

    **It would be impossible to hit.** ``contact_radii_au`` caps the aim radius
    at MAX_AIM_FACTOR times the true radius -- 1.6e-6 AU for the same hole,
    about 0.0001 pixels across at system zoom.

    ``units.apparent_radius`` stands in: the main-sequence mass-radius
    relation, on the grounds that the accretion disk is roughly as big as the
    star that made the hole. **Physics never reads this.** Impact energy, tidal
    stress, disruption thresholds and merged radii all use the true horizon;
    only drawing, picking and the yes/no contact test see this array, which is
    exactly the split ``radius`` and ``contact_radius`` already make.

    Recomputed per frame from at most 200 bodies -- nothing may cache a size
    across frames, and this is negligible beside the O(n^2) force kernel.
    """
    out = np.array(world.radii, dtype=np.float64)
    for i in range(world.n):
        if world.bodies[i].kind == KIND_BLACK_HOLE:
            out[i] = units.apparent_radius(float(world.mass[i]))
    return out
```

Replace `display_radii_px`:

```python
def display_radii_px(radii, zoom, internal_height, real_size=False,
                     floor_radii=None):
    """Vectorised form for drawing every body in one pass.

    ``floor_radii`` is what the *floor* is computed from, defaulting to the
    true radii. Only a black hole passes anything different -- see
    :func:`visual_radii`. The honest ``zoom * radius`` term always uses the
    true radii, so zooming far enough in still shows the real horizon, and
    real-size mode drops the floor entirely.
    """
    radii = np.asarray(radii, dtype=np.float64)
    true_px = zoom * radii
    if real_size:
        return np.maximum(true_px, 1.0)
    basis = radii if floor_radii is None else np.asarray(floor_radii,
                                                         dtype=np.float64)
    return np.maximum(true_px, _floor(basis, internal_height / REF_HEIGHT))
```

Replace `contact_radii_au`:

```python
def contact_radii_au(radii, zoom, internal_height, real_size=False,
                     aim_radii=None):
    """Radii to collide against, in AU: what is drawn, capped so nothing
    stretches far enough to eat a neighbour's orbit.

    ...existing docstring paragraph...

    ``aim_radii`` replaces the true radius **in the upper clip only**, so a
    black hole can be aimed at where its disc appears rather than at a target
    a ten-thousandth of a pixel wide. The lower bound stays the true radius:
    a contact radius may never fall below the real one.
    """
    radii = np.asarray(radii, dtype=np.float64)
    basis = radii if aim_radii is None else np.asarray(aim_radii,
                                                       dtype=np.float64)
    drawn_px = display_radii_px(radii, zoom, internal_height, real_size,
                                floor_radii=basis)
    drawn_au = drawn_px / max(zoom, 1e-30)
    return np.clip(drawn_au, radii, basis * MAX_AIM_FACTOR)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_black_hole_render.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Run the scale and contact tests**

Run: `python -m pytest tests/test_scale.py tests/test_contact_radius.py -v`
Expected: PASS — both new arguments default to the old behaviour.

- [ ] **Step 6: Run the full suite**

Run: `python selftest.py`
Expected: all pass. `test_layering.py` matters here: `render/scale.py` may import from `src.body` and `src.units`, but nothing in `src/*.py` may import it back.

- [ ] **Step 7: Commit**

```bash
git add src/render/scale.py tests/test_black_hole_render.py
git commit -m "feat: scale.visual_radii, fixing both the drawing and the aiming

A 10 solar mass hole's horizon is 0.0046 Earth radii, so _floor clamps
it up to one pixel against the Sun's six -- and contact_radii_au caps
the aim radius at 8x the true horizon, 1.6e-6 AU, which is a target
about 0.0001 pixels across at system zoom.  Both have one cause and take
one fix.

display_radii_px and contact_radii_au each gain an optional argument
defaulting to the true radii, so every existing caller is unchanged.  In
contact_radii_au it replaces the upper clip only: a contact radius may
still never fall below the real one.

Physics never reads any of this.  Both regressions are pinned by tests
that assert the broken behaviour of the unfixed path."
```

---

### Task 7: The black hole sprite

**Files:**
- Modify: `src/render/sprites.py`
- Test: `tests/test_black_hole_render.py` (append)

**Interfaces:**
- Consumes: `units.MAX_SPIN` (Task 1)
- Produces: `sprites.BH_DISK_OUTER`, `BH_DISK_SQUASH`, `BH_LENS_ARC`, `BH_LENS_SQUASH`, `BH_JET_LENGTH`, `BH_DOPPLER`, `BH_DISK_RINGS`, `BH_DETAIL_PX`, `BH_INNER_MAX`, `BH_INNER_MIN`, `sprites._make_black_hole(px, spin, disk, seed)`, `SpriteCache.get_black_hole(seed, px, spin, disk)` returning a `pygame.Surface`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_black_hole_render.py`:

```python
# --- the sprite -----------------------------------------------------------------

@pytest.fixture
def cache():
    from src.render.sprites import SpriteCache
    return SpriteCache(limit=64)


def _centre(surf):
    return surf.get_width() // 2, surf.get_height() // 2


def test_the_centre_is_opaque_black(cache):
    """The one thing a black hole must get right."""
    surf = cache.get_black_hole(seed=1, px=12, spin=0.5, disk=0.8)
    cx, cy = _centre(surf)
    r, g, b, a = surf.get_at((cx, cy))
    assert (r, g, b) == (0, 0, 0)
    assert a == 255


def test_a_bright_photon_ring_sits_at_the_shadow_edge(cache):
    px = 12
    surf = cache.get_black_hole(seed=1, px=px, spin=0.0, disk=0.0)
    cx, cy = _centre(surf)
    # Straight up from the centre: dark inside, bright at px.
    inside = surf.get_at((cx, cy - px + 3))
    rim = max(surf.get_at((cx, cy - px + d)) for d in (-1, 0, 1))
    assert sum(inside[:3]) < 30
    assert sum(rim[:3]) > 300


def test_a_starved_hole_draws_no_disk(cache):
    """disk=0 is a real state: a hole with nothing left to eat."""
    px = 12
    surf = cache.get_black_hole(seed=1, px=px, spin=0.9, disk=0.0)
    cx, cy = _centre(surf)
    # Well outside the photon ring, along the disk plane.
    far = surf.get_at((cx + int(px * 2.2), cy))
    assert far[3] == 0


def test_a_feeding_hole_draws_a_disk_along_the_horizontal(cache):
    px = 12
    surf = cache.get_black_hole(seed=1, px=px, spin=0.0, disk=1.0)
    cx, cy = _centre(surf)
    band = [surf.get_at((cx + int(px * 2.0), cy + dy))[3] for dy in (-1, 0, 1)]
    assert max(band) > 0


def test_the_disk_is_squashed_not_circular(cache):
    """Near-edge-on: much wider than it is tall."""
    px = 12
    surf = cache.get_black_hole(seed=1, px=px, spin=0.0, disk=1.0)
    cx, cy = _centre(surf)
    out = int(px * 2.4)
    along = surf.get_at((cx + out, cy))[3]
    across = surf.get_at((cx, cy + out))[3]
    assert along > across


def test_spin_pulls_the_inner_edge_of_the_disk_inward(cache):
    """The ISCO runs from 3 horizon radii to 0.62 as spin rises.  Exaggerated
    for legibility -- at 6px the true range is under one pixel -- but the
    ordering and direction are honest."""
    from src.render import sprites
    still = sprites._inner_disk_px(20, spin=0.0)
    spun = sprites._inner_disk_px(20, spin=0.998)
    assert spun < still
    assert still == pytest.approx(20 * sprites.BH_INNER_MAX)
    assert spun == pytest.approx(20 * sprites.BH_INNER_MIN, rel=1e-3)


def test_doppler_makes_one_side_of_the_disk_brighter(cache):
    px = 14
    surf = cache.get_black_hole(seed=0, px=px, spin=0.0, disk=1.0)
    cx, cy = _centre(surf)
    out = int(px * 2.0)
    left = sum(surf.get_at((cx - out, cy))[:3])
    right = sum(surf.get_at((cx + out, cy))[:3])
    assert left != right


def test_the_doppler_side_is_stable_for_a_given_seed(cache):
    """It comes from the body's seed, so a hole never flips direction when
    the sprite cache evicts it."""
    a = cache.get_black_hole(seed=7, px=14, spin=0.0, disk=1.0)
    cache.clear()
    b = cache.get_black_hole(seed=7, px=14, spin=0.0, disk=1.0)
    cx, cy = _centre(a)
    out = 28
    assert a.get_at((cx - out, cy)) == b.get_at((cx - out, cy))


def test_a_spinning_feeding_hole_grows_jets(cache):
    px = 14
    spun = cache.get_black_hole(seed=1, px=px, spin=0.9, disk=1.0)
    still = cache.get_black_hole(seed=1, px=px, spin=0.0, disk=1.0)
    assert spun.get_height() > still.get_height()


def test_a_tiny_hole_is_still_a_dark_dot_with_a_rim(cache):
    """At system zoom a hole draws at 6px; below that it must not vanish."""
    surf = cache.get_black_hole(seed=1, px=2, spin=0.5, disk=0.5)
    cx, cy = _centre(surf)
    assert surf.get_at((cx, cy))[:3] == (0, 0, 0)
    assert surf.get_at((cx, cy))[3] == 255


def test_the_cache_distinguishes_spin_and_disk(cache):
    a = cache.get_black_hole(seed=1, px=10, spin=0.0, disk=1.0)
    b = cache.get_black_hole(seed=1, px=10, spin=0.9, disk=1.0)
    c = cache.get_black_hole(seed=1, px=10, spin=0.0, disk=0.2)
    assert a is not b
    assert a is not c
    assert cache.get_black_hole(seed=1, px=10, spin=0.0, disk=1.0) is a
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_black_hole_render.py -k "photon or disk or spin or doppler or jets or dot or distinguishes or opaque" -v`
Expected: FAIL — `AttributeError: 'SpriteCache' object has no attribute 'get_black_hole'`

- [ ] **Step 3: Implement**

In `src/render/sprites.py`, add the import:

```python
from .. import units
```

Add the constants after `GLOW_STRENGTH`:

```python
# --- black holes --------------------------------------------------------------
#
# Everything below is a multiple of the drawn *shadow* radius -- the dark disc
# an observer actually sees, which light bending magnifies to sqrt(27)/2 horizon
# radii.  The horizon itself is never drawn: it is 2.6x smaller and, at any zoom
# a child uses, well under a pixel.

#: How far the accretion disk reaches, in shadow radii.
BH_DISK_OUTER = 2.6
#: Vertical squash of the disk: near-edge-on, which is the view that makes a
#: black hole legible instead of a plain ring.
BH_DISK_SQUASH = 0.22
#: The disk's far side, lensed up over the top of the shadow and down under it.
#: Much rounder than the disk itself, which is what lifts it clear.
BH_LENS_ARC = 1.6
BH_LENS_SQUASH = 0.62
#: Jet length at full spin, in shadow radii.
BH_JET_LENGTH = 4.0
#: Brightness ratio between the approaching and receding halves of the disk --
#: relativistic beaming.  About right for gas orbiting at a few tenths of c.
BH_DOPPLER = 2.5
#: Concentric ellipses the disk is built from.
BH_DISK_RINGS = 7
#: Below this many pixels there is no room for a disk; draw shadow and rim only.
BH_DETAIL_PX = 5

#: Where the disk's inner edge is drawn, in shadow radii, at zero spin and at
#: the Thorne limit.
#:
#: The true ISCO/shadow ratio runs 1.155 down to 0.238 over that range -- but a
#: black hole draws at 6 px, so the whole span is under one pixel, and past
#: a* = 0.87 the true inner edge passes *inside* the shadow and is occluded
#: outright.  Widened to a legible range, exactly the trade ``scale._floor``
#: already makes for radii: the ordering and the direction stay honest, and the
#: dark band between the photon ring and the disk really does close as spin
#: rises.
BH_INNER_MAX = 2.00
BH_INNER_MIN = 1.05

#: Disk colours, inner to outer: white, blue-white, orange, deep red.
_DISK_RAMP = (
    (0xFF, 0xFF, 0xF4),
    (0xCF, 0xE4, 0xFF),
    (0xFF, 0xD2, 0x8A),
    (0xE8, 0x82, 0x28),
    (0x9A, 0x30, 0x10),
)
#: The photon ring: the sharpest edge in the picture.
_PHOTON_RING = (255, 250, 235)
#: Jets are synchrotron light -- blue-white, and dimmer than the disk.
_JET_COLOUR = (170, 205, 255)


def _inner_disk_px(px, spin):
    """Drawn inner edge of the accretion disk, in pixels.  See BH_INNER_MAX."""
    t = min(abs(float(spin)), units.MAX_SPIN) / units.MAX_SPIN
    return px * (BH_INNER_MAX + (BH_INNER_MIN - BH_INNER_MAX) * t)


def _ramp_colour(t):
    """A point along the disk's inner-to-outer colour ramp."""
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    span = len(_DISK_RAMP) - 1
    k = min(int(t * span), span - 1)
    local = t * span - k
    a, b = _DISK_RAMP[k], _DISK_RAMP[k + 1]
    return tuple(int(a[i] + (b[i] - a[i]) * local) for i in range(3))


def _draw_ellipse_ring(surface, colour, alpha, centre, rx, ry, width):
    """One concentric ellipse of the disk, centred on ``centre``."""
    rx, ry = max(1, int(rx)), max(1, int(ry))
    rect = pygame.Rect(centre[0] - rx, centre[1] - ry, rx * 2, ry * 2)
    pygame.draw.ellipse(surface, tuple(colour) + (alpha,), rect,
                        max(1, int(width)))


def _make_black_hole(px, spin, disk, seed):
    """A black hole, painted back to front.

    Blitted **normally, never with BLEND_ADD**.  A star's sprite is additive
    because a star only ever lightens what is behind it; a black hole's whole
    point is an opaque hole in the sky, and BLEND_ADD cannot draw black.  The
    outer falloff is therefore carried in alpha rather than baked into colour,
    which is the opposite of the arrangement the star path warns against.
    """
    px = max(1, int(px))
    spin = min(max(float(spin), 0.0), units.MAX_SPIN)
    disk = min(max(float(disk), 0.0), 1.0)

    detailed = px >= BH_DETAIL_PX
    outer = int(px * BH_DISK_OUTER) + 2
    jet = int(px * BH_JET_LENGTH * spin) if (detailed and disk > 0.0) else 0
    half = max(outer, jet)
    size = half * 2 + 1
    centre = (half, half)
    surface = pygame.Surface((size, size), pygame.SRCALPHA)

    if detailed and disk > 0.0:
        # 1. Jets, first, so everything else covers their roots.
        if jet > 0:
            width = max(1, int(px * 0.28))
            alpha = int(150 * disk)
            for sign in (-1, 1):
                tip = centre[1] + sign * jet
                pygame.draw.polygon(
                    surface, _JET_COLOUR + (alpha,),
                    [(centre[0] - width, centre[1]),
                     (centre[0] + width, centre[1]),
                     (centre[0] + max(1, width // 3), tip),
                     (centre[0] - max(1, width // 3), tip)])

        inner = _inner_disk_px(px, spin)
        reach = px * BH_DISK_OUTER * (0.6 + 0.4 * disk)
        approaching_left = bool(int(seed) & 1)

        # 2. The far side of the disk, bent up over the shadow and down under
        #    it.  Rounder than the disk, which is what lifts it clear.
        arc_rx = px * BH_LENS_ARC
        _draw_ellipse_ring(surface, _ramp_colour(0.35), int(190 * disk),
                           centre, arc_rx, arc_rx * BH_LENS_SQUASH,
                           max(1, px // 5))

        # 3. The disk itself, inner to outer, with 4. Doppler beaming applied
        #    by redrawing the approaching half brighter.
        for k in range(BH_DISK_RINGS):
            frac = k / max(1, BH_DISK_RINGS - 1)
            rx = inner + (reach - inner) * frac
            if rx <= 0:
                continue
            ry = rx * BH_DISK_SQUASH
            base = _ramp_colour(frac)
            # A fast-spinning hole radiates 32% of what it eats against a
            # still one's 5.7%, so the inner rings brighten with spin too.
            gain = 1.0 + 0.6 * (spin / units.MAX_SPIN) * (1.0 - frac)
            alpha = int(min(255, 230 * disk * (1.0 - 0.45 * frac)))
            dim = tuple(min(255, int(c * gain / BH_DOPPLER)) for c in base)
            bright = tuple(min(255, int(c * gain)) for c in base)
            width = max(1, int(px * 0.22))

            _draw_ellipse_ring(surface, dim, alpha, centre, rx, ry, width)
            clip = pygame.Rect(0 if approaching_left else centre[0], 0,
                               centre[0] + 1, size)
            previous = surface.get_clip()
            surface.set_clip(clip)
            _draw_ellipse_ring(surface, bright, alpha, centre, rx, ry, width)
            surface.set_clip(previous)

    # 5. The shadow, opaque, over everything drawn so far -- which is what
    #    makes the disk's far side emerge from behind it.
    pygame.draw.circle(surface, (0, 0, 0, 255), centre, px)

    # 6. The photon ring, right at the shadow's edge.
    pygame.draw.circle(surface, _PHOTON_RING + (255,), centre, px,
                       1 if detailed else 0)
    if not detailed:
        # Too small for a one-pixel outline to survive; draw the rim as a
        # filled disc and punch the shadow back out of it.
        pygame.draw.circle(surface, _PHOTON_RING + (255,), centre, px + 1)
        pygame.draw.circle(surface, (0, 0, 0, 255), centre, px)
    return surface
```

Add the cache method to `SpriteCache`:

```python
    def get_black_hole(self, seed, px, spin, disk):
        """A black hole sprite, cached like any other.

        Spin and disk are bucketed for the same reason ``get`` buckets the
        lighting angle: a continuous value would miss the cache on every frame
        the child holds a stepper down, and regenerating the sprite every frame
        is exactly what the cache exists to avoid.
        """
        px = max(1, int(px))
        spin_bucket = int(round(min(abs(float(spin)), units.MAX_SPIN)
                                / units.MAX_SPIN * 8))
        disk_bucket = int(round(min(max(float(disk), 0.0), 1.0) * 8))
        key = ("bh", int(seed) & 1, px, spin_bucket, disk_bucket)
        hit = self._entries.get(key)
        if hit is not None:
            self._entries.move_to_end(key)
            return hit
        surface = _make_black_hole(px, spin_bucket / 8.0 * units.MAX_SPIN,
                                   disk_bucket / 8.0, seed)
        self._entries[key] = surface
        if len(self._entries) > self.limit:
            self._entries.popitem(last=False)
        return surface
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_black_hole_render.py -v`
Expected: PASS (21 tests)

If `test_a_starved_hole_draws_no_disk` fails, the `jet` branch is running when `disk == 0.0` — check the guard.

- [ ] **Step 5: Run the sprite tests**

Run: `python -m pytest tests/test_sprites.py -v`
Expected: PASS — `get` is untouched and `get_black_hole` uses a disjoint key namespace.

- [ ] **Step 6: Run the full suite**

Run: `python selftest.py`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/render/sprites.py tests/test_black_hole_render.py
git commit -m "feat: the black hole sprite

Shadow, photon ring, near-edge-on accretion disk with the far side
lensed over the top, Doppler beaming, and jets along the spin axis.

Blitted normally rather than with BLEND_ADD, which is the opposite of
the star path and the reason belongs in the docstring: BLEND_ADD only
ever lightens, and a black hole's whole point is an opaque hole in the
sky.  The falloff is carried in alpha instead.

The disk's inner edge moves with spin, exaggerated onto 2.00..1.05
shadow radii -- the true ISCO ratio spans under one pixel at 6px, and
past a* = 0.87 it passes inside the shadow and is occluded outright.
Same trade scale._floor already makes; ordering and direction stay
honest.  Which side is Doppler-brightened comes from the body's seed, so
a hole never flips direction when the cache evicts it."
```

---

### Task 8: Wire the black hole into the render stack

**Files:**
- Modify: `src/render/draw.py`
- Modify: `src/app/state.py` (`pick_body_at` only)
- Modify: `src/app/main.py` (the contact refresh, around line 98)
- Test: `tests/test_black_hole_render.py` (append)

**Interfaces:**
- Consumes: `scale.visual_radii` (Task 6), `SpriteCache.get_black_hole` (Task 7)
- Produces: nothing new — this task makes the previous two visible.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_black_hole_render.py`:

```python
# --- wiring ---------------------------------------------------------------------

def _draw(world, zoom=64.0, size=(320, 200)):
    from src.render.camera import Camera
    from src.render.draw import ViewSettings, WorldRenderer
    from src.render.pixelfont import PixelFont
    from src.render.sprites import SpriteCache
    from src.sim import Simulation

    surface = pygame.Surface(size)
    surface.fill((0, 0, 0))
    camera = Camera(size[0], size[1])
    camera.zoom = zoom
    sim = Simulation(world=world)
    renderer = WorldRenderer(SpriteCache(limit=64), PixelFont())
    renderer.draw(surface, sim, camera, ViewSettings(show_trails=False,
                                                     show_labels=False))
    return surface, camera


def test_a_black_hole_is_drawn_as_a_dark_disc_not_a_bright_one(hole_world):
    """The star path blits additively; a hole must not, or it draws white."""
    hole_world.pos[0] = (0.0, 0.0)
    surface, camera = _draw(hole_world)
    x, y = camera.world_to_screen(hole_world.pos[0])
    assert surface.get_at((int(x), int(y)))[:3] == (0, 0, 0)


def test_a_black_hole_is_drawn_large_enough_to_see(hole_world):
    """Six pixels, not one -- the whole point of visual_radii."""
    hole_world.pos[0] = (0.0, 0.0)
    surface, camera = _draw(hole_world)
    x, y = camera.world_to_screen(hole_world.pos[0])
    lit = [dx for dx in range(-12, 13)
           if surface.get_at((int(x) + dx, int(y)))[:3] != (0, 0, 0)]
    assert lit, "nothing but background was drawn"
    assert max(abs(dx) for dx in lit) >= 4


def test_clicking_where_a_black_hole_appears_selects_it(hole_world):
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.app.state import AppState
    from src.sim import Simulation
    hole_world.pos[0] = (0.0, 0.0)
    camera = Camera(320, 200)
    camera.zoom = 64.0
    state = AppState(Simulation(world=hole_world), camera, ViewSettings())
    x, y = camera.world_to_screen(hole_world.pos[0])
    # Four pixels off centre: inside the drawn disc, astronomically far
    # outside the true horizon.
    assert state.pick_body_at(int(x) + 4, int(y)) == 0


def test_the_contact_refresh_makes_a_black_hole_hittable(hole_world):
    """What App.step_once does once a frame, asserted directly."""
    from src.render import scale
    drawn = scale.contact_radii_au(hole_world.radii, 64.0, 200.0,
                                   aim_radii=scale.visual_radii(hole_world))
    hole_world.contact_radius[:hole_world.n] = \
        hole_world.clamp_contact_radii(drawn)
    assert float(hole_world.contact_radius[0]) > \
        float(hole_world.radius[0]) * 1000.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_black_hole_render.py -k "wiring or drawn or clicking or refresh" -v`
Expected: FAIL — the hole draws at one pixel and the click misses.

- [ ] **Step 3: Implement in `src/render/draw.py`**

Extend the import:

```python
from ..body import KIND_BLACK_HOLE, KIND_STAR
```

In `WorldRenderer.draw`, pass the floor radii:

```python
        radii = scale.display_radii_px(world.radii, camera.zoom, height,
                                       real_size=view.real_size,
                                       floor_radii=scale.visual_radii(world))
```

Replace the sprite branch inside the loop:

```python
            body = world.bodies[i]
            is_star = body.kind == KIND_STAR
            is_hole = body.kind == KIND_BLACK_HOLE

            if px < 2.0:
                if 0 <= int(x) < width and 0 <= int(y) < height:
                    surface.set_at((int(x), int(y)), body.material.tones[2])
            elif px > MAX_SPRITE_HEIGHTS * height:
                # Bigger than the screen: no sprite could add detail, and
                # allocating one this large would fail outright.  A hole this
                # close fills the view with its own shadow, which is correct.
                pygame.draw.circle(surface,
                                   (0, 0, 0) if is_hole
                                   else body.material.tones[1],
                                   (int(x), int(y)),
                                   int(min(px, MAX_DRAW_PX)))
            elif is_hole:
                # Never BLEND_ADD: additive blending cannot draw black, and
                # the shadow is the whole point.
                sprite = self.cache.get_black_hole(body.seed, int(px),
                                                   body.spin, body.disk)
                surface.blit(sprite, sprite.get_rect(center=(int(x), int(y))))
            else:
                if body.rings and px >= 4:
                    rings = ring_surface(int(px), body.material.tones)
                    surface.blit(rings, rings.get_rect(center=(int(x), int(y))))
                sprite = self.cache.get(body.material, body.seed, int(px),
                                        nearest_star_angle(world, i), is_star)
                rect = sprite.get_rect(center=(int(x), int(y)))
                surface.blit(sprite, rect,
                             special_flags=pygame.BLEND_ADD if is_star else 0)
```

`nearest_star_angle` needs no change: it already tests `!= KIND_STAR`, so a black hole never lights anything — correct, since what glows is the disk, not the hole.

- [ ] **Step 4: Implement in `src/app/state.py`**

In `pick_body_at`, pass the same basis so clicking matches drawing:

```python
        radii = scale.display_radii_px(world.radii, self.camera.zoom, height,
                                       real_size=self.view.real_size,
                                       floor_radii=scale.visual_radii(world))
```

- [ ] **Step 5: Implement in `src/app/main.py`**

At the contact refresh (around line 98):

```python
        drawn = scale.contact_radii_au(world.radii, state.camera.zoom,
                                       ...existing arguments...,
                                       aim_radii=scale.visual_radii(world))
        world.contact_radius[:world.n] = world.clamp_contact_radii(drawn)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_black_hole_render.py -v`
Expected: PASS (25 tests)

- [ ] **Step 7: Run the render and app tests**

Run: `python -m pytest tests/test_draw.py tests/test_state.py tests/test_smoke.py tests/test_contact_radius.py -v`
Expected: PASS.

- [ ] **Step 8: Run the full suite**

Run: `python selftest.py`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add src/render/draw.py src/app/state.py src/app/main.py tests/test_black_hole_render.py
git commit -m "feat: draw, click and aim at black holes

One basis -- scale.visual_radii -- read by all three, so they cannot
disagree about where a hole appears to be.  The sprite is blitted
normally rather than additively, since BLEND_ADD cannot draw black."
```

---

### Task 9: The swallow effect

**Files:**
- Modify: `src/render/effects.py`
- Test: `tests/test_black_hole_render.py` (append)

**Interfaces:**
- Consumes: `collisions.SWALLOW` (Task 4)
- Produces: `effects.SWALLOW_LIFE_S`, `effects.SWALLOW_RINGS`. `EffectField.spawn` handles `event.outcome == "swallow"` and returns before the ordinary explosion path.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_black_hole_render.py`:

```python
# --- the swallow effect ---------------------------------------------------------

def _swallow_event(energy=1e-6):
    from src.collisions import SWALLOW, ImpactEvent
    from src import units
    return ImpactEvent(
        pos=(0.0, 0.0), vel=(0.0, 0.0), energy=energy, outcome=SWALLOW,
        target_radius=units.schwarzschild_radius(10.0),
        projectile_radius=units.EARTH_RADIUS,
        tones=((0x40, 0x08, 0x04), (0xE0, 0x70, 0x18), (0xFF, 0xF0, 0xD0)),
        on_star=False,
    )


def test_a_swallow_spawns_particles():
    from src.render.effects import EffectField
    field = EffectField(seed=1)
    field.spawn(_swallow_event())
    assert field.count > 0


def test_a_swallow_collapses_inward_rather_than_exploding_outward():
    """Every other effect grows.  This one is the only one that shrinks."""
    from src.render.effects import EffectField
    field = EffectField(seed=1)
    field.spawn(_swallow_event())
    rings = [p for p in field.particles if p.kind == "ring"]
    assert rings
    assert all(p.r1 < p.r0 for p in rings)


def test_a_swallow_throws_no_debris_outward():
    """Nothing comes back out of a horizon."""
    from src.render.effects import EffectField
    field = EffectField(seed=1)
    field.spawn(_swallow_event(energy=1.0))
    assert not [p for p in field.particles if p.kind == "plume"]


def test_swallow_sparks_travel_inward():
    from src.render.effects import EffectField
    import numpy as np
    field = EffectField(seed=1)
    field.spawn(_swallow_event())
    sparks = [p for p in field.particles if p.kind == "spark"]
    assert sparks
    for p in sparks:
        # Position is offset from the centre; velocity points back at it.
        assert float(np.dot(p.pos, p.vel)) < 0.0


def test_an_ordinary_impact_still_explodes_outward():
    import dataclasses
    from src.collisions import ACCRETE
    from src.render.effects import EffectField
    ordinary = dataclasses.replace(_swallow_event(), outcome=ACCRETE)
    field = EffectField(seed=1)
    field.spawn(ordinary)
    rings = [p for p in field.particles if p.kind == "ring"]
    assert rings
    assert all(p.r1 > p.r0 for p in rings)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_black_hole_render.py -k "swallow" -v`
Expected: FAIL — `spawn` runs the ordinary path, so rings grow and plumes appear.

- [ ] **Step 3: Implement**

In `src/render/effects.py`, add the constants beside the other lifetimes:

```python
#: A swallow is slower than an explosion: matter spirals in, it does not
#: scatter.
SWALLOW_LIFE_S = 1.4
SWALLOW_RINGS = 3
```

At the top of `EffectField.spawn`, after `scale_r` and `magnitude` are computed, insert the branch:

```python
        pos = event.pos
        tones = event.tones
        no_drift = np.zeros(2)

        if event.outcome == "swallow":
            self._spawn_swallow(pos, tones, scale_r, magnitude)
            return

        self._add(_Particle("flash", pos, no_drift, FLASH_LIFE_S,
                            ...unchanged from here...
```

Add the method after `spawn`:

```python
    def _spawn_swallow(self, pos, tones, scale_r, magnitude):
        """Something crossed a horizon.

        The reverse of every other effect in this file: rings collapse inward
        instead of expanding, sparks fall toward the centre instead of away
        from it, and there is no plume at all, because nothing comes back out.
        """
        no_drift = np.zeros(2)
        peak = scale_r * (2.5 + 0.8 * magnitude)

        for k in range(SWALLOW_RINGS):
            delay = 1.0 + 0.35 * k
            self._add(_Particle("ring", pos, no_drift, SWALLOW_LIFE_S * delay,
                                peak * delay, scale_r * 0.05,
                                tones[2], tones[0]))

        n_sparks = int(np.clip(8 + 4.0 * magnitude, 8, 32))
        angles = self.rng.uniform(0.0, 2.0 * math.pi, n_sparks)
        starts = self.rng.uniform(1.2, 3.0, n_sparks)
        for k in range(n_sparks):
            direction = np.array([math.cos(angles[k]), math.sin(angles[k])])
            travel = scale_r * starts[k]
            start = np.asarray(pos, dtype=float) + direction * travel
            # Inward, arriving at the centre exactly as the particle expires.
            spark_vel = -direction * (travel / SWALLOW_LIFE_S)
            self._add(_Particle("spark", start, spark_vel, SWALLOW_LIFE_S,
                                scale_r * 0.10, scale_r * 0.02,
                                (255, 250, 235), tones[1], seed=float(k)))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_black_hole_render.py -v`
Expected: PASS (30 tests)

If `test_swallow_sparks_travel_inward` fails, check that `_Particle` stores `pos` as a copy — the spark's start position must not alias `pos`.

- [ ] **Step 5: Run the effects tests**

Run: `python -m pytest tests/test_effects.py -v`
Expected: PASS — the branch is entered only by the new outcome.

- [ ] **Step 6: Run the full suite**

Run: `python selftest.py`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/render/effects.py tests/test_black_hole_render.py
git commit -m "feat: the swallow effect

The reverse of every other effect here: rings collapse inward, sparks
fall toward the centre, and there is no plume, because nothing comes
back out of a horizon.  Still on wall-clock time, so it looks the same
at any time scale."
```

---

### Task 10: Throwing a black hole

**Files:**
- Modify: `src/render/widgets.py` (the `blackhole` icon)
- Modify: `src/app/tools.py` (`LaunchPreset`, `PRESETS`, `preset_body`, `LaunchTool`)
- Test: `tests/test_black_hole_render.py` (append)

**Interfaces:**
- Consumes: `units.schwarzschild_radius` (Task 1), `body.KIND_BLACK_HOLE` and the two creation defaults (Task 2)
- Produces: `tools.LaunchPreset` gains `kind: str = KIND_PLANET`; `PRESETS` gains a sixth entry `LaunchPreset(6, "Black hole", "singularity", 0.0, mass=10.0, kind=KIND_BLACK_HOLE)`; `preset_body(preset)` returns `(mass, r_s)` for it; the icon name `"blackhole"` is valid in `draw_icon`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_black_hole_render.py`:

```python
# --- throwing -------------------------------------------------------------------

def test_there_is_a_black_hole_preset_on_key_six():
    from src.app.tools import PRESETS
    hole = [p for p in PRESETS if p.label == "Black hole"]
    assert len(hole) == 1
    assert hole[0].key == 6


def test_the_preset_derives_its_radius_from_the_horizon():
    from src import units
    from src.app.tools import PRESETS, preset_body
    hole = next(p for p in PRESETS if p.label == "Black hole")
    mass, radius = preset_body(hole)
    assert radius == pytest.approx(units.schwarzschild_radius(mass))


def test_existing_presets_are_unaffected():
    from src.app.tools import PRESETS, preset_body
    rocky = next(p for p in PRESETS if p.label == "Rocky planet")
    mass, radius = preset_body(rocky)
    assert radius == pytest.approx(6371.0 / 1.495978707e8)
    assert mass > 0.0


def test_launching_the_preset_creates_a_feeding_black_hole():
    import numpy as np
    from src.body import (DEFAULT_BLACK_HOLE_DISK, DEFAULT_BLACK_HOLE_SPIN,
                          KIND_BLACK_HOLE)
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.app.state import AppState
    from src.app.tools import PRESETS, LaunchTool
    from src.sim import Simulation
    from src import scenarios

    sim = Simulation(world=scenarios.two_body())
    state = AppState(sim, Camera(320, 200), ViewSettings())
    tool = LaunchTool()
    tool.preset_index = next(k for k, p in enumerate(PRESETS)
                             if p.label == "Black hole")
    tool._origin = np.array([3.0, 0.0])
    tool._cursor = (170, 100)
    before = sim.world.n
    tool._launch(state)

    assert sim.world.n == before + 1
    body = sim.world.bodies[-1]
    assert body.kind == KIND_BLACK_HOLE
    assert body.spin == pytest.approx(DEFAULT_BLACK_HOLE_SPIN)
    assert body.disk == pytest.approx(DEFAULT_BLACK_HOLE_DISK)
    assert float(sim.world.radius[-1]) == pytest.approx(
        __import__("src.units", fromlist=["x"]).schwarzschild_radius(
            float(sim.world.mass[-1])))


def test_the_blackhole_icon_draws_something():
    from src.render.widgets import draw_icon
    surface = pygame.Surface((16, 16), pygame.SRCALPHA)
    draw_icon(surface, "blackhole", pygame.Rect(0, 0, 16, 16), (255, 255, 255))
    assert any(surface.get_at((x, y))[3] for x in range(16) for y in range(16))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_black_hole_render.py -k "preset or throwing or launching or icon" -v`
Expected: FAIL — `StopIteration` looking for the "Black hole" preset.

- [ ] **Step 3: Implement the icon in `src/render/widgets.py`**

Add a branch in `draw_icon`, beside `"star"`:

```python
    elif name == "blackhole":
        radius = max(2, min(rect.width, rect.height) // 2 - 1)
        pygame.draw.circle(surface, colour, rect.center, radius, 1)
        pygame.draw.circle(surface, colour, rect.center, max(1, radius - 2))
```

- [ ] **Step 4: Implement in `src/app/tools.py`**

Extend the import:

```python
from ..body import (DEFAULT_BLACK_HOLE_DISK, DEFAULT_BLACK_HOLE_SPIN,
                    KIND_BLACK_HOLE, KIND_PLANET, KIND_STAR)
```

Extend the preset record and the tuple:

```python
@dataclass(frozen=True)
class LaunchPreset:
    key: int                 # 1..6, the number key that selects it
    label: str
    material_key: str
    radius_km: float
    #: Solar masses, or None to derive from the material's real density.
    mass: float = None
    #: Overrides the mass-derived kind.  A black hole is not a heavy star, and
    #: no threshold can tell them apart.
    kind: str = KIND_PLANET


PRESETS = (
    LaunchPreset(1, "Asteroid", "rock", 50.0),
    LaunchPreset(2, "Rocky planet", "rock", 6371.0),
    LaunchPreset(3, "Ice planet", "ice", 2500.0),
    LaunchPreset(4, "Gas giant", "gas", 69911.0),
    LaunchPreset(5, "Star", "hydrogen", 0.3 * 695700.0, mass=0.3),
    LaunchPreset(6, "Black hole", "singularity", 0.0, mass=10.0,
                 kind=KIND_BLACK_HOLE),
)
```

Update `preset_body`:

```python
def preset_body(preset):
    """(mass, radius) in simulation units for a launch preset."""
    if preset.kind == KIND_BLACK_HOLE:
        # radius_km means nothing here: a horizon is 2GM/c^2 and nothing else.
        return preset.mass, units.schwarzschild_radius(preset.mass)
    radius = preset.radius_km / units.AU_KM
    if preset.mass is not None:
        return preset.mass, radius
    density = materials.get(preset.material_key).density
    return units.FOUR_THIRDS_PI * density * radius ** 3, radius
```

In `LaunchTool.controls`, give the preset its icon:

```python
    def controls(self, state, metrics):
        rows = []
        for k, preset in enumerate(PRESETS):
            if preset.kind == KIND_BLACK_HOLE:
                icon = "blackhole"
            elif preset.mass is not None:
                icon = "star"
            else:
                icon = "circle"
            rows.append([Control("throw.preset", label=preset.label, payload=k,
                                 icon=icon, hotkey=str(preset.key),
                                 tooltip="Throw %s" % preset.label.lower(),
                                 active=k == self.preset_index)])
```

In `LaunchTool.on_event`, extend the key range:

```python
            # K_1 is left alone so that pressing 1 returns to the Look tool.
            if pygame.K_2 <= event.key <= pygame.K_6:
                self.preset_index = event.key - pygame.K_1
                return True
```

In `LaunchTool._launch`, honour the preset's kind and seed the new fields:

```python
        kind = preset.kind
        if kind == KIND_PLANET and mass >= units.STAR_MASS_THRESHOLD:
            kind = KIND_STAR
        body = world.add(preset.label, materials.get(preset.material_key),
                         mass, radius, np.array(pos, dtype=float), vel,
                         kind=kind,
                         seed=(int(world.time * 1000) + world.n * 7919) % (1 << 30))
        if body is None:
            return
        if kind == KIND_BLACK_HOLE:
            body.spin = DEFAULT_BLACK_HOLE_SPIN
            body.disk = DEFAULT_BLACK_HOLE_DISK
        state.selected_id = body.id
        state.sim.refresh()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_black_hole_render.py -v`
Expected: PASS (35 tests)

- [ ] **Step 6: Run the tool and layout tests**

Run: `python -m pytest tests/test_tools_launch.py tests/test_layout.py tests/test_ui.py tests/test_widgets.py -v`
Expected: PASS. `test_layout.py` builds every panel at the minimum buffer — the Throw palette is now six rows instead of five, so if it overflows, that is a real failure to fix here, not a test to loosen.

- [ ] **Step 7: Run the full suite**

Run: `python selftest.py`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/render/widgets.py src/app/tools.py tests/test_black_hole_render.py
git commit -m "feat: throw a black hole

A sixth Throw preset on key 6.  LaunchPreset gains an explicit kind:
a black hole is not a heavy star, and no mass threshold can tell them
apart, so the preset says which it is rather than leaving it to be
inferred.  radius_km means nothing for one -- a horizon is 2GM/c^2."
```

---

### Task 11: Editing a black hole, and the Collapse button

**Files:**
- Modify: `src/app/tools.py` (`EditTool`)
- Modify: `src/render/hud.py` (`KEY_HELP` only)
- Modify: `src/app/ui.py` (`perform`)
- Test: `tests/test_black_hole_render.py` (append)

**Interfaces:**
- Consumes: `edits.edit_fields`, `edits.collapse`, `edits.scrub_spin`, `edits.scrub_disk`, `edits.set_spin`, `edits.set_disk` (Task 3)
- Produces: control ids `edit.collapse` (hotkey `C`), and `edit.step` payloads carrying the `"spin"` and `"disk"` fields.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_black_hole_render.py`:

```python
# --- editing --------------------------------------------------------------------

def _edit_state(world, select=0):
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.app.state import AppState
    from src.app.tools import EditTool
    from src.sim import Simulation
    sim = Simulation(world=world)
    state = AppState(sim, Camera(320, 200), ViewSettings())
    state.set_tool(EditTool)
    state.selected_id = world.bodies[select].id
    return state


def _ids(rows):
    return [c.id for row in rows for c in row]


def _labels(rows):
    return [c.label for row in rows for c in row]


def test_a_star_is_offered_a_collapse_button(hole_world):
    from src.render.widgets import Metrics
    from src.render.pixelfont import PixelFont
    from src import edits, materials, units
    from src.body import KIND_STAR
    import numpy as np
    hole_world.add("Sun", materials.get("hydrogen"), 1.0, units.SUN_RADIUS,
                   np.array([5.0, 0.0]), np.zeros(2), kind=KIND_STAR)
    state = _edit_state(hole_world, select=2)
    rows = state.tool.controls(state, Metrics(PixelFont(), 320, 200))
    assert "edit.collapse" in _ids(rows)


def test_a_planet_is_not_offered_a_collapse_button(hole_world):
    from src.render.widgets import Metrics
    from src.render.pixelfont import PixelFont
    state = _edit_state(hole_world, select=1)      # Earth
    rows = state.tool.controls(state, Metrics(PixelFont(), 320, 200))
    collapse = [c for row in rows for c in row if c.id == "edit.collapse"]
    assert not collapse or not collapse[0].enabled


def test_a_black_hole_is_offered_spin_and_disk_not_density(hole_world):
    from src.render.widgets import Metrics
    from src.render.pixelfont import PixelFont
    state = _edit_state(hole_world, select=0)
    rows = state.tool.controls(state, Metrics(PixelFont(), 320, 200))
    labels = " ".join(_labels(rows))
    assert "Spin" in labels
    assert "Disk" in labels
    assert "Density" not in labels
    assert "Material" not in labels


def test_the_collapse_button_collapses_the_selected_star(hole_world):
    from src.render.widgets import Control
    from src import materials, units
    from src.body import KIND_BLACK_HOLE, KIND_STAR
    import numpy as np
    hole_world.add("Sun", materials.get("hydrogen"), 1.0, units.SUN_RADIUS,
                   np.array([5.0, 0.0]), np.zeros(2), kind=KIND_STAR)
    state = _edit_state(hole_world, select=2)
    assert state.tool.perform(Control("edit.collapse"), state)
    assert hole_world.bodies[2].kind == KIND_BLACK_HOLE


def test_scrubbing_spin_changes_it(hole_world):
    from src import edits
    state = _edit_state(hole_world, select=0)
    state.tool.field_index = edits.BLACK_HOLE_EDIT_FIELDS.index("spin")
    before = hole_world.bodies[0].spin
    state.tool._scrub(state, +3)
    assert hole_world.bodies[0].spin > before


def test_scrubbing_disk_changes_it(hole_world):
    from src import edits
    state = _edit_state(hole_world, select=0)
    state.tool.field_index = edits.BLACK_HOLE_EDIT_FIELDS.index("disk")
    before = hole_world.bodies[0].disk
    state.tool._scrub(state, -3)
    assert hole_world.bodies[0].disk < before


def test_scrubbing_mass_still_works_on_a_black_hole(hole_world):
    from src import edits, units
    state = _edit_state(hole_world, select=0)
    state.tool.field_index = edits.BLACK_HOLE_EDIT_FIELDS.index("mass")
    state.tool._scrub(state, +5)
    assert float(hole_world.radius[0]) == pytest.approx(
        units.schwarzschild_radius(float(hole_world.mass[0])))


def test_the_c_key_is_documented():
    """tests/test_ui.py reads KEY_HELP and fails if a documented key has no
    button; this asserts the other direction."""
    from src.render.hud import KEY_HELP
    assert any(keys == "C" for keys, _ in KEY_HELP)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_black_hole_render.py -k "collapse or spin or disk or documented" -v`
Expected: FAIL — `EditTool` still uses the fixed three-field list.

- [ ] **Step 3: Implement `EditTool` in `src/app/tools.py`**

Replace the class's field handling. `CLICK_NOTCHES` gains the two new fields:

```python
class EditTool(Tool):
    name = "Edit"
    hint = ("Click a body, then use the -- and + buttons  -  or Up/Down to pick a "
            "row and Left/Right to change it  -  hold a button to go faster")
    context_title = "Edit"

    #: Notches one press of a stepper applies.  Mass moves three at a time
    #: because its scrub is a *ratio*: at 1.06 per notch a single one is
    #: invisible, and igniting Jupiter is eighty of them.
    CLICK_NOTCHES = {"mass": 3, "density": 1, "material": 1,
                     "spin": 1, "disk": 1}

    def __init__(self):
        self.field_index = 0
        self._drag_x = None

    def _fields(self, world, index):
        return edits.edit_fields(world.bodies[index])

    def field_values(self, world, index):
        """The editable values, formatted the way the panel shows them.

        A black hole has no density to hold constant and nothing it is made
        of, so it is offered its spin and how hard it is feeding instead.
        """
        body = world.bodies[index]
        if body.kind == KIND_BLACK_HOLE:
            return {
                "mass": _mass_label(float(world.mass[index])),
                "spin": "%.2f" % body.spin,
                "disk": "off" if body.disk <= 0.0 else "%.0f%%" % (body.disk * 100),
            }
        return {
            "mass": _mass_label(float(world.mass[index])),
            "density": "%.3g g/cm3" % units.to_gcm3(edits.body_density(world, index)),
            "material": world.bodies[index].material.name,
        }

    def status_label(self, world, index):
        body = world.bodies[index]
        if body.kind == KIND_BLACK_HOLE:
            return "nothing escapes"
        if body.kind == KIND_STAR:
            return "burning"
        return "%.0fx heavier to ignite" % edits.ignition_ratio(world, index)

    def controls(self, state, metrics):
        index = state.selected_index
        if index is None:
            return [[Control("edit.none", label="Click a body first", readout=True)]]

        world = state.sim.world
        body = world.bodies[index]
        fields = self._fields(world, index)
        self.field_index = min(self.field_index, len(fields) - 1)
        values = self.field_values(world, index)
        rows = []
        for k, field in enumerate(fields):
            step = self.CLICK_NOTCHES[field]
            # Material is a list of substances, not a quantity, so it gets
            # arrows rather than a minus and a plus, and it does not repeat.
            icons = ("left", "right") if field == "material" else ("minus", "plus")
            rows.append([
                Control("edit.step", icon=icons[0], payload=(field, -step),
                        compact=True, repeat=field != "material",
                        tooltip="Less %s" % field),
                Control("edit.field", label="%s  %s" % (field.title(), values[field]),
                        payload=field, active=k == self.field_index,
                        tooltip="Or pick this row with Up/Down and change it "
                                "with Left/Right"),
                Control("edit.step", icon=icons[1], payload=(field, step),
                        compact=True, repeat=field != "material",
                        tooltip="More %s" % field),
            ])
        if body.kind != KIND_BLACK_HOLE:
            rows.append([Control(
                "edit.collapse", label="Collapse", icon="blackhole", hotkey="C",
                enabled=body.kind == KIND_STAR,
                tooltip="Crush this star into a black hole -- there is no way back")])
        rows.append([Control("edit.status", label=self.status_label(world, index),
                             readout=True)])
        return rows

    def perform(self, control, state):
        if control.id == "edit.field":
            fields = self._fields(state.sim.world, state.selected_index)
            self.field_index = fields.index(control.payload)
            return True
        if control.id == "edit.step":
            field, notches = control.payload
            fields = self._fields(state.sim.world, state.selected_index)
            self.field_index = fields.index(field)
            self._scrub(state, notches)
            return True
        if control.id == "edit.collapse":
            return self._collapse(state)
        return False

    def _collapse(self, state):
        index = state.selected_index
        if index is None:
            return False
        edits.collapse(state.sim.world, index)
        state.sim.refresh()
        return True
```

Extend `on_event` with the hotkey, before the arrow keys:

```python
    def on_event(self, event, state):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                return self._collapse(state)
            if event.key == pygame.K_DOWN:
                ...unchanged...
```

Replace `_scrub`:

```python
    def _scrub(self, state, notches):
        index = state.selected_index
        if index is None:
            return
        world = state.sim.world
        fields = self._fields(world, index)
        field = fields[min(self.field_index, len(fields) - 1)]
        body = world.bodies[index]

        if field == "mass":
            edits.set_mass(world, index,
                           edits.scrub_mass(float(world.mass[index]), notches))
        elif field == "density":
            current = units.to_gcm3(edits.body_density(world, index))
            edits.set_density(
                world, index,
                units.from_gcm3(edits.scrub_density_gcm3(current, notches)))
        elif field == "spin":
            edits.set_spin(world, index, edits.scrub_spin(body.spin, notches))
        elif field == "disk":
            edits.set_disk(world, index, edits.scrub_disk(body.disk, notches))
        else:
            edits.cycle_material(world, index, 1 if notches > 0 else -1)

        state.sim.refresh()
```

Also update the module-level alias, which now means only the ordinary case:

```python
EDIT_FIELDS = edits.PLANET_EDIT_FIELDS
```

- [ ] **Step 4: Document the key in `src/render/hud.py`**

Add to `KEY_HELP`, after the `1 2 3` row:

```python
    ("C", "collapse a star into a black hole (Edit)"),
```

- [ ] **Step 5: Confirm `ui.perform` needs no change**

`UiLayer.perform` already falls through to `state.tool.perform(control, state)` for any unrecognised id, and `edit.collapse` starts with neither `tool.` nor any other reserved prefix. **No change is needed in `src/app/ui.py` for this task.** Verify by running the UI tests in Step 6 rather than editing speculatively.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_black_hole_render.py -v`
Expected: PASS (43 tests)

- [ ] **Step 7: Run the tool, UI and layout tests**

Run: `python -m pytest tests/test_tools_edit.py tests/test_ui.py tests/test_layout.py tests/test_hud.py -v`
Expected: PASS. The `KEY_HELP` parity test now requires a button carrying hotkey `C`, which the Collapse control provides — but only when a body is selected. If the parity test builds the interface with nothing selected, select a star in it.

- [ ] **Step 8: Run the full suite**

Run: `python selftest.py`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add src/app/tools.py src/render/hud.py tests/test_black_hole_render.py
git commit -m "feat: edit a black hole's spin and disk, and collapse a star

The Edit panel's fields now come from edits.edit_fields(body), so a
black hole is offered Mass / Spin / Disk instead of density and
composition -- neither of which means anything for one.

Collapse (C) turns the selected star into a black hole.  Watching the
planets carry on orbiting a dark point is the most valuable thing this
feature makes possible, so it is one click."
```

---

### Task 12: The inspector

**Files:**
- Modify: `src/render/hud.py` (`_rows`)
- Test: `tests/test_black_hole_render.py` (append)

**Interfaces:**
- Consumes: every formula from Task 1
- Produces: nothing new — `Hud._rows` gains a black hole branch.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_black_hole_render.py`:

```python
# --- the inspector ---------------------------------------------------------------

def _inspector_text(world, index):
    from src.render.hud import Hud
    from src.render.pixelfont import PixelFont
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.app.state import AppState
    from src.sim import Simulation
    state = AppState(Simulation(world=world), Camera(320, 200), ViewSettings())
    state.selected_id = world.bodies[index].id
    return "\n".join(text for text, _ in Hud(PixelFont())._rows(state, index))


def test_the_inspector_names_a_black_hole(hole_world):
    assert "black hole" in _inspector_text(hole_world, 0)


def test_the_inspector_reports_escape_velocity_as_the_speed_of_light(hole_world):
    """Falls out of escape_velocity(M, r_s) with no special case."""
    text = _inspector_text(hole_world, 0)
    assert "299792" in text.replace(",", "") or "2.998e+05" in text


def test_the_inspector_shows_the_horizon_shadow_and_isco(hole_world):
    text = _inspector_text(hole_world, 0)
    for label in ("Horizon", "Shadow", "Orbit edge"):
        assert label in text


def test_the_inspector_shows_spin_and_efficiency(hole_world):
    text = _inspector_text(hole_world, 0)
    assert "Spin" in text
    assert "%" in text


def test_the_inspector_never_offers_to_ignite_a_black_hole(hole_world):
    """"0x heavier to ignite" would be nonsense."""
    assert "ignite" not in _inspector_text(hole_world, 0)


def test_an_ordinary_planet_still_offers_to_ignite(hole_world):
    assert "ignite" in _inspector_text(hole_world, 1)


def test_the_inspector_still_describes_an_ordinary_planet(hole_world):
    text = _inspector_text(hole_world, 1)
    assert "Density" in text
    assert "Gravity" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_black_hole_render.py -k "inspector" -v`
Expected: FAIL — "Horizon" is not in the output, and "ignite" is.

- [ ] **Step 3: Implement**

In `src/render/hud.py`, extend the import:

```python
from ..body import KIND_BLACK_HOLE, KIND_STAR
```

Add a formatter beside the others:

```python
def format_temperature(kelvin):
    """Kelvin, without exponent notation for the ordinary range."""
    if kelvin >= 1.0:
        return "%.4g K" % kelvin
    return "%.3g billionths of a K" % (kelvin * 1e9)
```

In `Hud._rows`, branch after `density` is computed:

```python
        if body.kind == KIND_BLACK_HOLE:
            spin = body.spin
            rows = [
                (body.name, palette.SELECT),
                ("black hole", palette.TEXT_DIM),
                ("", palette.TEXT),
                ("Mass       %s" % format_mass(mass), palette.TEXT),
                ("Horizon    %s" % format_length(radius), palette.TEXT),
                ("Shadow     %s" % format_length(units.shadow_radius(mass)),
                 palette.TEXT),
                ("Orbit edge %s" % format_length(units.isco_radius(mass, spin)),
                 palette.TEXT_DIM),
                ("Spin       %.3f" % spin, palette.TEXT),
                ("Feeding    %s" % ("nothing" if body.disk <= 0.0
                                    else "%.0f%%" % (body.disk * 100.0)),
                 palette.TEXT),
                ("Converts   %.1f%% to light"
                 % (units.accretion_efficiency(spin) * 100.0), palette.ACCENT),
                ("Density    %.3g g/cm3" % units.to_gcm3(density), palette.TEXT),
                ("Escape     %s"
                 % format_speed(units.escape_velocity(mass, radius)),
                 palette.WARN),
                ("Its own    %s" % format_temperature(
                    units.hawking_temperature_k(mass)), palette.TEXT_DIM),
                ("heat", palette.TEXT_DIM),
            ]
            if mass < units.TOV_LIMIT_MASS:
                rows.append(("Too light to be real -- under %g Suns"
                             % units.TOV_LIMIT_MASS, palette.WARN))
                rows.append(("nature makes a neutron star.", palette.WARN))
        else:
            rows = [
                (body.name, palette.SELECT),
                ("%s  -  %s" % (body.kind, body.material.name), palette.TEXT_DIM),
                ("", palette.TEXT),
                ("Mass       %s" % format_mass(mass), palette.TEXT),
                ("Radius     %s" % format_length(radius), palette.TEXT),
                ("           %.3g Earths" % units.to_earth_radii(radius),
                 palette.TEXT_DIM),
                ("Density    %.3g g/cm3" % units.to_gcm3(density), palette.TEXT),
                ("Gravity    %.3g g" % units.surface_gravity_g(mass, radius),
                 palette.TEXT),
                ("Escape     %s"
                 % format_speed(units.escape_velocity(mass, radius)),
                 palette.TEXT),
            ]
```

Then guard the ignition block further down:

```python
        if body.kind not in (KIND_STAR, KIND_BLACK_HOLE) and mass > 0.0:
            rows += [
                ("", palette.TEXT),
                ("%.0fx heavier to ignite" % (units.STAR_MASS_THRESHOLD / mass),
                 palette.WARN),
            ]
```

The orbit-facts block and the closing blurb are unchanged and apply to both.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_black_hole_render.py -v`
Expected: PASS (50 tests)

- [ ] **Step 5: Run the HUD tests**

Run: `python -m pytest tests/test_hud.py -v`
Expected: PASS. `visible_rows` trims to fit, so a longer row list is already handled.

- [ ] **Step 6: Run the full suite**

Run: `python selftest.py`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/render/hud.py tests/test_black_hole_render.py
git commit -m "feat: the black hole inspector

Horizon, shadow, ISCO, spin, feeding rate, accretion efficiency and
Hawking temperature replace the density/gravity/ignition block -- which
would otherwise offer to ignite a black hole.

Escape velocity needs no special case: escape_velocity(M, r_s) already
returns exactly c, which is the whole unit system checking itself in
public."
```

---

### Task 13: The black hole scenario

**Files:**
- Modify: `src/scenarios.py`
- Modify: `src/app/state.py` (scenario selection)
- Modify: `src/app/ui.py` (the cycle control and its dispatch)
- Test: `tests/test_black_hole_render.py` (append)

**Interfaces:**
- Consumes: everything above
- Produces: `scenarios.black_hole(world=None)`; `scenarios.REGISTRY["black_hole"]`; `state.scenario` (str, default `"solar_system"`), `state.scenario_label` (str), `state.cycle_scenario()`; control id `scenario.cycle`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_black_hole_render.py`:

```python
# --- the scenario ----------------------------------------------------------------

def test_the_black_hole_scenario_is_registered():
    from src import scenarios
    assert "black_hole" in scenarios.REGISTRY


def test_the_scenario_contains_exactly_one_black_hole():
    from src import scenarios
    from src.body import KIND_BLACK_HOLE
    world = scenarios.black_hole()
    holes = [b for b in world.bodies if b.kind == KIND_BLACK_HOLE]
    assert len(holes) == 1
    assert holes[0].disk > 0.0


def test_the_scenario_shreds_something_almost_immediately():
    """It exists to show a tidal disruption; requiring the child to arrange
    one would defeat the point."""
    from src import scenarios
    from src.sim import Simulation
    sim = Simulation(world=scenarios.black_hole())
    before = sim.world.n
    for _ in range(400):
        sim.advance(0.5, budget_s=1.0)
        if sim.world.n != before:
            break
    assert sim.world.n != before, "nothing was disrupted in 200 days"


def test_the_scenario_is_stable_enough_to_run():
    """No NaNs, nothing flung to infinity."""
    import numpy as np
    from src import scenarios
    from src.sim import Simulation
    sim = Simulation(world=scenarios.black_hole())
    for _ in range(200):
        sim.advance(1.0, budget_s=1.0)
    assert np.all(np.isfinite(sim.world.positions))


def test_cycling_the_scenario_changes_what_restart_loads():
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.app.state import AppState
    from src.body import KIND_BLACK_HOLE
    from src.sim import Simulation
    state = AppState(Simulation(), Camera(320, 200), ViewSettings())
    assert state.scenario == "solar_system"
    while state.scenario != "black_hole":
        state.cycle_scenario()
    state.reset()
    assert any(b.kind == KIND_BLACK_HOLE for b in state.sim.world.bodies)


def test_restart_reloads_the_current_scenario():
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.app.state import AppState
    from src.sim import Simulation
    state = AppState(Simulation(), Camera(320, 200), ViewSettings())
    state.reset()
    assert any(b.name == "Sun" for b in state.sim.world.bodies)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_black_hole_render.py -k "scenario or restart" -v`
Expected: FAIL — `AttributeError: module 'src.scenarios' has no attribute 'black_hole'`

- [ ] **Step 3: Implement in `src/scenarios.py`**

Extend the import and append the scenario before `REGISTRY`:

```python
from . import edits, materials, orbits, units
from .body import KIND_BLACK_HOLE, KIND_STAR
```

```python
def black_hole(world=None, hole_mass=10.0):
    """A stellar-mass black hole with three bodies around it, one doomed.

    The doomed one starts on a steeply eccentric orbit whose periapsis falls
    well inside its tidal radius, so the disruption happens within the first
    minute rather than needing the child to arrange it.  Earth's tidal radius
    from a ten-solar-mass hole is 0.00636 AU -- some 32,000 horizon radii out
    -- so the shredding is large, slow and nowhere near the dot at the centre,
    which is exactly what makes it readable.
    """
    world = world or World()
    world.clear()

    world.add("Black hole", materials.get("hydrogen"), hole_mass,
              units.SUN_RADIUS, np.zeros(2), np.zeros(2), kind=KIND_STAR)
    edits.collapse(world, 0)
    mu = units.G * hole_mass

    # name, a (AU), e, phase (deg), mass, radius (km), material
    entries = (
        ("Doomed", 0.30, 0.985, 175.0, units.EARTH_MASS, 6371.0, "rock"),
        ("Wanderer", 0.60, 0.35, 40.0, units.EARTH_MASS * 0.6, 4800.0, "rock_ice"),
        ("Far one", 1.40, 0.10, 250.0, units.EARTH_MASS * 2.0, 9000.0, "ice"),
    )
    for name, a, e, phase, mass, radius_km, mat in entries:
        nu = orbits.true_anomaly_from_mean(math.radians(phase), e)
        pos, vel = orbits.state_from_elements(mu, a, e, 0.0, nu)
        world.add(name, materials.get(mat), mass, _km(radius_km), pos, vel)

    world.zero_net_momentum()
    return world
```

Register it:

```python
REGISTRY = {
    "solar_system": solar_system,
    "two_body": two_body,
    "black_hole": black_hole,
}
```

**Check the periapsis before moving on.** `a=0.30, e=0.985` gives `r_p = a(1-e) = 0.0045` AU, comfortably inside the 0.008 AU at which `tidal_stress_ratio` reaches 1.0 for an Earth-sized rock against a ten-solar-mass hole. The orbit's period is 19 days and it starts near apoapsis, so the plunge takes about nine. If `test_the_scenario_shreds_something_almost_immediately` fails, raise `e` rather than shrinking `a` — a smaller `a` tightens the timestep for every body.

**Expect the frame budget to bite briefly.** At periapsis the pair's dynamical time is 5.5e-3 days, which drives the integrator to `MAX_K` and makes `sim.advance` fall behind its 12 ms budget for a few seconds. That is the budget mechanism working as designed — the HUD reports `running at N%` and the simulation degrades to slow motion rather than freezing — and it ends the moment the body is disrupted. Do not respond by widening the orbit; that removes the disruption this scenario exists to show.

- [ ] **Step 4: Implement scenario selection in `src/app/state.py`**

Add near `TIME_LABELS`:

```python
#: Starting arrangements the Restart button can load, and what to call them.
SCENARIOS = ("solar_system", "black_hole")
SCENARIO_LABELS = {"solar_system": "Solar system", "black_hole": "Black hole"}
```

In `AppState.__init__`, beside `quit_armed`:

```python
        #: Which arrangement Restart loads.  sim.reset() used to hard-code the
        #: solar system, so there was no way to reach any other.
        self.scenario = SCENARIOS[0]
```

Add the methods beside `reset`:

```python
    @property
    def scenario_label(self):
        return SCENARIO_LABELS[self.scenario]

    def cycle_scenario(self):
        """Step to the next arrangement.  Does not restart on its own -- the
        child picks, then presses Restart, so a stray click costs nothing."""
        here = SCENARIOS.index(self.scenario)
        self.scenario = SCENARIOS[(here + 1) % len(SCENARIOS)]

    def reset(self):
        self.sim.reset(self.scenario)
        self.selected_id = None
        self.camera.follow_id = None
        self.body_list_offset = 0
```

- [ ] **Step 5: Implement the control in `src/app/ui.py`**

In `_build_transport`, add the control before `sim.reset` in the same row:

```python
            Control("scenario.cycle", label=state.scenario_label, icon="list",
                    tooltip="Pick what Restart loads"),
            Control("sim.reset", label="Restart", icon="reset", hotkey="F5",
                    ...unchanged...),
```

In `perform`, add the branch beside `sim.reset`:

```python
        if cid == "scenario.cycle":
            state.cycle_scenario()
            return True
        if cid == "sim.reset":
            state.reset()
            return True
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_black_hole_render.py -v`
Expected: PASS (56 tests)

- [ ] **Step 7: Run the layout, UI and scenario tests**

Run: `python -m pytest tests/test_layout.py tests/test_ui.py tests/test_sim.py tests/test_smoke.py -v`
Expected: PASS. The transport row is one control wider — `test_layout.py` builds every panel at the minimum buffer, so if it now overflows, shorten the label (`"Solar"` / `"Hole"`) rather than loosening the test.

- [ ] **Step 8: Run the full suite**

Run: `python selftest.py`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add src/scenarios.py src/app/state.py src/app/ui.py tests/test_black_hole_render.py
git commit -m "feat: a black hole starting scenario

Three bodies around a ten-solar-mass hole, one on a steeply eccentric
orbit whose periapsis falls well inside its tidal radius, so the
disruption happens in the first minute rather than needing the child to
arrange one.

sim.reset() hard-coded the solar system, so no other arrangement was
reachable.  state.scenario picks; Restart loads it.  Cycling does not
restart on its own, so a stray click costs nothing."
```

---

### Task 14: Documentation

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: everything
- Produces: nothing

- [ ] **Step 1: Update the architecture table**

In the `src/*.py` row, add black holes to the physics description:

```
| `src/*.py` | physics. Pure numpy. **Must never import pygame** ... Includes
`collisions.py` ... and `destruction.py` (tidal and thermal destruction near a star
**or a black hole**), both emitting the plain-data `ImpactEvent`. |
```

- [ ] **Step 2: Add the invariants**

Under **Invariants**, after the `contact_radius` entry:

```markdown
- **A black hole's radius is its Schwarzschild radius**, `2GM/c^2`, not a density
  derivation — `units.schwarzschild_radius`. Collapse is one-way: `edits._refresh_kind`
  returns early for a black hole, so a hole that loses mass is a smaller hole, never a
  star again. Kind still follows mass for everything else, and `units.COLLAPSE_MASS_THRESHOLD`
  (150 solar masses) is both the top of the mass scrub and where nature stops making stars.
- **A black hole is drawn and aimed at through `render/scale.visual_radii`, never
  through its true radius.** The horizon is seven orders of magnitude below anything
  else in the simulation: a 10-solar-mass hole floors at one pixel against the Sun's
  six, and `contact_radii_au` would cap its aim radius at a target 0.0001 pixels
  across. Drawing, picking and the contact refresh all read the one helper so they
  cannot disagree. **Physics never reads it** — every energy, tide, threshold and
  merged radius still uses the true horizon.
- **A black hole always takes the `i` slot in `find_collisions`**, whatever the masses
  say, or a light hole striking a heavy star is absorbed by the star. It always
  accretes: no graze, no fragments, no ejecta.
```

- [ ] **Step 3: Add to the design-decisions section**

Under **Design decisions worth knowing before changing them**:

```markdown
- **A black hole's sprite is blitted normally, never with `BLEND_ADD`**
  (`render/sprites.py`). A star is additive because it only ever lightens what is
  behind it; `BLEND_ADD` cannot draw black, and an opaque hole in the sky is the whole
  point. The falloff is carried in alpha instead — the opposite of the arrangement the
  star path's comment warns against.
- **The accretion disk's inner edge is exaggerated, like drawn radii are.** The true
  ISCO/shadow ratio runs 1.155 to 0.238 across the spin range, which is under one pixel
  at the 6 px a hole draws at, and past `a* = 0.87` the true edge is occluded by the
  shadow entirely. It is mapped onto 2.00..1.05 shadow radii, preserving ordering and
  direction — the same trade `scale._floor` already makes.
- **Spin and disk live on `Body`, not in World's arrays**, because the O(n^2) force
  kernel never reads them — the same reason `rings` and `seed` live there.
```

- [ ] **Step 4: Update the commands block**

The suite has grown; update the count in the `python selftest.py` line to whatever it now reports.

- [ ] **Step 5: Run the full suite one last time**

Run: `python selftest.py`
Expected: all pass. Note the final test count for Step 4.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: black hole invariants and design decisions in CLAUDE.md"
```

---

## Self-Review

**Spec coverage**

| Spec section | Task |
|---|---|
| No-hair theorem, two editable properties + disk | 1, 3, 11 |
| Measured numbers (`r_s`, shadow, photon sphere, Kerr ISCO, efficiency, Hawking) | 1 |
| Escape velocity at the horizon is `c` | 1 (assert), 12 (display) |
| Tidal disruption already works | 5 |
| Density not editable | 3 (fields), 12 (read-only row) |
| TOV limit not enforced; collapse conserves mass | 3, 12 |
| Collapse threshold at 150; the mass ladder | 3 |
| Never grazed, never disrupted, no ejecta | 4 |
| Always wins regardless of mass | 4 |
| Raises tides but emits no heat | 5 |
| A star may be shredded by a hole, not by a star | 5 |
| `v_impact` clamped to `c` | 4 |
| The visibility problem and the aim problem | 6, 8 |
| Real-size mode collapses it to a dot | 6 |
| `clamp_contact_radii` still protects neighbours | 8 |
| The sprite: shadow, ring, disk, lensing, Doppler, jets | 7 |
| Normal blit, not additive | 7, 8 |
| Cache key extends with spin and disk | 7 |
| `blackhole` icon | 10 |
| `swallow` effect | 9 |
| Throw preset on key 6 | 10 |
| Edit fields per kind; Collapse on `C`; `KEY_HELP` | 11 |
| Scenario selection | 13 |
| Inspector rows | 12 |
| Layering unchanged | every task's Step "run the full suite" |
| Testing | 1–13, plus 14 for the count |

No gaps.

**Placeholder scan:** every code step carries real code. The three places that say "unchanged" (`classify`'s docstring, `accrete`'s `room` line, `spawn`'s tail) quote the surrounding lines so the insertion point is unambiguous.

**Type consistency**

- `schwarzschild_radius` / `shadow_radius` / `photon_sphere_radius` / `isco_radius` / `apparent_radius` all take solar masses and return AU. `isco_radius(mass, spin)`, `accretion_efficiency(spin)` — spin second and optional, consistently.
- `visual_radii(world)` -> array; `display_radii_px(..., floor_radii=)` and `contact_radii_au(..., aim_radii=)` — deliberately different names because they do different jobs (a floor versus a cap), both defaulting to `None`.
- `edit_fields(body)` takes a `Body`, not a world/index pair; `EditTool._fields(world, index)` is the adapter, and every call site inside the tool goes through it.
- `get_black_hole(seed, px, spin, disk)` — `seed` first, matching `get(material, seed, px, ...)`.
- `_bodies_and_attractors` returns three values everywhere it is used.
- `SWALLOW` is a `collisions` constant; `effects` compares against the literal `"swallow"`, matching how `"tidal"` and `"vaporise"` already cross that boundary (the render layer must not import from `collisions`).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-27-black-holes.md`.
