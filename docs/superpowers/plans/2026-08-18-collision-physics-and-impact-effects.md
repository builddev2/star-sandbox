# Collision Physics and Impact Effects Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make an impact do something — physically correct outcomes (accretion, graze, disruption, tidal and thermal destruction) and a visible explosion — and make a shot aimed at the drawn Sun actually hit it.

**Architecture:** The physics layer gains a real disruption criterion (Benz & Asphaug Q\*_D), an outcome classifier, energy-bounded fragments, and non-contact destruction near stars; it emits plain `ImpactEvent` dataclasses. A new render module turns those records into flashes, shock rings and ejecta sparks. The two layers never import each other: `Simulation` holds a list of records, the frame loop hands them to `EffectField`.

**Tech Stack:** Python 3.11, numpy, pygame. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-18-collision-physics-and-impact-effects-design.md`

## Global Constraints

- Simulation units are **AU, solar masses, days**. `src/units.py` is the only place a constant or conversion factor may be defined.
- `src/*.py` must never import pygame. `tests/test_layering.py` asserts it in a subprocess.
- Never hold an array index across a frame; store `body.id` and resolve with `World.index_of(id)`.
- Every mutation of mass, radius or membership is followed by `Simulation.refresh()`.
- Mass and density are set; **radius is always derived** (`units.radius_from_mass_density`).
- `World.MAX_BODIES` is 200 and must not be raised.
- Angles are radians everywhere except `src/scenarios.py`.
- Pixel-size constants scale by `internal_height / scale.REF_HEIGHT` (360). No hard-coded pixel counts.
- Tests touching pygame must set `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")` before importing pygame.
- Run the suite with `./.venv/bin/python selftest.py`.

## File Structure

| File | Responsibility |
|---|---|
| `src/units.py` (modify) | add specific-energy conversion and `disruption_threshold` |
| `src/materials.py` (modify) | add `toughness` and `vapour_temperature_k` per material |
| `src/collisions.py` (modify) | contact geometry, outcome classification, merge / graze / disrupt, `ImpactEvent` |
| `src/destruction.py` (create) | tidal break-up and thermal ablation near stars |
| `src/world.py` (modify) | `contact_radius` array kept parallel to `radius` |
| `src/sim.py` (modify) | drive destruction, collect `impacts` |
| `src/render/effects.py` (create) | flash, shock ring, ejecta sparks, plume, stellar flare |
| `src/app/main.py` (modify) | refresh contact radii, update and draw effects |
| `src/app/state.py` (modify) | hold the `EffectField` |
| `tests/test_disruption.py` (create) | Q\*_D curve and outcome classification |
| `tests/test_collision_graze.py` (create) | hit-and-run |
| `tests/test_destruction.py` (create) | tidal and thermal |
| `tests/test_contact_radius.py` (create) | aim floor and the reported-bug regression |
| `tests/test_effects.py` (create) | particle lifecycle, cap, wall-clock timing |

---

### Task 1: Disruption threshold

**Files:**
- Modify: `src/units.py`, `src/materials.py`
- Test: `tests/test_disruption.py`

**Interfaces:**
- Produces: `units.SPECIFIC_ENERGY_SI` (float, J/kg per sim specific energy), `units.disruption_threshold(radius, density, toughness=1.0) -> float` in **sim units** (AU²/day²); `Material.toughness`, `Material.vapour_temperature_k`.

- [ ] **Step 1: Write the failing test**

```python
import math
from src import units, materials


def _si(q):
    """Sim specific energy -> J/kg."""
    return q * units.SPECIFIC_ENERGY_SI


def test_strength_branch_matches_published_basalt_fit():
    # Benz & Asphaug 1999 basalt, converted to SI: 608 * R**-0.38 J/kg.
    rho = units.from_gcm3(3.0)
    for r_m in (1.0, 10.0, 100.0):
        r = r_m / units.AU_M
        got = _si(units.disruption_threshold(r, rho))
        assert math.isclose(got, 608.0 * r_m ** -0.38, rel_tol=0.02)


def test_gravity_branch_dominates_for_a_hundred_kilometre_body():
    rho = units.from_gcm3(3.0)
    r = 1.0e5 / units.AU_M
    got = _si(units.disruption_threshold(r, rho))
    # ~3e5 J/kg is the measured value for a 100 km basalt target.
    assert 1.0e5 < got < 1.0e6


def test_crossover_between_branches_is_near_two_hundred_metres():
    rho = units.from_gcm3(3.0)
    lo = units.disruption_threshold(50.0 / units.AU_M, rho)
    hi = units.disruption_threshold(50.0e3 / units.AU_M, rho)
    mid = units.disruption_threshold(230.0 / units.AU_M, rho)
    assert mid < lo and mid < hi          # the curve has its minimum in between


def test_toughness_scales_only_the_strength_branch():
    rho = units.from_gcm3(3.0)
    small = 10.0 / units.AU_M
    big = 1.0e5 / units.AU_M
    assert units.disruption_threshold(small, rho, 3.0) > \
           2.5 * units.disruption_threshold(small, rho, 1.0)
    assert math.isclose(units.disruption_threshold(big, rho, 3.0),
                        units.disruption_threshold(big, rho, 1.0), rel_tol=0.01)


def test_every_material_declares_toughness_and_vapour_point():
    for m in materials.MATERIALS:
        assert m.toughness > 0.0
        assert m.vapour_temperature_k > 0.0
```

- [ ] **Step 2: Run it and watch it fail**

Run: `./.venv/bin/python -m pytest tests/test_disruption.py -v`
Expected: FAIL, `AttributeError: module 'src.units' has no attribute 'SPECIFIC_ENERGY_SI'`.

- [ ] **Step 3: Implement**

In `src/units.py`, after the derived-conversion block:

```python
#: 1 (AU/day)^2 expressed in J/kg (== m^2/s^2).  Specific energies -- the
#: energy per unit mass an impact delivers -- are quoted in J/kg in every
#: paper, and in AU^2/day^2 in here.
SPECIFIC_ENERGY_SI = (AU_M / SEC_PER_DAY) ** 2

#: Benz & Asphaug (1999) basalt fit for the catastrophic-disruption specific
#: energy, converted from erg/g and centimetres to J/kg and metres.  The first
#: term is material strength and falls with size; the second is self-gravity and
#: rises with it.  They cross near 200 m, which is why a boulder is harder to
#: break than a mountain.
QD_STRENGTH_SI = 608.0
QD_STRENGTH_EXP = -0.38
QD_GRAVITY_SI = 1.574e-2
QD_GRAVITY_EXP = 1.36


def disruption_threshold(radius, density, toughness=1.0):
    """Specific energy needed to disperse half of a body, in simulation units.

    ``radius`` and ``density`` are in simulation units; ``toughness`` scales the
    strength branch only, because how hard a material is to crack is chemistry
    while its self-gravity is not.
    """
    r_m = max(radius * AU_M, 1.0e-3)
    strength = toughness * QD_STRENGTH_SI * r_m ** QD_STRENGTH_EXP
    gravity = QD_GRAVITY_SI * to_gcm3(density) * r_m ** QD_GRAVITY_EXP
    return (strength + gravity) / SPECIFIC_ENERGY_SI
```

In `src/materials.py`, add two fields to `Material` and fill them for all seven:

```python
    #: Scales the strength branch of the disruption threshold.  Iron is tough,
    #: ice is brittle, and a gas giant has no meaningful strength at all.
    toughness: float = 1.0
    #: Temperature at which the surface boils away near a star, in kelvin.
    vapour_temperature_k: float = 2000.0
```

Values: iron `3.0 / 3130`, rock `1.0 / 2700`, rock_ice `0.6 / 1200`, water `0.1 / 373`, ice `0.4 / 180`, gas `0.05 / 150`, hydrogen `0.05 / 100`.

- [ ] **Step 4: Run the test — expect PASS.** Then `./.venv/bin/python selftest.py` for regressions.

- [ ] **Step 5: Commit** — `feat: real disruption threshold, not just binding energy`

---

### Task 2: Impact geometry — speed at contact and impact angle

**Files:**
- Modify: `src/collisions.py`
- Test: `tests/test_collision_detect.py`

**Interfaces:**
- Produces: `Collision` gains `v_impact` (speed at contact, AU/day, gravitationally focused) and `b` (normalised impact parameter, 0 head-on .. 1 grazing). `impact_energy` and `specific_impact_energy(world, c)` use `v_impact`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_collision_detect.py`)

```python
def test_impact_speed_includes_gravitational_focusing():
    from src import units
    from src.collisions import find_collisions
    w = World(8)
    m = units.EARTH_MASS
    r = units.EARTH_RADIUS
    w.add("A", materials.get("rock"), m, r, [0.0, 0.0], [0.0, 0.0])
    w.add("B", materials.get("rock"), m, r, [1.5 * r, 0.0], [0.0, 0.0])
    prev = w.pos.copy()
    w.pos[1] = [1.9 * r, 0.0]          # drifted, but effectively at rest
    hits = find_collisions(w, prev)
    assert hits
    v_esc = units.escape_velocity(2 * m, 2 * r)
    # Released from rest at contact separation, they still arrive at v_esc.
    assert hits[0].v_impact >= 0.9 * v_esc


def test_head_on_and_grazing_impact_parameters():
    from src.collisions import find_collisions
    w = World(8)
    r = 1.0e-4
    w.add("A", materials.get("rock"), 1e-8, r, [0.0, 0.0], [0.0, 0.0])
    w.add("B", materials.get("rock"), 1e-9, r, [0.01, 0.0], [-1.0, 0.0])
    prev = w.pos.copy()
    w.pos[1] = [-0.01, 0.0]
    assert find_collisions(w, prev)[0].b < 0.1

    w2 = World(8)
    w2.add("A", materials.get("rock"), 1e-8, r, [0.0, 0.0], [0.0, 0.0])
    w2.add("B", materials.get("rock"), 1e-9, r, [0.01, 1.9 * r], [-1.0, 0.0])
    prev2 = w2.pos.copy()
    w2.pos[1] = [-0.01, 1.9 * r]
    assert find_collisions(w2, prev2)[0].b > 0.9
```

- [ ] **Step 2: Run — expect FAIL** (`Collision` has no attribute `v_impact`).

- [ ] **Step 3: Implement.** Add fields to the `Collision` dataclass and compute them in `find_collisions`:

```python
    #: Speed at contact, AU/day: the relative speed with the pair's mutual
    #: escape velocity folded in, which is what a body actually arrives at.
    v_impact: float = 0.0
    #: Normalised impact parameter at contact, 0 head-on .. 1 grazing.
    b: float = 0.0
```

In the pair loop, with `sep = radii[big] + radii[small]` (true radii, not contact radii):

```python
        v_esc = math.sqrt(2.0 * units.G * (mass[big] + mass[small]) / sep)
        v_rel = float(np.hypot(dv[0], dv[1]))
        v_impact = math.sqrt(v_rel * v_rel + v_esc * v_esc)
        miss = math.sqrt(max(dist2[a, b], 0.0))
        b = float(min(miss / sep, 1.0)) if sep > 0.0 else 0.0
```

Replace `impact_energy`'s use of `c.v_rel` with `c.v_impact`, and add:

```python
def specific_impact_energy(world, c):
    """Impact kinetic energy per unit total mass -- the Q_R of the literature."""
    total = float(world.mass[c.i] + world.mass[c.j])
    return impact_energy(world, c) / total if total > 0.0 else 0.0
```

- [ ] **Step 4: Run the file, then the suite.**
- [ ] **Step 5: Commit** — `fix: measure impact speed at contact, not at the end of the step`

---

### Task 3: Outcome classification

**Files:**
- Modify: `src/collisions.py`
- Test: `tests/test_disruption.py`

**Interfaces:**
- Produces: constants `ACCRETE`, `GRAZE`, `DISRUPT`; `threshold_of_target(world, c) -> float`; `classify(world, c) -> str`. `is_catastrophic` is removed and every caller updated.

- [ ] **Step 1: Write the failing test** (append to `tests/test_disruption.py`)

```python
import numpy as np
from src import collisions, materials, units
from src.world import World


def _pair(m1, r1, m2, r2, sep_factor, v, offset=0.0):
    w = World(8)
    rock = materials.get("rock")
    w.add("A", rock, m1, r1, [0.0, 0.0], [0.0, 0.0])
    w.add("B", rock, m2, r2, [sep_factor * (r1 + r2), offset], [-v, 0.0])
    prev = w.pos.copy()
    w.pos[1] = [-sep_factor * (r1 + r2), offset]
    hits = collisions.find_collisions(w, prev)
    return w, hits[0]


def test_a_gentle_touch_accretes():
    m = units.EARTH_MASS
    w, c = _pair(m, units.EARTH_RADIUS, m * 1e-6, units.EARTH_RADIUS * 0.01, 2.0, 1e-6)
    assert collisions.classify(w, c) == collisions.ACCRETE


def test_a_fast_head_on_hit_disrupts():
    m = units.EARTH_MASS * 1e-6
    r = units.radius_from_mass_density(m, units.from_gcm3(3.0))
    w, c = _pair(m, r, m, r, 2.0, 0.5)
    assert collisions.classify(w, c) == collisions.DISRUPT


def test_equal_bodies_that_only_clip_each_other_hit_and_run():
    m = units.EARTH_MASS
    r = units.EARTH_RADIUS
    w, c = _pair(m, r, m, r, 4.0, 0.02, offset=1.9 * r)
    assert c.b > 0.9
    assert collisions.classify(w, c) == collisions.GRAZE


def test_an_asteroid_can_never_hit_and_run_off_the_sun():
    w = World(8)
    w.add("Sun", materials.get("hydrogen"), 1.0, units.SUN_RADIUS, [0.0, 0.0], [0.0, 0.0])
    r = 50.0 / units.AU_KM
    m = units.FOUR_THIRDS_PI * materials.get("rock").density * r ** 3
    w.add("Rock", materials.get("rock"), m, r,
          [units.SUN_RADIUS * 0.999, 0.0], [-0.05, 0.0])
    prev = w.pos.copy()
    w.pos[1] = [-units.SUN_RADIUS * 0.999, 0.0]
    c = collisions.find_collisions(w, prev)[0]
    assert collisions.classify(w, c) == collisions.ACCRETE
```

- [ ] **Step 2: Run — expect FAIL** (`module 'src.collisions' has no attribute 'classify'`).

- [ ] **Step 3: Implement**

```python
ACCRETE = "accrete"
GRAZE = "graze"
DISRUPT = "disrupt"

#: Below this mass ratio the projectile cannot survive a graze -- it is a
#: bullet, not a second planet, and it simply buries itself.
GRAZE_MIN_MASS_RATIO = 0.02


def threshold_of_target(world, c):
    """Specific energy needed to disperse the pair, in simulation units."""
    material = world.bodies[c.i].material
    density = units.density_from_mass_radius(float(world.mass[c.i]),
                                             float(world.radius[c.i]))
    return units.disruption_threshold(float(world.radius[c.i]), density,
                                      material.toughness)


def classify(world, c):
    """Which of the three outcomes this impact is."""
    r_i, r_j = float(world.radius[c.i]), float(world.radius[c.j])
    b_crit = r_i / (r_i + r_j)
    ratio = float(world.mass[c.j] / world.mass[c.i])
    if c.b > b_crit and ratio >= GRAZE_MIN_MASS_RATIO:
        return GRAZE
    if specific_impact_energy(world, c) >= threshold_of_target(world, c):
        return DISRUPT
    return ACCRETE
```

- [ ] **Step 4: Run the file and the suite.** `tests/test_collision_shatter.py` will need its thresholds updated — a shattering now needs a genuinely energetic impact, so raise the test velocities until they disrupt, and keep one case asserting a slow touch merges.
- [ ] **Step 5: Commit** — `feat: three collision outcomes chosen by energy and impact angle`

---

### Task 4: Energy-bounded fragments and the Leinhardt & Stewart remnant

**Files:**
- Modify: `src/collisions.py`
- Test: `tests/test_collision_shatter.py`

**Interfaces:**
- Produces: `largest_remnant_fraction(q_ratio) -> float`; `CollisionResolver.shatter` unchanged in signature, changed in behaviour.

- [ ] **Step 1: Write the failing test**

```python
def test_remnant_fraction_follows_the_universal_law():
    from src.collisions import largest_remnant_fraction
    assert abs(largest_remnant_fraction(1.0) - 0.5) < 1e-9
    assert largest_remnant_fraction(0.5) > 0.5      # barely damaged
    assert largest_remnant_fraction(1.8) < 0.15     # nearly all dust
    assert largest_remnant_fraction(20.0) < largest_remnant_fraction(5.0)


def test_fragment_kinetic_energy_never_exceeds_the_impact_energy():
    # Build an energetic disruption, shatter it, and add up the pieces.
    ...  # (uses the helper below; asserts ke_after_com <= impact_energy_before)


def test_a_marginal_disruption_barely_separates_and_an_overkill_sprays():
    # Same pair, two speeds; the faster one's fragment speeds are strictly larger.
    ...
```

Write these three out fully against the existing helpers in `tests/test_collision_shatter.py`, computing centre-of-mass kinetic energy as
`0.5 * sum(m_k * |v_k - v_com|^2)`.

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
#: Leinhardt & Stewart (2012): above this multiple of the threshold the
#: largest remnant collapses into a supercatastrophic power law.
SUPERCATASTROPHIC_RATIO = 1.8
#: Fraction of the excess impact energy that ends up as fragment motion.  The
#: rest goes to heat and to grinding rock into gravel, which is where most of
#: it goes in reality.
DISPERSAL_EFFICIENCY = 0.2


def largest_remnant_fraction(q_ratio):
    """Mass of the biggest surviving piece as a fraction of the pair's total."""
    if q_ratio <= SUPERCATASTROPHIC_RATIO:
        return float(np.clip(0.5 * (2.0 - q_ratio), 0.05, 1.0))
    return float(np.clip(0.1 * (q_ratio / SUPERCATASTROPHIC_RATIO) ** -1.5,
                         0.01, 0.5))
```

In `shatter`, replace the invented `remnant_share` with `largest_remnant_fraction(q_ratio)`, keep the random partition for the rest, and replace the fixed `kick` with a speed distribution rescaled to the energy budget:

```python
        q_r = specific_impact_energy(world, c)
        q_star = threshold_of_target(world, c)
        budget = DISPERSAL_EFFICIENCY * max(q_r - q_star, 0.0) * total
        v_esc = units.escape_velocity(total, r_big)
        # Shape first, then scale to the budget; never below escape speed or
        # the pieces simply fall back together and the break-up is invisible.
        speeds = self.rng.uniform(0.55, 1.45, count)
        speeds[0] *= 0.35
        relative = velocities_from(speeds)         # about the drift frame
        ke = 0.5 * float((masses * (speeds ** 2)).sum())
        if ke > 0.0 and budget > 0.0:
            speeds *= math.sqrt(budget / ke)
        floor = 1.05 * v_esc
        if speeds.max() < floor:
            speeds *= floor / max(speeds.max(), 1e-30)
```

then build `velocities` from the rescaled `speeds` exactly as before, and keep the existing momentum correction.

- [ ] **Step 4: Run the file and the suite.**
- [ ] **Step 5: Commit** — `fix: fragments carry only the energy the impact delivered`

---

### Task 5: Hit-and-run

**Files:**
- Modify: `src/collisions.py`
- Test: `tests/test_collision_graze.py` (create)

**Interfaces:**
- Produces: `CollisionResolver.graze(world, c) -> tuple[int, int]` returning both survivors' indices.

- [ ] **Step 1: Write the failing test**

```python
def test_a_graze_leaves_both_bodies_alive():
def test_a_graze_conserves_momentum_and_mass():
def test_a_graze_removes_energy_rather_than_adding_it():
def test_grazed_bodies_do_not_immediately_re_collide():
```

Each written out in full: build the grazing pair from Task 3's `_pair` helper, call `resolver.graze`, assert `world.n == 2`, that summed `m*v` matches to 1e-15 relative, that the centre-of-mass kinetic energy after is strictly less than before, and that a second `find_collisions` on the same frame yields no unresolved pair for them (shared `event_id`, live `no_collide_until`).

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
#: A graze is inelastic: this much of the normal-component relative speed
#: survives it.  Rock does not bounce.
GRAZE_RESTITUTION = 0.3
#: Fraction of the projectile scraped off and accreted by the target.
GRAZE_MASS_TRANSFER = 0.1
```

`graze` reflects the normal component of the relative velocity with that restitution, keeps the tangential component, moves `GRAZE_MASS_TRANSFER` of the projectile's mass to the target (rederiving both radii from their unchanged densities), corrects both velocities so total momentum is exactly preserved, nudges the pair apart to `1.01 * (r_i + r_j)` along the normal so they are not interpenetrating, and stamps both with a shared `event_id` and a `no_collide_until` long enough to separate.

- [ ] **Step 4: Run the file and the suite.**
- [ ] **Step 5: Commit** — `feat: a glancing blow deflects instead of merging`

---

### Task 6: Cratering ejecta on accretion

**Files:**
- Modify: `src/collisions.py`
- Test: `tests/test_collision_merge.py`

**Interfaces:**
- Produces: `CollisionResolver.accrete(world, c) -> int`; `ejecta_mass_fraction(q_ratio) -> float`.

- [ ] **Step 1: Write the failing test**

```python
def test_a_feeble_impact_produces_no_ejecta_bodies():
def test_an_energetic_but_sub_catastrophic_impact_throws_real_debris():
def test_accretion_conserves_mass_and_momentum_with_ejecta():
def test_ejecta_is_suppressed_when_the_world_is_nearly_full():
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement.** `accrete` merges as now, then, when `q_ratio` exceeds `EJECTA_MIN_RATIO = 0.15` and the world has at least `MIN_EJECTA = 3` free rows, removes `ejecta_mass_fraction(q_ratio) = clip(0.3 * q_ratio, 0, 0.2)` of the total from the survivor and spawns that mass as escaping debris at slightly above escape velocity, in a cone opposite the impact direction. Below that, the ejecta is visual only and the merge is exact.

- [ ] **Step 4: Run the file and the suite.**
- [ ] **Step 5: Commit** — `feat: a big impact excavates a crater's worth of debris`

---

### Task 7: Contact radii — collide with what is drawn

**Files:**
- Modify: `src/world.py`, `src/collisions.py`, `src/app/main.py`, `src/render/scale.py`
- Test: `tests/test_contact_radius.py` (create)

**Interfaces:**
- Produces: `World.contact_radius` (numpy array, kept equal to `radius` unless overwritten), `World.contact_radii` property; `scale.contact_radii_au(radii, zoom, internal_height, real_size=False)` returning the clipped AU radii; `scale.MAX_AIM_FACTOR = 8.0`.

- [ ] **Step 1: Write the failing test**

```python
def test_contact_radius_defaults_to_the_true_radius():
def test_contact_radius_never_reaches_a_neighbouring_orbit():
    # Every body of scenarios.solar_system(), every zoom from 1 to 1e7 px/AU:
    # contact radius must stay below half the distance to its nearest neighbour.
def test_an_asteroid_aimed_at_the_drawn_sun_is_destroyed():
    # The reported bug. Impact parameter 50% of the drawn solar disc at 64 px/AU
    # -- which flew straight through before -- must now end the asteroid.
def test_real_size_mode_uses_true_radii_for_contact_too():
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement.** `World.__init__` allocates `contact_radius` alongside `radius`; `add`, `remove` and `remove_many` keep it parallel and default it to the true radius. `find_collisions` reads `world.contact_radius[:n]` for `touch` and keeps `world.radius` for `sep`, `b` and every energy. In `render/scale.py`:

```python
#: How far past its true radius a body's contact radius may be stretched so
#: that aiming at what is drawn works.  Uncapped, the Sun's drawn disc is 0.6 AU
#: across at 10 px/AU and would eat Mercury the moment the player zoomed out.
MAX_AIM_FACTOR = 8.0


def contact_radii_au(radii, zoom, internal_height, real_size=False):
    """Radii to collide against: what is drawn, capped so orbits stay safe."""
    radii = np.asarray(radii, dtype=np.float64)
    drawn = display_radii_px(radii, zoom, internal_height, real_size) / max(zoom, 1e-30)
    return np.clip(drawn, radii, radii * MAX_AIM_FACTOR)
```

`App.step_once` calls it once per frame, before `sim.advance`.

- [ ] **Step 4: Run the file and the suite.**
- [ ] **Step 5: Commit** — `fix: an asteroid aimed at the drawn Sun now hits it`

---

### Task 8: Tidal and thermal destruction

**Files:**
- Create: `src/destruction.py`
- Modify: `src/sim.py`
- Test: `tests/test_destruction.py` (create)

**Interfaces:**
- Produces: `destruction.tidal_stress_ratio(world, i, k) -> float`, `destruction.equilibrium_temperature(world, i, k) -> float`, `destruction.apply(world, resolver, dt) -> list[ImpactEvent]`.

- [ ] **Step 1: Write the failing test**

```python
def test_a_comet_passing_close_to_the_sun_is_boiled_away():
def test_a_body_at_one_au_is_untouched_by_heat():
def test_a_loosely_bound_rubble_pile_is_torn_apart_inside_the_roche_limit():
def test_a_small_strong_rock_survives_the_same_pass():
def test_destruction_conserves_nothing_it_should_not_touch():  # n, ids, masses at 1 AU
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement.** Stars carry an effective temperature derived from mass (`T = 5772 K * (M/Msun)**0.5`, a serviceable main-sequence approximation) in `src/units.py` as `star_temperature(mass)`. For each non-star body and each star:

- tidal: `2 G M_star R_body / d**3` against `G m_body / R_body**2 + strength_accel`, where `strength_accel = toughness * QD_STRENGTH_SI * ... / R_body` reduced to a simple `units.strength_acceleration(radius, toughness)`. Above 1.0, disrupt via the resolver.
- thermal: `T = star_temperature(M) * sqrt(R_star / (2 d))`; above the material's `vapour_temperature_k`, ablate mass at a rate proportional to the excess and to `dt`, rederiving radius from the unchanged density; remove the body when the mass falls below a floor, emitting a `VAPORISE` impact event.

`Simulation.advance` calls `destruction.apply` once per substep and extends `self.impacts`.

- [ ] **Step 4: Run the file and the suite.**
- [ ] **Step 5: Commit** — `feat: sungrazers are torn apart and boiled away`

---

### Task 9: Impact events

**Files:**
- Modify: `src/collisions.py`, `src/sim.py`
- Test: `tests/test_sim.py`, `tests/test_layering.py`

**Interfaces:**
- Produces: `collisions.ImpactEvent` dataclass with fields `pos` (tuple of 2 floats), `vel` (tuple of 2), `energy` (float, sim units), `outcome` (str), `target_radius`, `projectile_radius` (floats), `tones` (tuple of three RGB tuples), `on_star` (bool). `Simulation.impacts` is a list, cleared at the top of `advance`; `Simulation.events` stays an int count for compatibility.

- [ ] **Step 1: Write the failing test**

```python
def test_a_collision_records_an_impact_event():
def test_impacts_are_cleared_at_the_start_of_each_advance():
def test_impact_events_hold_only_plain_data():   # no numpy arrays, no surfaces
```

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement.** `CollisionResolver.resolve` builds the record from the pair *before* it mutates the world and returns it alongside the count.
- [ ] **Step 4: Run the suite, including `tests/test_layering.py` which must still pass.**
- [ ] **Step 5: Commit** — `feat: the simulation reports what it just smashed`

---

### Task 10: The effects layer

**Files:**
- Create: `src/render/effects.py`
- Test: `tests/test_effects.py` (create)

**Interfaces:**
- Produces: `EffectField(seed=0)` with `spawn(event, camera, internal_height)`, `update(dt_seconds)`, `draw(surface, camera)`, `count` property, and `MAX_PARTICLES = 400`.

- [ ] **Step 1: Write the failing test**

```python
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

def test_an_impact_spawns_particles():
def test_particles_expire_on_wall_clock_time_not_simulated_time():
def test_the_particle_cap_holds_under_a_flood_of_impacts():
def test_drawing_an_empty_field_is_a_no_op():
def test_a_star_impact_spawns_a_flare_and_a_planet_impact_does_not():
def test_effects_drift_with_the_impact_velocity():
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement** the five elements from the spec: `Flash`, `Ring`, `Spark`, `Plume`, `Flare`. All hold world positions and world velocities, advance on `dt` in real seconds, and are drawn through `camera.world_to_screen`. Colour ramps are computed from a normalised age with a small blackbody-ish white→yellow→orange→red table in `render/palette.py`. Additive blending (`pygame.BLEND_ADD`) for flash, ring and flare; plain blits for sparks and plume.

- [ ] **Step 4: Run the file and the suite.**
- [ ] **Step 5: Commit** — `feat: impacts flash, ring and throw sparks`

---

### Task 11: Wire it into the frame

**Files:**
- Modify: `src/app/main.py`, `src/app/state.py`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `AppState.effects` (an `EffectField`).

- [ ] **Step 1: Write the failing test**

```python
def test_the_app_survives_a_real_collision():
    # Put two bodies on a collision course, drive step_once until they hit,
    # and assert the app raised nothing and the effect field became non-empty.
def test_effects_are_drawn_after_the_world_and_before_the_hud():
```

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement.** In `step_once`, after `ui.update` and before `sim.advance`, refresh contact radii from the camera; after `sim.advance`, drain `sim.impacts` into `state.effects`; after `renderer.draw` and before `tool.draw_overlay`, call `state.effects.update(frame_dt)` and `state.effects.draw(surface, camera)`.
- [ ] **Step 4: Run the whole suite.**
- [ ] **Step 5: Commit** — `feat: show the explosion`

---

### Task 12: Documentation

**Files:**
- Modify: `CLAUDE.md`, `docs/superpowers/specs/2026-08-18-solar-system-sandbox-design.md`

- [ ] **Step 1** Add to CLAUDE.md's invariants: contact radius versus true radius and what each is for; that outcome thresholds are Q\*_D, not binding energy; that effects advance on wall-clock time. Add `src/destruction.py` and `src/render/effects.py` to the architecture table.
- [ ] **Step 2** Note in the original design doc that collision handling has been superseded by the new spec, with a link.
- [ ] **Step 3** Run `./.venv/bin/python selftest.py` one final time.
- [ ] **Step 4: Commit** — `docs: write down what an impact now does`
