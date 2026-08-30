# Body Temperature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every body carries a real, displayed, editable temperature that rises with starlight and with the heat of nearby bodies, and falls by radiating.

**Architecture:** A new `world.temperature` array holds the state. A new pure-numpy `src/thermal.py` owns the model: an equilibrium target from the summed blackbody flux of every other body, and an analytic exponential relaxation toward it on a 1-metre thermal skin. Stars and black holes are pinned to their own physics and never relax. `Simulation.advance` steps it once a frame; `destruction.py`, the Edit tool, the HUD and the sprite cache read it.

**Tech Stack:** Python 3.11, numpy, pygame 2.6.1. Tests with pytest.

**Spec:** `docs/superpowers/specs/2026-08-29-body-temperature-design.md`

## Global Constraints

- **Simulation units are AU, solar masses, days.** `src/units.py` is the single source of truth for every constant and conversion factor. No other module may define one. Temperatures are the one exception in kind: they are in **kelvin**, an SI unit, because there is no useful sandbox-scaled temperature and every source in the literature is in kelvin. Say so in the docstring.
- **`src/*.py` must never import pygame.** `tests/test_layering.py` enforces it in a subprocess. `src/thermal.py` imports `math`, `numpy`, `src.units` and `src.body`, and nothing else.
- **Never hold an array index across a frame.** `World.remove` is a swap-remove.
- **Every mutation of mass, radius or membership must be followed by `Simulation.refresh()`.**
- **`World.contact_radius` is not `World.radius`.** Any mutation of `radius` must reset `contact_radius` to match.
- **`World.MAX_BODIES` is 200 and must not be raised.**
- Run the venv Python: `./.venv/bin/python`. Tests need `PYTHONPATH` set to the repo root, or run via `./.venv/bin/python -m pytest`.
- Whole suite: `./.venv/bin/python selftest.py` (~3:40). One file: `./.venv/bin/python -m pytest tests/test_thermal.py -q`.
- Comments and docstrings in this codebase explain **why**, not what. Match that. British spelling in prose (`colour`, `vaporise`), en-dash written as `--`.

---

### Task 1: Radiation constants in `units.py`

**Files:**
- Modify: `src/units.py` (append a new section after the black holes section, before "Conversions to human-facing units")
- Test: `tests/test_thermal.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `units.STEFAN_BOLTZMANN`, `units.CMB_TEMPERATURE_K`, `units.THERMAL_SKIN_DEPTH_M`, `units.MIN_TEMPERATURE_K`, `units.MAX_TEMPERATURE_K`, `units.ZERO_CELSIUS_K`, `units.to_celsius(kelvin) -> float`, `units.to_kgm3(density) -> float`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_thermal.py` with exactly this:

```python
"""Temperature: what sets it, what changes it, and how fast."""

import math

import numpy as np
import pytest

from src import materials, thermal, units
from src.body import KIND_BLACK_HOLE, KIND_STAR
from src.world import World

ROCK = materials.get("rock")
ICE = materials.get("ice")
HYDROGEN = materials.get("hydrogen")


# --- constants and conversions -------------------------------------------------

def test_stefan_boltzmann_is_the_si_value():
    assert units.STEFAN_BOLTZMANN == pytest.approx(5.670374419e-8, rel=1e-9)


def test_the_microwave_background_is_2_point_725_kelvin():
    assert units.CMB_TEMPERATURE_K == pytest.approx(2.725, rel=1e-6)


def test_celsius_is_kelvin_less_absolute_zero():
    assert units.to_celsius(273.15) == pytest.approx(0.0, abs=1e-9)
    assert units.to_celsius(0.0) == pytest.approx(-273.15, abs=1e-9)
    assert units.to_celsius(373.15) == pytest.approx(100.0, abs=1e-9)


def test_density_converts_to_si_kilograms_per_cubic_metre():
    """Water is 1000 kg/m3 by definition, whichever way round we go."""
    assert units.to_kgm3(units.from_gcm3(1.0)) == pytest.approx(1000.0, rel=1e-9)


def test_the_editable_range_spans_the_background_to_hotter_than_any_star():
    assert units.MIN_TEMPERATURE_K == pytest.approx(units.CMB_TEMPERATURE_K)
    assert units.MAX_TEMPERATURE_K > units.SUN_TEMPERATURE_K * 10.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_thermal.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'src.thermal'`.

- [ ] **Step 3: Create a stub `src/thermal.py` so the import resolves**

Create `src/thermal.py` containing only:

```python
"""Placeholder -- filled in by Task 4."""
```

Re-run. Expected now: five failures, `AttributeError: module 'src.units' has no attribute 'STEFAN_BOLTZMANN'`.

- [ ] **Step 4: Add the constants to `src/units.py`**

Insert this block in `src/units.py` immediately **before** the line `# --- Conversions to human-facing units ---`:

```python
# --- Radiation and temperature ------------------------------------------------------
#
# Temperatures are in **kelvin**, not a sandbox-scaled unit.  Every source in the
# literature is in kelvin, there is no natural AU/solar-mass/day temperature to
# scale to, and the readout has to print a number a person recognises.  So this
# is the one SI quantity that crosses into the physics layer, and the conversions
# below are what keep it honest.

#: Stefan-Boltzmann constant, W m^-2 K^-4.  Radiated power goes as T^4, which is
#: why a glowing body cools in hours and a frozen one takes centuries.
STEFAN_BOLTZMANN = 5.670374419e-8

#: The cosmic microwave background, kelvin.  Nothing gets colder than this:
#: however far a body runs from every star, it still sits in a 2.725 K bath and
#: comes to equilibrium with it.  Sunlight does not drop this low until 10,400 AU.
CMB_TEMPERATURE_K = 2.725

#: How deep a layer takes part in a body's temperature changes, metres.
#:
#: Heating a *whole* planet gives Earth a thermal time constant near a hundred
#: thousand years -- entirely real, and entirely invisible in a sandbox where a
#: frame is a day.  The reason real surfaces equilibrate in days is that only a
#: thin skin follows the changing sunlight beneath it.  This is that skin, and it
#: is the one free parameter in the whole thermal model.
THERMAL_SKIN_DEPTH_M = 1.0

#: A body's temperature may not be scrubbed below the background it sits in, nor
#: above this -- hotter than any star's surface, and well clear of anything the
#: T^4 arithmetic could overflow on.
MIN_TEMPERATURE_K = CMB_TEMPERATURE_K
MAX_TEMPERATURE_K = 1.0e5

#: The freezing point of water in kelvin -- the offset between the unit the
#: physics is in and the unit the audience has.
ZERO_CELSIUS_K = 273.15


def to_celsius(kelvin):
    """Kelvin to degrees Celsius."""
    return kelvin - ZERO_CELSIUS_K


def to_kgm3(density):
    """Density in simulation units to kg/m^3, which is what the thermal
    arithmetic wants: specific heats are quoted per kilogram."""
    return density / KGM3_TO_SIM


```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_thermal.py -q`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/units.py src/thermal.py tests/test_thermal.py
git commit -m "feat: radiation constants and temperature conversions in units"
```

---

### Task 2: Albedo and heat capacity on `Material`

**Files:**
- Modify: `src/materials.py:20-35` (the dataclass), `src/materials.py:38-110` (every `Material(...)` literal including `SINGULARITY`)
- Test: `tests/test_thermal.py` (append)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `Material.albedo: float`, `Material.heat_capacity_j_kgk: float` on every material in `materials.MATERIALS` and on `materials.SINGULARITY`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_thermal.py`:

```python
# --- material properties -------------------------------------------------------

def test_every_material_has_a_real_albedo_and_heat_capacity():
    for m in materials.MATERIALS:
        assert 0.0 <= m.albedo < 1.0, m.key
        assert m.heat_capacity_j_kgk > 0.0, m.key


def test_ice_is_bright_and_water_is_dark():
    """Albedo is what makes an icy world colder than a rocky one at the same
    distance -- the physical fact, not a tuning knob."""
    assert materials.get("ice").albedo > materials.get("rock").albedo
    assert materials.get("water").albedo < materials.get("rock").albedo


def test_water_stores_far_more_heat_than_iron():
    """4180 against 450 J/kg/K: the real ratio, and why oceans moderate
    climate and a frying pan does not."""
    assert (materials.get("water").heat_capacity_j_kgk
            > 8.0 * materials.get("iron").heat_capacity_j_kgk)


def test_a_singularity_reflects_nothing_and_cannot_be_heated():
    assert materials.SINGULARITY.albedo == 0.0
    assert materials.SINGULARITY.heat_capacity_j_kgk == math.inf
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_thermal.py -q`
Expected: 4 failures, `AttributeError: 'Material' object has no attribute 'albedo'`.

- [ ] **Step 3: Add the two fields to the dataclass**

In `src/materials.py`, in the `Material` dataclass, immediately after the `vapour_temperature_k` field, add:

```python
    #: Fraction of incident light reflected rather than absorbed, 0..1.  Real
    #: bolometric values: fresh ice throws most of the light back, open water
    #: swallows nearly all of it, and that alone is tens of kelvin of difference
    #: between two worlds at the same distance from the same star.
    albedo: float = 0.15
    #: Specific heat capacity, J/kg/K -- what one kelvin costs.  Together with
    #: density it sets how long a body takes to warm up or cool down.
    heat_capacity_j_kgk: float = 800.0
```

- [ ] **Step 4: Give every material its real values**

Edit each `Material(...)` literal in `MATERIALS`, extending its existing keyword line. The complete set:

| key | add |
|---|---|
| `iron` | `albedo=0.20, heat_capacity_j_kgk=450.0,` |
| `rock` | `albedo=0.15, heat_capacity_j_kgk=800.0,` |
| `rock_ice` | `albedo=0.35, heat_capacity_j_kgk=1400.0,` |
| `water` | `albedo=0.06, heat_capacity_j_kgk=4180.0,` |
| `ice` | `albedo=0.60, heat_capacity_j_kgk=2000.0,` |
| `gas` | `albedo=0.50, heat_capacity_j_kgk=10000.0,` |
| `hydrogen` | `albedo=0.40, heat_capacity_j_kgk=14300.0,` |

For example, `iron` becomes:

```python
    Material(
        "iron", "Iron", 7.87,
        ((0x3A, 0x3C, 0x42), (0x6B, 0x6E, 0x75), (0xC0, 0x8A, 0x5A)),
        "cratered",
        "Iron worlds are tiny for how heavy they are.",
        toughness=3.0, vapour_temperature_k=3130.0,
        albedo=0.20, heat_capacity_j_kgk=450.0,
    ),
```

And `SINGULARITY` becomes:

```python
SINGULARITY = Material(
    "singularity", "Singularity", 1.0,
    ((0x40, 0x08, 0x04), (0xE0, 0x70, 0x18), (0xFF, 0xF0, 0xD0)),
    "disk",
    "Not made of anything. Just gravity, wound too tight to let go.",
    toughness=1e9, vapour_temperature_k=float("inf"),
    albedo=0.0, heat_capacity_j_kgk=float("inf"),
)
```

A horizon reflects nothing, and an infinite heat capacity is the arithmetic saying what `thermal.pinned_temperature` says in words: a hole's temperature is not something an impact or a nearby star can move.

- [ ] **Step 5: Run to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_thermal.py tests/test_disruption.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/materials.py tests/test_thermal.py
git commit -m "feat: real albedo and specific heat on every material"
```

---

### Task 3: The `temperature` array on `World`

**Files:**
- Modify: `src/world.py` (`__init__`, `add`, `remove`, and the views section)
- Test: `tests/test_world.py` (append)

**Interfaces:**
- Consumes: `units.CMB_TEMPERATURE_K` from Task 1.
- Produces: `World.temperature` (numpy float64 array, length `capacity`), `World.temperatures` (view property over `[:n]`), and `World.add(..., temperature=None)` where `None` means the microwave background.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_world.py`:

```python
def test_a_new_body_starts_at_the_microwave_background():
    """Not zero.  Absolute zero is not a place anything in the universe is,
    and a body created at 0 K would then spend its first frames warming up
    from a temperature it was never at."""
    from src import units
    w = World(4)
    b = w.add("Rock", materials.get("rock"), 1e-6, 1e-5,
              np.array([1.0, 0.0]), np.zeros(2))
    assert w.temperature[w.index_of(b.id)] == pytest.approx(units.CMB_TEMPERATURE_K)


def test_a_body_can_be_created_at_a_chosen_temperature():
    w = World(4)
    b = w.add("Ember", materials.get("rock"), 1e-6, 1e-5,
              np.array([1.0, 0.0]), np.zeros(2), temperature=1500.0)
    assert w.temperature[w.index_of(b.id)] == pytest.approx(1500.0)


def test_removal_carries_the_temperature_with_the_swapped_body():
    """The swap-remove has to move temperature along with mass and radius, or
    a body silently inherits the temperature of whichever row it landed in."""
    w = World(4)
    a = w.add("A", materials.get("rock"), 1e-6, 1e-5, np.array([1.0, 0.0]),
              np.zeros(2), temperature=100.0)
    b = w.add("B", materials.get("rock"), 1e-6, 1e-5, np.array([2.0, 0.0]),
              np.zeros(2), temperature=200.0)
    c = w.add("C", materials.get("rock"), 1e-6, 1e-5, np.array([3.0, 0.0]),
              np.zeros(2), temperature=300.0)
    w.remove(w.index_of(a.id))
    assert w.temperature[w.index_of(b.id)] == pytest.approx(200.0)
    assert w.temperature[w.index_of(c.id)] == pytest.approx(300.0)


def test_the_temperatures_view_is_as_long_as_the_world():
    w = World(8)
    for k in range(3):
        w.add("B%d" % k, materials.get("rock"), 1e-6, 1e-5,
              np.array([float(k + 1), 0.0]), np.zeros(2))
    assert w.temperatures.shape == (3,)
```

Check the top of `tests/test_world.py` for the imports it already has; add `import pytest`, `import numpy as np` and `from src import materials` only if they are missing.

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_world.py -q`
Expected: 4 failures, `AttributeError: 'World' object has no attribute 'temperature'`.

- [ ] **Step 3: Add the array**

In `src/world.py`, in `World.__init__`, immediately after the `self.contact_radius = ...` block, add:

```python
        #: Surface temperature in kelvin.  Physical state like mass and radius,
        #: not appearance like ``Body.spin`` -- it changes every step, so it
        #: belongs in the arrays where a vectorised pass can reach it.
        self.temperature = np.zeros(capacity, dtype=np.float64)
```

Change the `add` signature to:

```python
    def add(self, name, material, mass, radius, pos, vel,
            kind=KIND_PLANET, seed=0, temperature=None):
```

and extend its docstring with:

```
        ``temperature`` defaults to the microwave background, which is where a
        body with nothing shining on it actually sits.  Callers that know
        better -- fragments inheriting their parent's heat, a scenario built in
        thermal balance -- pass it.
```

In the body of `add`, immediately after `self.contact_radius[i] = radius`, add:

```python
        self.temperature[i] = (units.CMB_TEMPERATURE_K if temperature is None
                               else float(temperature))
```

In `remove`, inside the `if index != last:` block, immediately after
`self.contact_radius[index] = self.contact_radius[last]`, add:

```python
            self.temperature[index] = self.temperature[last]
```

In the views section, immediately after the `contact_radii` property, add:

```python
    @property
    def temperatures(self):
        return self.temperature[:self.n]
```

- [ ] **Step 4: Run to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_world.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/world.py tests/test_world.py
git commit -m "feat: World carries a temperature array"
```

---

### Task 4: Equilibrium temperature in `src/thermal.py`

**Files:**
- Modify: `src/thermal.py` (replace the Task 1 stub entirely)
- Modify: `tests/test_layering.py:12-16`
- Test: `tests/test_thermal.py` (append)

**Interfaces:**
- Consumes: `units.CMB_TEMPERATURE_K`, `units.star_temperature`, `units.hawking_temperature_k`, `Material.albedo`, `World.temperature`.
- Produces:
  - `thermal.pinned_temperature(world, i) -> float | None`
  - `thermal.radiator_temperatures(world) -> np.ndarray` (length `world.n`)
  - `thermal.equilibrium_temperatures(world) -> np.ndarray` (length `world.n`)
  - `thermal.equilibrium_temperature(world, i) -> float`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_thermal.py`:

```python
# --- equilibrium ---------------------------------------------------------------

def _sun_and(material, distance, mass=units.EARTH_MASS,
             radius=units.EARTH_RADIUS, sun_mass=1.0):
    """A star at the origin and one body ``distance`` AU away."""
    world = World(16)
    world.add("Sun", HYDROGEN, sun_mass, units.SUN_RADIUS,
              np.zeros(2), np.zeros(2), kind=KIND_STAR)
    world.add("Body", material, mass, radius,
              np.array([distance, 0.0]), np.zeros(2))
    return world


class _Black:
    """A perfectly absorbing stand-in, so a test can check the bare formula
    without an albedo in the way."""
    albedo = 0.0
    heat_capacity_j_kgk = 800.0
    key = "black"
    tones = ((0, 0, 0), (0, 0, 0), (0, 0, 0))
    vapour_temperature_k = 1e9


def test_a_black_body_at_one_au_sits_at_the_textbook_temperature():
    """T = T_star * sqrt(R_star / 2d).  278.3 K is the number every textbook
    quotes for Earth before albedo, and it is what falls out of the sum."""
    world = _sun_and(ROCK, 1.0)
    world.bodies[1].material = _Black()
    assert thermal.equilibrium_temperature(world, 1) == pytest.approx(278.3, abs=0.5)


def test_temperature_falls_as_the_square_root_of_distance():
    """Flux goes as 1/d^2 and temperature as its fourth root, so four times
    further is half as hot."""
    near = _sun_and(ROCK, 1.0)
    far = _sun_and(ROCK, 4.0)
    assert (thermal.equilibrium_temperature(far, 1)
            == pytest.approx(0.5 * thermal.equilibrium_temperature(near, 1), rel=1e-3))


def test_albedo_cools_a_body_by_the_fourth_root_of_what_it_reflects():
    world = _sun_and(ROCK, 1.0)
    black = _sun_and(ROCK, 1.0)
    black.bodies[1].material = _Black()
    expected = (1.0 - ROCK.albedo) ** 0.25
    assert (thermal.equilibrium_temperature(world, 1)
            == pytest.approx(expected * thermal.equilibrium_temperature(black, 1),
                             rel=1e-6))


def test_an_icy_world_is_colder_than_a_rocky_one_at_the_same_distance():
    rocky = _sun_and(ROCK, 3.0)
    icy = _sun_and(ICE, 3.0)
    assert thermal.equilibrium_temperature(icy, 1) < thermal.equilibrium_temperature(rocky, 1)


def test_two_suns_add_their_light_not_their_temperatures():
    """Fluxes add; temperatures do not.  Two identical stars make a body
    2**0.25 = 1.19 times hotter, not twice."""
    one = _sun_and(ROCK, 1.0)
    two = _sun_and(ROCK, 1.0)
    two.add("Sun B", HYDROGEN, 1.0, units.SUN_RADIUS,
            np.array([0.0, 1.0e-9]), np.zeros(2), kind=KIND_STAR)
    assert (thermal.equilibrium_temperature(two, 1)
            == pytest.approx(2.0 ** 0.25 * thermal.equilibrium_temperature(one, 1),
                             rel=1e-3))


def test_a_warm_planet_heats_its_moon():
    """The same sum that makes the Sun heat Mercury.  A 288 K Earth puts about
    26 K on the Moon at its real distance -- small, real, and free: it needs
    no code of its own."""
    world = World(8)
    world.add("Earth", ROCK, units.EARTH_MASS, units.EARTH_RADIUS,
              np.zeros(2), np.zeros(2), temperature=288.0)
    world.bodies[0].material = _Black()
    world.add("Moon", ROCK, units.MOON_MASS, 1737.4 / units.AU_KM,
              np.array([384400.0 / units.AU_KM, 0.0]), np.zeros(2))
    world.bodies[1].material = _Black()
    assert thermal.equilibrium_temperature(world, 1) == pytest.approx(26.2, abs=1.0)


def test_a_molten_planet_heats_its_moon_far_harder():
    world = World(8)
    world.add("Earth", ROCK, units.EARTH_MASS, units.EARTH_RADIUS,
              np.zeros(2), np.zeros(2), temperature=1800.0)
    world.bodies[0].material = _Black()
    world.add("Moon", ROCK, units.MOON_MASS, 1737.4 / units.AU_KM,
              np.array([384400.0 / units.AU_KM, 0.0]), np.zeros(2))
    world.bodies[1].material = _Black()
    assert thermal.equilibrium_temperature(world, 1) == pytest.approx(163.9, abs=3.0)


def test_a_rogue_body_between_the_stars_sits_at_the_microwave_background():
    """Three light years out.

    The approach to the floor is far slower than it looks: flux falls as 1/d^2
    but temperature is its fourth root, so a body is still 0.9% above the
    background at 50,000 AU -- five times past the distance where the Sun's
    contribution and the CMB's are equal -- and only settles onto it out here.
    """
    world = _sun_and(ROCK, 200000.0)
    assert (thermal.equilibrium_temperature(world, 1)
            == pytest.approx(units.CMB_TEMPERATURE_K, rel=1e-3))


def test_nothing_is_ever_colder_than_the_microwave_background():
    """The floor is a floor at every distance, not only in the limit."""
    for distance in (1.0, 100.0, 5000.0, 50000.0, 1.0e7):
        world = _sun_and(ICE, distance)
        assert thermal.equilibrium_temperature(world, 1) >= units.CMB_TEMPERATURE_K


def test_a_star_is_pinned_to_the_temperature_its_mass_gives_it():
    world = _sun_and(ROCK, 1.0)
    assert thermal.pinned_temperature(world, 0) == pytest.approx(
        units.star_temperature(1.0))
    assert thermal.pinned_temperature(world, 1) is None


def test_a_black_hole_is_pinned_to_its_hawking_temperature():
    world = World(4)
    world.add("Hole", materials.SINGULARITY, 10.0,
              units.schwarzschild_radius(10.0), np.zeros(2), np.zeros(2),
              kind=KIND_BLACK_HOLE)
    assert thermal.pinned_temperature(world, 0) == pytest.approx(
        units.hawking_temperature_k(10.0))


def test_a_black_hole_contributes_no_meaningful_heat_to_anything():
    """No special case needed -- 6e-8 K over a horizon a few km wide simply
    evaluates to nothing."""
    world = World(4)
    world.add("Hole", materials.SINGULARITY, 10.0,
              units.schwarzschild_radius(10.0), np.zeros(2), np.zeros(2),
              kind=KIND_BLACK_HOLE)
    world.add("Rock", ROCK, units.EARTH_MASS, units.EARTH_RADIUS,
              np.array([0.01, 0.0]), np.zeros(2))
    assert thermal.equilibrium_temperature(world, 1) == pytest.approx(
        units.CMB_TEMPERATURE_K, rel=1e-6)


def test_equilibrium_of_an_empty_world_is_an_empty_array():
    assert thermal.equilibrium_temperatures(World(4)).shape == (0,)
```

`_Black` is a duck-typed stand-in rather than a real `Material` so a test can check the bare formula with no albedo in the way; it carries only the attributes the thermal code touches.

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_thermal.py -q`
Expected: failures with `AttributeError: module 'src.thermal' has no attribute 'equilibrium_temperature'`.

- [ ] **Step 3: Write `src/thermal.py`**

Replace the whole file with:

```python
"""How hot everything is.

One rule, and no special cases: **every body radiates as a blackbody at its own
temperature and its own true radius, and every body is warmed by the sum of what
falls on it.**  The Sun heating Mercury and a molten Earth heating the Moon are
the same term in the same sum, which is why "a hot planet warms its neighbours"
needed no code of its own.

Balancing absorption over a cross-section against emission over a whole surface:

    T_eq(i)^4  =  (1 - A_i)/4 * SUM_j T_j^4 (R_j / d_ij)^2  +  T_CMB^4

For a single star that reduces to the textbook `T = T_star sqrt(R_star/2d)
(1-A)^(1/4)`, which reproduces the real solar system to within a few kelvin --
Mercury 440 K, Earth 254 K, Neptune 47 K.  There is deliberately **no
greenhouse term**: it would be a tuned per-material constant rather than derived
physics, and one invented number would undermine every honest number beside it.
So Earth reads -19 C here, which is true, and is exactly the number that makes
the greenhouse effect worth explaining.

The microwave background is added as a floor with no albedo applied -- albedo is
a visible-light property, and in the microwave everything is black.  Nothing in
the sandbox gets colder than 2.725 K, and sunlight does not fall below it until
10,400 AU.

**Stars and black holes are pinned, never relaxed.**  A star's temperature is set
by fusion and follows its mass; a hole's is its Hawking temperature.  Neither
warms in its own light, and neither is editable.  A stellar-mass hole radiates at
six hundredths of a microkelvin across a horizon a few kilometres wide, so it
falls out of the general sum as nothing -- which is why it needs no exception.  A
*feeding* hole's accretion disk is genuinely luminous and is deliberately left
out: ``Body.disk`` is a 0..1 dial for how brightly to draw one, not an accretion
rate, and inventing a rate to get a luminosity would be inventing physics.

Imports numpy and nothing from pygame -- ``tests/test_layering.py`` asserts it.
"""

import math

import numpy as np

from . import units
from .body import KIND_BLACK_HOLE, KIND_STAR


def pinned_temperature(world, i):
    """The temperature this body's own physics fixes, or None if it relaxes.

    One function, so nothing can disagree about which bodies are held and which
    are free: ``advance`` writes these back every step, ``settle`` seeds from
    them, and ``edits`` refuses to scrub them.
    """
    kind = world.bodies[i].kind
    if kind == KIND_STAR:
        return units.star_temperature(float(world.mass[i]))
    if kind == KIND_BLACK_HOLE:
        return units.hawking_temperature_k(float(world.mass[i]))
    return None


def radiator_temperatures(world):
    """The temperature every body *shines* at.

    The stored array for anything that relaxes, the pinned value for anything
    that does not -- so an equilibrium is correct even on a world nobody has
    settled yet, which is what a freshly built scenario is.
    """
    n = world.n
    temps = world.temperature[:n].astype(np.float64, copy=True)
    for i in range(n):
        pinned = pinned_temperature(world, i)
        if pinned is not None:
            temps[i] = pinned
    return temps


def equilibrium_temperatures(world):
    """The temperature every body is heading toward, kelvin.

    O(n^2) and vectorised.  At the 200-body ceiling this is a 40,000-element
    matrix and costs a fraction of a millisecond -- which is affordable once a
    frame and nowhere near affordable once a substep.  See ``sim.advance``.
    """
    n = world.n
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    pos = world.pos[:n]
    radius = world.radius[:n]
    temps = radiator_temperatures(world)
    albedo = np.array([world.bodies[i].material.albedo for i in range(n)],
                      dtype=np.float64)

    delta = pos[:, None, :] - pos[None, :, :]
    dist2 = np.einsum("ijk,ijk->ij", delta, delta)
    # A body does not warm itself, and two bodies at the same point would
    # divide by zero rather than reach an infinite temperature.
    np.fill_diagonal(dist2, np.inf)
    dist2 = np.where(dist2 > 0.0, dist2, np.inf)

    flux = (temps ** 4 * radius ** 2)[None, :] / dist2
    absorbed = (1.0 - albedo) * 0.25 * flux.sum(axis=1)
    return np.clip(absorbed + units.CMB_TEMPERATURE_K ** 4, 0.0, None) ** 0.25


def equilibrium_temperature(world, i):
    """Where body ``i`` is heading, kelvin.  The whole-world sum, indexed --
    never a second implementation that could drift from it."""
    return float(equilibrium_temperatures(world)[i])
```

- [ ] **Step 4: Add the module to the layering test**

In `tests/test_layering.py`, change `PHYSICS_MODULES` to:

```python
PHYSICS_MODULES = [
    "src.units", "src.body", "src.materials", "src.world",
    "src.integrator", "src.orbits", "src.scenarios",
    "src.collisions", "src.destruction", "src.thermal", "src.sim", "src.edits",
]
```

`src.destruction` was missing from this list, which meant nothing stopped it importing pygame. Adding it costs nothing and closes the gap.

- [ ] **Step 5: Run to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_thermal.py tests/test_layering.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/thermal.py tests/test_thermal.py tests/test_layering.py
git commit -m "feat: equilibrium temperature from summed blackbody flux"
```

---

### Task 5: Relaxation, `advance` and `settle`

**Files:**
- Modify: `src/thermal.py` (append)
- Test: `tests/test_thermal.py` (append)

**Interfaces:**
- Consumes: everything from Task 4, plus `units.THERMAL_SKIN_DEPTH_M`, `units.STEFAN_BOLTZMANN`, `units.to_kgm3`, `units.density_from_mass_radius`, `units.SEC_PER_DAY`, `Material.heat_capacity_j_kgk`.
- Produces: `thermal.relaxation_days(world, i, t_eq) -> float`, `thermal.advance(world, days) -> None`, `thermal.settle(world) -> None`, `thermal.SETTLE_PASSES`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_thermal.py`:

```python
# --- relaxation ----------------------------------------------------------------

def test_a_hot_rock_near_earths_orbit_cools_over_days_not_millennia():
    """The whole point of the skin: heating a body right through gives a time
    constant near a hundred thousand years, which no one would ever see."""
    world = _sun_and(ROCK, 1.0)
    world.temperature[1] = 255.0
    tau = thermal.relaxation_days(world, 1, 255.0)
    assert 5.0 < tau < 20.0


def test_a_glowing_body_cools_far_faster_than_a_frozen_one():
    """Radiated power goes as T^4, so tau goes as 1/T^3.  Stefan-Boltzmann,
    not a tuning choice."""
    world = _sun_and(ROCK, 1.0)
    world.temperature[1] = 1500.0
    hot = thermal.relaxation_days(world, 1, 1500.0)
    world.temperature[1] = 50.0
    cold = thermal.relaxation_days(world, 1, 50.0)
    assert cold > 1000.0 * hot


def test_a_denser_body_holds_its_heat_longer():
    """rho is the body's own current density, so squeezing a planet in the
    Edit tool really does change how long it stays warm."""
    light = _sun_and(ROCK, 1.0, radius=units.EARTH_RADIUS)
    heavy = _sun_and(ROCK, 1.0, radius=units.EARTH_RADIUS * 0.5)
    light.temperature[1] = heavy.temperature[1] = 300.0
    assert (thermal.relaxation_days(heavy, 1, 300.0)
            > thermal.relaxation_days(light, 1, 300.0))


def test_relaxation_moves_toward_equilibrium_and_never_past_it():
    world = _sun_and(ROCK, 1.0)
    target = thermal.equilibrium_temperature(world, 1)
    world.temperature[1] = 3000.0
    previous = 3000.0
    for _ in range(50):
        thermal.advance(world, 1.0)
        current = float(world.temperature[1])
        assert current < previous
        assert current >= target - 1e-6
        previous = current


def test_a_cold_body_near_a_star_warms_up_instead_of_staying_frozen():
    """tau is linearised at whichever end of the exchange is hotter.  Taken at
    the body's own temperature instead, a body dropped to 3 K next to the Sun
    would take eighteen thousand years to notice."""
    world = _sun_and(ROCK, 1.0)
    world.temperature[1] = 3.0
    thermal.advance(world, 30.0)
    assert float(world.temperature[1]) > 200.0


def test_a_huge_step_lands_exactly_on_equilibrium_and_does_not_explode():
    """The step is integrated analytically, so dt >> tau collapses to T = T_eq.
    An explicit Euler step would need dt < tau, and tau is under an hour for
    anything glowing -- at ten years a second the sandbox would blow up."""
    world = _sun_and(ROCK, 1.0)
    target = thermal.equilibrium_temperature(world, 1)
    world.temperature[1] = 90000.0
    thermal.advance(world, 1.0e6)
    assert float(world.temperature[1]) == pytest.approx(target, rel=1e-6)


def test_settle_puts_everything_at_equilibrium_and_is_a_fixed_point():
    world = _sun_and(ROCK, 1.0)
    thermal.settle(world)
    settled = float(world.temperature[1])
    assert settled == pytest.approx(thermal.equilibrium_temperature(world, 1), rel=1e-9)
    thermal.advance(world, 100.0)
    assert float(world.temperature[1]) == pytest.approx(settled, rel=1e-9)


def test_settling_the_solar_system_reduces_to_the_textbook_law():
    """The end-to-end check, and the one worth making: the whole-world O(n^2)
    sum has to agree with `T = T_star sqrt(R_star / 2d) (1-A)^(1/4)` for every
    planet at the distance it actually is.

    Distances are the J2000 positions, not the semi-major axes: Mercury sits
    near aphelion at 0.4665 AU and comes out at 385 K, where its *average*
    distance would give 440 K.  Quoting a remembered textbook number here
    rather than the body's real distance is how you get a test that is wrong
    and confident about it.

    rel=1e-4 rather than tighter because the expected value below leaves out
    the microwave background, which is a few parts per million of Neptune's
    temperature and less than that for everything closer in.
    """
    from src import scenarios
    world = scenarios.solar_system(moons=False)
    thermal.settle(world)
    for i in range(1, world.n):
        body = world.bodies[i]
        d = float(np.hypot(*(world.pos[i] - world.pos[0])))
        expected = (units.star_temperature(float(world.mass[0]))
                    * math.sqrt(units.SUN_RADIUS / (2.0 * d))
                    * (1.0 - body.material.albedo) ** 0.25)
        assert float(world.temperature[i]) == pytest.approx(expected, rel=1e-4), body.name


def test_the_inner_planets_come_out_hotter_than_the_outer_ones():
    from src import scenarios
    world = scenarios.solar_system(moons=False)
    thermal.settle(world)
    by_name = {b.name: float(world.temperature[world.index_of(b.id)])
               for b in world.bodies}
    assert by_name["Mercury"] > by_name["Earth"] > by_name["Jupiter"] > by_name["Neptune"]
    assert 250.0 < by_name["Earth"] < 290.0
    assert 30.0 < by_name["Neptune"] < 50.0


def test_a_star_never_relaxes_however_long_it_runs():
    world = _sun_and(ROCK, 1.0)
    thermal.advance(world, 1.0e6)
    assert float(world.temperature[0]) == pytest.approx(units.star_temperature(1.0))


def test_advancing_by_nothing_changes_nothing():
    world = _sun_and(ROCK, 1.0)
    thermal.settle(world)
    before = world.temperature[:world.n].copy()
    thermal.advance(world, 0.0)
    assert np.allclose(world.temperature[:world.n], before)


def test_temperature_never_leaves_the_allowed_range():
    world = _sun_and(ROCK, 0.005)      # inside the Sun's own radius
    thermal.settle(world)
    for _ in range(20):
        thermal.advance(world, 10.0)
    assert units.MIN_TEMPERATURE_K <= float(world.temperature[1]) <= units.MAX_TEMPERATURE_K
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_thermal.py -q`
Expected: failures, `AttributeError: module 'src.thermal' has no attribute 'relaxation_days'`.

- [ ] **Step 3: Append the implementation to `src/thermal.py`**

```python
#: How many times ``settle`` sweeps the world.  One pass is exact for bodies
#: warmed only by stars, since a star's temperature is pinned rather than
#: solved for; the extra passes are what let planets warming *each other*
#: converge.  Those terms are tiny, so this converges immediately -- the loop
#: is here so the answer does not depend on the order bodies happen to sit in.
SETTLE_PASSES = 4

#: Largest exponent handed to exp().  Anything past this is zero to a double
#: anyway; clamping keeps a tau of microseconds from underflowing noisily.
_MAX_DECAY_EXPONENT = 700.0


def relaxation_days(world, i, t_eq):
    """The thermal time constant of body ``i``, in days.

        tau = rho * h * c_p / (4 sigma T_ref^3)

    ``rho`` is the body's *current* density, so squeezing a planet denser in
    the Edit tool really does make it hold its heat longer, and ``h`` is
    ``units.THERMAL_SKIN_DEPTH_M`` -- see that constant for why a skin rather
    than the whole body.

    ``T_ref`` is the *hotter* of the body and its target, not the body alone.
    The exchange runs at the rate the hot end sets: linearising at the body's
    own temperature would leave something dropped to 3 K beside the Sun frozen
    for eighteen thousand years, which is an artefact of the linearisation and
    not a fact about the world.
    """
    material = world.bodies[i].material
    c_p = material.heat_capacity_j_kgk
    mass = float(world.mass[i])
    radius = float(world.radius[i])
    if not math.isfinite(c_p) or c_p <= 0.0 or mass <= 0.0 or radius <= 0.0:
        return math.inf

    rho = units.to_kgm3(units.density_from_mass_radius(mass, radius))
    t_ref = max(float(world.temperature[i]), float(t_eq), units.CMB_TEMPERATURE_K)
    denominator = 4.0 * units.STEFAN_BOLTZMANN * t_ref ** 3
    if denominator <= 0.0:
        return math.inf
    seconds = rho * units.THERMAL_SKIN_DEPTH_M * c_p / denominator
    return seconds / units.SEC_PER_DAY


def _clamp(kelvin):
    return float(min(max(kelvin, units.MIN_TEMPERATURE_K), units.MAX_TEMPERATURE_K))


def advance(world, days):
    """Relax every body toward its equilibrium over ``days``.

    Integrated **analytically**, never by an explicit step::

        T <- T_eq + (T - T_eq) exp(-dt / tau)

    which is unconditionally stable and collapses to ``T = T_eq`` when the step
    dwarfs the time constant -- the physically right limit, and the reason the
    sandbox can run at ten years per second without a temperature diverging.
    An explicit step would need ``dt < tau``, and tau is under an hour for
    anything glowing.

    Stars and black holes are written back to their pinned values rather than
    relaxed: neither warms in its own light.
    """
    n = world.n
    if n == 0 or days <= 0.0:
        return
    target = equilibrium_temperatures(world)
    for i in range(n):
        pinned = pinned_temperature(world, i)
        if pinned is not None:
            world.temperature[i] = pinned
            continue
        t_eq = float(target[i])
        tau = relaxation_days(world, i, t_eq)
        if not math.isfinite(tau) or tau <= 0.0:
            world.temperature[i] = _clamp(t_eq)
            continue
        decay = math.exp(-min(days / tau, _MAX_DECAY_EXPONENT))
        current = float(world.temperature[i])
        world.temperature[i] = _clamp(t_eq + (current - t_eq) * decay)


def settle(world):
    """Put every body *at* its equilibrium, with no transient.

    Called when a world is built or reset, so a scenario opens in thermal
    balance.  Without it every planet would start at the microwave background
    and spend its first simulated months warming up to where it has in fact
    been sitting for four billion years.
    """
    for _ in range(SETTLE_PASSES):
        target = equilibrium_temperatures(world)
        for i in range(world.n):
            pinned = pinned_temperature(world, i)
            world.temperature[i] = (pinned if pinned is not None
                                    else _clamp(float(target[i])))
```

- [ ] **Step 4: Run to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_thermal.py -q`
Expected: all pass. If `test_settling_the_solar_system_gets_every_planet_about_right` fails, print the actual values and check them against the spec's table before changing any tolerance -- a real disagreement is a bug in the sum, not a tolerance problem.

- [ ] **Step 5: Commit**

```bash
git add src/thermal.py tests/test_thermal.py
git commit -m "feat: analytic thermal relaxation on a one-metre skin"
```

---

### Task 6: Impact heating, and collisions that carry temperature

**Files:**
- Modify: `src/thermal.py` (append `IMPACT_HEAT_FRACTION`, `impact_heating`)
- Modify: `src/collisions.py` -- imports, `merge`, `CollisionResolver.graze`, `CollisionResolver.accrete`, `CollisionResolver.tidal_disrupt`, `CollisionResolver.shatter`
- Test: `tests/test_thermal.py` (append)

**Interfaces:**
- Consumes: `units.SPECIFIC_ENERGY_SI`, `collisions.specific_impact_energy(world, c)`, `World.add(..., temperature=)`, `World.temperature`.
- Produces: `thermal.IMPACT_HEAT_FRACTION`, `thermal.impact_heating(material, specific_energy) -> float`. Every collision outcome leaves `world.temperature` set on its survivors and fragments.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_thermal.py`:

```python
# --- impact heating ------------------------------------------------------------

def test_impact_heating_scales_with_energy_and_against_heat_capacity():
    a = thermal.impact_heating(ROCK, 1.0)
    b = thermal.impact_heating(ROCK, 2.0)
    assert b == pytest.approx(2.0 * a)
    water = thermal.impact_heating(materials.get("water"), 1.0)
    assert water == pytest.approx(
        a * ROCK.heat_capacity_j_kgk / materials.get("water").heat_capacity_j_kgk)


def test_a_giant_impact_melts_a_planet():
    """About 5e6 J/kg for the Moon-forming impact, which is roughly 1900 K into
    rock.  Earth really was a magma ocean afterwards."""
    q_r = 5.0e6 / units.SPECIFIC_ENERGY_SI
    assert thermal.impact_heating(ROCK, q_r) == pytest.approx(1875.0, rel=0.1)


def test_a_singularity_cannot_be_heated_by_an_impact():
    assert thermal.impact_heating(materials.SINGULARITY, 1.0e9) == 0.0


def test_a_merge_mixes_the_two_temperatures_by_mass_and_adds_the_impact_heat():
    from src import collisions
    world = World(8)
    world.add("Big", ROCK, units.EARTH_MASS, units.EARTH_RADIUS,
              np.zeros(2), np.zeros(2), temperature=300.0)
    world.add("Small", ROCK, units.EARTH_MASS * 0.1, units.EARTH_RADIUS * 0.46,
              np.array([units.EARTH_RADIUS * 1.4, 0.0]),
              np.array([-0.005, 0.0]), temperature=100.0)
    c = collisions.Contact(i=0, j=1, distance=float(world.radius[0]),
                           impact_parameter=0.0)
    mixed = (units.EARTH_MASS * 300.0 + units.EARTH_MASS * 0.1 * 100.0) \
        / (units.EARTH_MASS * 1.1)
    heat = thermal.impact_heating(ROCK, collisions.specific_impact_energy(world, c))
    idx = collisions.merge(world, c)
    assert float(world.temperature[idx]) == pytest.approx(mixed + heat, rel=1e-6)


def test_an_impact_makes_the_remnant_hotter_than_either_body_was():
    from src import collisions
    world = World(8)
    world.add("Big", ROCK, units.EARTH_MASS, units.EARTH_RADIUS,
              np.zeros(2), np.zeros(2), temperature=300.0)
    world.add("Small", ROCK, units.EARTH_MASS * 0.1, units.EARTH_RADIUS * 0.46,
              np.array([units.EARTH_RADIUS * 1.4, 0.0]),
              np.array([-0.01, 0.0]), temperature=100.0)
    c = collisions.Contact(i=0, j=1, distance=float(world.radius[0]),
                           impact_parameter=0.0)
    idx = collisions.merge(world, c)
    assert float(world.temperature[idx]) > 300.0


def test_tidal_fragments_inherit_the_temperature_of_what_they_were():
    from src import collisions
    world = World(32)
    world.add("Sun", HYDROGEN, 1.0, units.SUN_RADIUS, np.zeros(2), np.zeros(2),
              kind=KIND_STAR)
    world.add("Doomed", ROCK, units.EARTH_MASS, units.EARTH_RADIUS,
              np.array([0.02, 0.0]), np.zeros(2), temperature=900.0)
    resolver = collisions.CollisionResolver(seed=1)
    made = resolver.tidal_disrupt(world, 1, 0)
    assert made, "expected fragments"
    for k in made:
        assert float(world.temperature[k]) == pytest.approx(900.0)


def test_no_collision_outcome_leaves_a_body_at_absolute_zero():
    """Every path that creates a body has to give it a temperature.  A fragment
    left at 0.0 would be colder than the universe allows and would then warm
    up from a temperature it was never at."""
    from src import collisions
    world = World(64)
    world.add("A", ROCK, units.EARTH_MASS, units.EARTH_RADIUS,
              np.zeros(2), np.zeros(2), temperature=400.0)
    world.add("B", ROCK, units.EARTH_MASS * 0.8, units.EARTH_RADIUS * 0.93,
              np.array([units.EARTH_RADIUS * 1.9, 0.0]),
              np.array([-0.05, 0.0]), temperature=400.0)
    c = collisions.Contact(i=0, j=1, distance=float(world.radius[0]),
                           impact_parameter=0.0)
    resolver = collisions.CollisionResolver(seed=3)
    resolver.shatter(world, c)
    assert world.n > 0
    for k in range(world.n):
        assert float(world.temperature[k]) >= units.MIN_TEMPERATURE_K
```

Before running, check the real constructor of `collisions.Contact` (near the top of `src/collisions.py`, close to `ImpactEvent`) and adjust the keyword names in these tests to match it exactly. If `find_collisions` is the only supported way to build one, build the pair through `find_collisions(world, prev_pos)` with a `prev_pos` that places the projectile clear of the target, and take `hits[0]`.

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_thermal.py -q -k "impact or merge or fragment or tidal or absolute"`
Expected: `AttributeError: module 'src.thermal' has no attribute 'impact_heating'`.

- [ ] **Step 3: Append `impact_heating` to `src/thermal.py`**

```python
#: Fraction of an impact's specific energy that stays behind as heat in what
#: survives.  Giant-impact studies put the retained share at 0.2 to 0.5 -- the
#: rest is radiated away or carried off by ejecta.  ``collisions.py`` already
#: observed, in its own comment, that most of an impact's energy goes to heat
#: rather than motion; nothing consumed that heat until now.
IMPACT_HEAT_FRACTION = 0.3


def impact_heating(material, specific_energy):
    """Temperature rise from an impact, kelvin.

    ``specific_energy`` is the Q_R of the literature in **simulation** units
    (AU^2/day^2), exactly what ``collisions.specific_impact_energy`` returns;
    ``units.SPECIFIC_ENERGY_SI`` converts it to the J/kg the specific heats are
    quoted in.  A Moon-forming impact delivers about 5e6 J/kg, so rock gains
    roughly 1900 K -- the remnant is a magma ocean, glows, and cools over
    months.  That is what really happened.
    """
    c_p = material.heat_capacity_j_kgk
    if not math.isfinite(c_p) or c_p <= 0.0:
        return 0.0
    return IMPACT_HEAT_FRACTION * float(specific_energy) * units.SPECIFIC_ENERGY_SI / c_p
```

- [ ] **Step 4: Carry temperature through every collision outcome**

In `src/collisions.py`, add `thermal` to the package import line so it reads:

```python
from . import edits, thermal, units
```

(keep whatever else that line already imports; `thermal` imports only `units` and `body`, so there is no cycle.)

**a. `merge`** -- inside `merge(world, c)`, capture the material and the impact energy *before* anything is mutated. Immediately after the line `total = mi + mj`, add:

```python
    # Both read before the pair is mutated: specific_impact_energy needs the
    # positions and velocities as they were on arrival, and a merge past the
    # collapse threshold replaces the survivor's material with SINGULARITY.
    heat = thermal.impact_heating(world.bodies[i].material,
                                  specific_impact_energy(world, c))
    mixed = (mi * float(world.temperature[i])
             + mj * float(world.temperature[j])) / total
```

Then immediately after the existing line `world.contact_radius[i] = world.radius[i]` (the one with the "Reset rather than leave stale" comment) and **before** `world.remove(j)`, add:

```python
    # Heat is conserved the way momentum is: the pair's own warmth mixes by
    # mass, and the impact adds its own on top.
    world.temperature[i] = min(mixed + heat, units.MAX_TEMPERATURE_K)
```

**b. `graze`** -- in `CollisionResolver.graze`, immediately after the line `total = mi + mj`, add (eight spaces of indentation -- it is a method body):

```python
        # Read before the velocities change: a graze is still an impact, and
        # both bodies keep a share of its heat.
        q_r = specific_impact_energy(world, c)
```

Then immediately before the `return i, j` line, add:

```python
        for k in (i, j):
            heat = thermal.impact_heating(world.bodies[k].material, q_r)
            world.temperature[k] = min(float(world.temperature[k]) + heat,
                                       units.MAX_TEMPERATURE_K)
```

**c. `accrete`** -- ejecta comes off the body that just merged, so it leaves at that body's temperature. Change the `world.add(` call in `accrete` (the one whose name argument is `"Ejecta"`) to pass it:

```python
            body = world.add(
                "Ejecta", material, mass,
                units.radius_from_mass_density(mass, density),
                positions[k], velocities[k],
                kind=KIND_DEBRIS, seed=int(self.rng.integers(0, 1 << 30)),
                temperature=float(world.temperature[idx]),
            )
```

**d. `tidal_disrupt`** -- a tide adds no impact heat, because no impact happened; the pieces are simply as warm as the body was. In `tidal_disrupt`, immediately after the line `pos_i, vel_i = world.pos[i].copy(), world.vel[i].copy()`, add:

```python
        # Captured before world.remove(i) below takes the row away.
        parent_temperature = float(world.temperature[i])
```

and change the `world.add(` call whose name argument is `"Debris"` to pass `temperature=parent_temperature,` as its final keyword argument, in the same style as (c).

**e. `shatter`** -- in `CollisionResolver.shatter`, immediately after the existing line `material = parent.material`, add:

```python
        # Read before world.remove_many([i, j]) below: the fragments carry the
        # pair's mixed heat plus what the impact put in.
        fragment_temperature = min(
            (mi * float(world.temperature[i]) + mj * float(world.temperature[j])) / total
            + thermal.impact_heating(material, q_r),
            units.MAX_TEMPERATURE_K)
```

and change the `world.add(` call whose name argument is `"Fragment"` to pass `temperature=fragment_temperature,` as its final keyword argument.

Note that `q_r` is already computed above this point in `shatter`; do not recompute it. The early-return branch that calls `merge(world, c)` needs no change -- `merge` now handles its own temperature.

- [ ] **Step 5: Run to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_thermal.py tests/test_collision_merge.py tests/test_collision_graze.py tests/test_collision_shatter.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/thermal.py src/collisions.py tests/test_thermal.py
git commit -m "feat: impacts heat what survives them"
```

---

### Task 7: Wire it into the simulation, and let ablation read it

**Files:**
- Modify: `src/sim.py` (imports, `__init__`, `reset`, `advance`)
- Modify: `src/destruction.py` (module docstring, `_bodies_and_attractors`, `apply`; remove `equilibrium_temperature`)
- Modify: `tests/test_destruction.py` (the `_sun_and` helper)
- Test: `tests/test_sim.py` (append)

**Interfaces:**
- Consumes: `thermal.advance`, `thermal.settle`.
- Produces: after any `Simulation.advance(days)` that simulates a non-zero span, `world.temperature` is up to date. `Simulation()` and `Simulation.reset()` leave the world thermally settled. `destruction.apply` reads `world.temperature[i]` and no longer computes an equilibrium itself. `destruction._bodies_and_attractors` now returns `(victims, attractors)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sim.py`:

```python
def test_a_new_simulation_opens_in_thermal_balance():
    """Not at absolute zero.  A scenario that had to warm up over its first
    simulated months would be showing a transient nobody asked for."""
    from src import thermal, units
    sim = Simulation()
    world = sim.world
    earth = next(b for b in world.bodies if b.name == "Earth")
    i = world.index_of(earth.id)
    assert float(world.temperature[i]) == pytest.approx(
        thermal.equilibrium_temperature(world, i), rel=1e-6)
    assert float(world.temperature[i]) > units.CMB_TEMPERATURE_K


def test_advancing_the_simulation_relaxes_a_body_that_was_heated():
    from src import units
    sim = Simulation()
    world = sim.world
    earth = next(b for b in world.bodies if b.name == "Earth")
    i = world.index_of(earth.id)
    settled = float(world.temperature[i])
    world.temperature[i] = 2000.0
    sim.advance(40.0)
    i = world.index_of(earth.id)
    cooled = float(world.temperature[i])
    assert cooled < 2000.0
    assert cooled > settled - 30.0


def test_a_paused_simulation_does_not_change_any_temperature():
    sim = Simulation()
    sim.paused = True
    before = sim.world.temperature[:sim.world.n].copy()
    sim.advance(10.0)
    assert np.allclose(sim.world.temperature[:sim.world.n], before)


def test_reset_settles_the_new_world():
    from src import thermal
    sim = Simulation()
    sim.world.temperature[:sim.world.n] = 1.0
    sim.reset("solar_system")
    world = sim.world
    for i in range(world.n):
        assert float(world.temperature[i]) == pytest.approx(
            thermal.equilibrium_temperatures(world)[i], rel=1e-6)
```

Check the imports already at the top of `tests/test_sim.py` and add `numpy as np` / `pytest` only if missing.

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_sim.py -q`
Expected: failures -- Earth's temperature is the microwave background, not its equilibrium.

- [ ] **Step 3: Wire `sim.py`**

Change the import line in `src/sim.py` to:

```python
from . import destruction, orbits, scenarios, thermal
```

In `Simulation.__init__`, immediately after `self.integrator.refresh(self.world)`, add:

```python
        thermal.settle(self.world)
```

In `Simulation.reset`, immediately after `self.integrator.refresh(self.world)`, add:

```python
        thermal.settle(self.world)
```

In `Simulation.advance`, immediately after the `while done < days:` loop closes and **before** `self.last_ratio = done / days`, add:

```python
        # Once a frame, over what was actually simulated -- never per substep.
        # The same call update_trails makes, and for the same reason: at ten
        # years a second the integrator takes thousands of substeps per frame,
        # and an O(n^2) thermal pass at 200 bodies would eat the wall-clock
        # budget many times over.  Temperature is slow-varying and feeds back
        # into nothing the integrator reads, so a frame is the right grain.
        thermal.advance(world, done)
```

- [ ] **Step 4: Let ablation read the stored temperature**

In `src/destruction.py`:

Delete the whole `equilibrium_temperature` function -- the model lives in `thermal.py` now, and two implementations of the same equation is one of them eventually being wrong.

Change `_bodies_and_attractors` to return two lists instead of three, and rewrite its docstring:

```python
def _bodies_and_attractors(world):
    """(victims, attractors) for this world.

    **Attractors** raise tides: stars and black holes both.  Who *radiates* is
    no longer this module's business -- ``thermal.py`` owns that, and every
    body here simply carries the temperature it worked out.

    A black hole is never a victim: nothing tears a horizon apart.  A star may
    be a victim, but only of a hole -- a tidal disruption event is the most
    spectacular thing in this simulation, while two ordinary stars would have
    to be nearly touching, and letting them shred each other would wreck every
    binary arrangement.
    """
    n = world.n
    kinds = [world.bodies[k].kind for k in range(n)]
    attractors = [k for k in range(n)
                  if kinds[k] in (KIND_STAR, KIND_BLACK_HOLE)]
    victims = [i for i in range(n) if kinds[i] != KIND_BLACK_HOLE]
    return victims, attractors
```

In `apply`, change the unpacking line to:

```python
    victims, attractors = _bodies_and_attractors(world)
```

and in the ablation loop replace:

```python
        hottest = max((equilibrium_temperature(world, i, k) for k in stars),
                     default=0.0)
```

with:

```python
        hottest = float(world.temperature[i])
```

Update the module docstring's second paragraph to say where the temperature comes from now. Replace the first paragraph's second sentence and the sentence about the thermal check with:

```
A body can be torn apart by a tide -- raised by a star or a black hole -- or
boiled away by its own heat, without ever touching what did it.  The tidal
check runs over every attractor; the thermal check reads the temperature
``thermal.py`` maintains, so there is one answer to "how hot is this" and the
ablation threshold is measured against the same number the readout shows.
Ablation therefore has real thermal lag: a comet warms over its approach
rather than snapping to equilibrium the instant it arrives.
```

Leave `KIND_STAR` imported -- the "a star does not boil in its own light" guard in the ablation loop still needs it, and it matters more now than it did: a star's pinned temperature is thousands of kelvin, far above any vaporisation point.

- [ ] **Step 5: Settle the world in the destruction tests**

In `tests/test_destruction.py`, add `thermal` to the import line:

```python
from src import collisions, destruction, materials, thermal, units
```

and add one line at the end of the `_sun_and` helper, immediately before `return world`:

```python
    thermal.settle(world)
```

These tests drive `destruction.apply` directly rather than through `Simulation.advance`, so nothing else would ever set a temperature for them to read. Settling reproduces exactly what the old instantaneous equilibrium gave, which is why both existing ablation tests keep passing unchanged.

- [ ] **Step 6: Run to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_sim.py tests/test_destruction.py tests/test_black_hole.py -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/sim.py src/destruction.py tests/test_sim.py tests/test_destruction.py
git commit -m "feat: the simulation steps temperature, and ablation reads it"
```

---

### Task 8: Editing a temperature

**Files:**
- Modify: `src/edits.py` (field tuples, `edit_fields`, `_refresh_kind`, `collapse`, `set_mass`, new scrub and setter)
- Modify: `src/app/tools.py` (`EditTool.CLICK_NOTCHES`, `field_values`, `_scrub`, `on_event`, new `_temperature_label`)
- Modify: `tests/test_tools_edit.py:35-42`
- Test: `tests/test_edits.py` (append), `tests/test_tools_edit.py` (append)

**Interfaces:**
- Consumes: `units.MIN_TEMPERATURE_K`, `units.MAX_TEMPERATURE_K`, `units.star_temperature`, `units.hawking_temperature_k`, `World.temperature`.
- Produces: `edits.PLANET_EDIT_FIELDS == ("mass", "density", "temp", "material")`, `edits.STAR_EDIT_FIELDS == ("mass", "density", "material")`, `edits.TEMPERATURE_NOTCH`, `edits.scrub_temperature(kelvin, notches) -> float`, `edits.set_temperature(world, index, kelvin) -> None`, `tools._temperature_label(kelvin) -> str`, and an `EditTool` whose Up/Down respects the selected body's own field count.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_edits.py`:

```python
def test_a_planet_can_be_given_a_temperature():
    from src import units
    world = _world()                       # match this file's existing helper
    edits.set_temperature(world, 1, 900.0)
    assert float(world.temperature[1]) == pytest.approx(900.0)


def test_temperature_is_clamped_to_the_background_and_the_ceiling():
    from src import units
    world = _world()
    edits.set_temperature(world, 1, -50.0)
    assert float(world.temperature[1]) == pytest.approx(units.MIN_TEMPERATURE_K)
    edits.set_temperature(world, 1, 1.0e12)
    assert float(world.temperature[1]) == pytest.approx(units.MAX_TEMPERATURE_K)


def test_temperature_scrubs_by_ratio_so_it_can_cross_five_orders_of_magnitude():
    """Linear steps cannot serve both a 3 K comet and a 100,000 K ember; the
    same argument that makes the mass scrub logarithmic."""
    up = edits.scrub_temperature(100.0, 1)
    assert up == pytest.approx(100.0 * edits.TEMPERATURE_NOTCH)
    assert edits.scrub_temperature(up, -1) == pytest.approx(100.0)


def test_a_star_refuses_to_have_its_temperature_set():
    """Its temperature is fusion's business and follows its mass.  The Edit
    panel offers no Temp row for one, and the setter enforces the same rule so
    no other caller can route around the missing button."""
    from src import units
    world = _world()
    before = float(world.temperature[0])   # index 0 is the star in the helper
    edits.set_temperature(world, 0, 10.0)
    assert float(world.temperature[0]) == pytest.approx(before)


def test_a_planet_that_ignites_takes_its_stars_temperature_at_once():
    from src import units
    world = _world()
    edits.set_mass(world, 1, units.STAR_MASS_THRESHOLD * 1.1)
    assert float(world.temperature[1]) == pytest.approx(
        units.star_temperature(float(world.mass[1])), rel=1e-9)


def test_a_collapsed_star_takes_its_hawking_temperature():
    from src import units
    world = _world()
    edits.set_mass(world, 1, 10.0)
    edits.collapse(world, 1)
    assert float(world.temperature[1]) == pytest.approx(
        units.hawking_temperature_k(10.0), rel=1e-9)


def test_a_planet_is_offered_four_fields_and_a_star_only_three():
    from src.body import KIND_PLANET, KIND_STAR
    world = _world()
    assert edits.edit_fields(world.bodies[1]) == ("mass", "density", "temp", "material")
    assert edits.edit_fields(world.bodies[0]) == ("mass", "density", "material")
    assert "temp" not in edits.edit_fields(world.bodies[0])
```

Read the top of `tests/test_edits.py` and reuse its existing world-building helper rather than writing a new one; rename `_world()` in the tests above to whatever that helper is actually called, and adjust the indices so index 0 is a star and index 1 an ordinary planet (build such a world if the existing helper does not give one).

Append to `tests/test_tools_edit.py`:

```python
def test_the_temp_row_scrubs_the_selected_bodys_temperature(state):
    index = state.selected_index
    before = float(state.sim.world.temperature[index])
    state.tool.field_index = 2                      # mass, density, temp, material
    state.tool.on_event(_key(pygame.K_RIGHT), state)
    assert float(state.sim.world.temperature[state.selected_index]) > before


def test_down_stops_at_the_last_field_of_the_selected_body(state):
    """Not at the last field of some other kind of body.  A star has three
    fields and a planet four, so a count taken from a module-level tuple walks
    off the end of one of them."""
    from src import edits
    body = state.sim.world.bodies[state.selected_index]
    expected = len(edits.edit_fields(body)) - 1
    for _ in range(10):
        state.tool.on_event(_key(pygame.K_DOWN), state)
    assert state.tool.field_index == expected


def test_the_edit_panel_offers_a_temp_row_for_a_planet(state):
    values = state.tool.field_values(state.sim.world, state.selected_index)
    assert "temp" in values
    assert values["temp"].endswith("K")
```

And change the existing `test_up_and_down_move_between_fields` to take its length from the selected body rather than the module constant:

```python
def test_up_and_down_move_between_fields(state):
    from src import edits
    fields = edits.edit_fields(state.sim.world.bodies[state.selected_index])
    assert state.tool.field_index == 0
    state.tool.on_event(_key(pygame.K_DOWN), state)
    assert state.tool.field_index == 1
    for _ in range(10):
        state.tool.on_event(_key(pygame.K_DOWN), state)
    assert state.tool.field_index == len(fields) - 1
    for _ in range(10):
        state.tool.on_event(_key(pygame.K_UP), state)
    assert state.tool.field_index == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_edits.py tests/test_tools_edit.py -q`
Expected: `AttributeError: module 'src.edits' has no attribute 'set_temperature'` and a failing field-count assertion.

- [ ] **Step 3: Extend `src/edits.py`**

Replace the field-tuple block with:

```python
PLANET_EDIT_FIELDS = ("mass", "density", "temp", "material")
#: A star is offered no temperature: its surface temperature is set by fusion
#: and follows its mass, so a scrub that appeared to change it would be a lie
#: the very next frame, when thermal.advance pins it straight back.
STAR_EDIT_FIELDS = ("mass", "density", "material")
BLACK_HOLE_EDIT_FIELDS = ("mass", "spin", "disk")
#: Retained for callers that predate black holes and only ever meant the
#: ordinary case.
EDIT_FIELDS = PLANET_EDIT_FIELDS


def edit_fields(body):
    """The editable fields for this body -- a black hole has no density or
    composition to speak of, so it is offered spin and accretion instead, and
    a star has no temperature of its own to set."""
    if body.kind == KIND_BLACK_HOLE:
        return BLACK_HOLE_EDIT_FIELDS
    if body.kind == KIND_STAR:
        return STAR_EDIT_FIELDS
    return PLANET_EDIT_FIELDS
```

Add, alongside the other notch constants:

```python
#: Ratio per scrub notch for temperature.  The range runs from the microwave
#: background to 100,000 K -- four and a half orders of magnitude, which is the
#: same argument that makes the mass scrub logarithmic.
TEMPERATURE_NOTCH = 1.06
```

Add, alongside the other scrubs:

```python
def scrub_temperature(kelvin, notches):
    """Logarithmic scrub, clamped to the range a body may occupy."""
    scaled = max(float(kelvin), units.MIN_TEMPERATURE_K) \
        * (TEMPERATURE_NOTCH ** notches)
    return float(min(max(scaled, units.MIN_TEMPERATURE_K),
                     units.MAX_TEMPERATURE_K))
```

Add, alongside the other setters:

```python
def set_temperature(world, index, kelvin):
    """Set a body's temperature.  It is not pinned there.

    Heat a world to 3000 K and it glows, boils away whatever is above its
    vaporisation point, and cools back to where the sunlight on it puts it.
    That relaxation is the whole lesson -- pinning an edited temperature would
    turn a physical quantity back into a dial.

    Refuses on a star or a black hole, whose temperatures their own physics
    fixes.  Guarded here as well as by the missing button, since the panel is
    not the only caller.
    """
    if world.bodies[index].kind in (KIND_STAR, KIND_BLACK_HOLE):
        return
    world.temperature[index] = float(
        min(max(float(kelvin), units.MIN_TEMPERATURE_K), units.MAX_TEMPERATURE_K))
```

In `_refresh_kind`, change the ignition branch so a body that lights up is hot at once rather than a frame later:

```python
    elif mass >= units.STAR_MASS_THRESHOLD:
        body.kind = KIND_STAR
        # Pinned immediately, not on the next thermal step, so the readout can
        # never show a star at a planet's temperature.
        world.temperature[index] = units.star_temperature(mass)
```

In `collapse`, immediately after `world.contact_radius[index] = world.radius[index]`, add:

```python
    world.temperature[index] = units.hawking_temperature_k(float(world.mass[index]))
```

In `set_mass`, in the black hole branch, immediately after
`world.contact_radius[index] = world.radius[index]`, add:

```python
        world.temperature[index] = units.hawking_temperature_k(mass)
```

- [ ] **Step 4: Extend `EditTool` in `src/app/tools.py`**

Add `"temp": 3` to `EditTool.CLICK_NOTCHES`, so it reads:

```python
    CLICK_NOTCHES = {"mass": 3, "density": 1, "material": 1,
                     "temp": 3, "spin": 1, "disk": 1}
```

Mass and temperature both scrub by ratio at 1.06 a notch, so one notch a click is invisible for both.

In `field_values`, extend the ordinary-body dict:

```python
        return {
            "mass": _mass_label(float(world.mass[index])),
            "density": "%.3g g/cm3" % units.to_gcm3(edits.body_density(world, index)),
            "temp": _temperature_label(float(world.temperature[index])),
            "material": world.bodies[index].material.name,
        }
```

A star's fields do not include `temp`, so the extra key is simply never read for one; leave the black hole dict alone.

In `_scrub`, add a branch before the final `else`:

```python
        elif field == "temp":
            edits.set_temperature(
                world, index,
                edits.scrub_temperature(float(world.temperature[index]), notches))
```

Add a helper beside `_mass_label`:

```python
def _temperature_label(kelvin):
    """Kelvin, grouped rather than in exponent notation -- "1e+04 K" is not a
    number this audience reads."""
    return "{:,.0f} K".format(kelvin)
```

Fix the field-count bug: `on_event`'s Down branch takes its bound from the module-level `EDIT_FIELDS`, not from the selected body. That was harmless while every kind had three fields and is a real bug now that a planet has four and a star three. Add a helper to `EditTool`:

```python
    def _field_count(self, state):
        """How many rows the *selected* body offers.

        A hotkey never passes through ``UiLayer.perform``, so it cannot lean on
        anything the panel knows -- it has to ask the same question the panel
        asked.
        """
        index = state.selected_index
        if index is None:
            return 1
        return len(self._fields(state.sim.world, index))
```

and change the Down branch to:

```python
            if event.key == pygame.K_DOWN:
                self.field_index = min(self._field_count(state) - 1,
                                       self.field_index + 1)
                return True
```

- [ ] **Step 5: Run to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_edits.py tests/test_tools_edit.py tests/test_layout.py tests/test_ui.py -q`
Expected: all pass. `tests/test_layout.py` is the one that would catch a fourth Edit row falling off a small window -- it was measured to have 2.4 spare rows of headroom, so it should pass, and if it does not, that is a real finding to report rather than a tolerance to widen.

- [ ] **Step 6: Commit**

```bash
git add src/edits.py src/app/tools.py tests/test_edits.py tests/test_tools_edit.py
git commit -m "feat: a Temp row in the Edit tool, and a field count that follows the body"
```

---

### Task 9: The temperature in the readout

**Files:**
- Modify: `src/render/hud.py` (new formatter, one row in `_rows`)
- Test: `tests/test_hud.py` (append)

**Interfaces:**
- Consumes: `units.to_celsius`, `World.temperature`.
- Produces: `hud.format_body_temperature(kelvin) -> str`, and a `Temp` row in the inspector for every body that is not a black hole.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hud.py`:

```python
def test_temperature_is_shown_in_kelvin_with_celsius_alongside():
    """Kelvin because that is the unit the physics is in; Celsius because that
    is the unit the audience has."""
    from src.render.hud import format_body_temperature
    assert format_body_temperature(273.15) == "273 K  (0 C)"
    assert format_body_temperature(254.0) == "254 K  (-19 C)"


def test_a_hot_temperature_uses_grouped_digits_not_an_exponent():
    from src.render.hud import format_body_temperature
    text = format_body_temperature(5772.0)
    assert "e+" not in text
    assert "5,772 K" in text


def test_the_inspector_shows_a_temperature_for_a_planet(state):
    from src.render.hud import Hud
    from src.render.pixelfont import PixelFont
    hud = Hud(PixelFont(10))
    state.select(next(b for b in state.sim.world.bodies if b.name == "Earth").id)
    rows = hud._rows(state, state.selected_index)
    assert any(text.startswith("Temp") for text, _ in rows)


def test_the_temperature_row_fits_the_inspector_panel():
    """The panel is a fixed number of characters wide; a row that overruns it
    is a number nobody can read."""
    from src.render.hud import PANEL_WIDTH_CHARS, format_body_temperature
    from src import units
    hottest = "Temp       %s" % format_body_temperature(units.MAX_TEMPERATURE_K)
    assert len(hottest) <= PANEL_WIDTH_CHARS
```

Reuse whatever `state` fixture `tests/test_hud.py` already defines; do not add a second one.

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_hud.py -q`
Expected: `ImportError: cannot import name 'format_body_temperature'`.

- [ ] **Step 3: Add the formatter**

In `src/render/hud.py`, immediately after the existing `format_temperature` function, add:

```python
def format_body_temperature(kelvin):
    """Kelvin with Celsius beside it -- "254 K  (-19 C)".

    Two units rather than one because the two readers need different ones: the
    physics is in kelvin and every threshold in the sandbox is quoted there,
    while an eight-year-old knows what -19 C feels like and does not yet know
    what 254 K means.  Grouped digits, never an exponent, for the same reason
    ``format_length`` refuses them.
    """
    return "{:,.0f} K  ({:,.0f} C)".format(kelvin, units.to_celsius(kelvin))
```

- [ ] **Step 4: Add the row**

In `Hud._rows`, in the `else` branch (the ordinary-body one), insert the temperature between the Earth-radii line and the Density line, so the block reads:

```python
                ("Radius     %s" % format_length(radius), palette.TEXT),
                ("           %.3g Earths" % units.to_earth_radii(radius),
                 palette.TEXT_DIM),
                ("Temp       %s" % format_body_temperature(
                    float(world.temperature[index])), palette.TEXT),
                ("Density    %.3g g/cm3" % units.to_gcm3(density), palette.TEXT),
```

A star reaches this same branch, and shows the surface temperature its mass gives it. A black hole keeps its own `Own heat` row and gains nothing.

- [ ] **Step 5: Run to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_hud.py tests/test_smoke.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/render/hud.py tests/test_hud.py
git commit -m "feat: the inspector shows a body's temperature in K and C"
```

---

### Task 10: Hot bodies glow

**Files:**
- Modify: `src/render/palette.py` (new `blackbody_rgb`)
- Modify: `src/render/sprites.py` (glow constants, `glow_fraction`, `glow_key`, `_blend_toward`, `_make_body`, `SpriteCache.get`)
- Modify: `src/render/draw.py:126-127` (pass the temperature)
- Test: `tests/test_sprites.py` (append)

**Interfaces:**
- Consumes: `World.temperature`.
- Produces: `palette.blackbody_rgb(kelvin) -> (int, int, int)`, `sprites.GLOW_MIN_K`, `sprites.GLOW_FULL_K`, `sprites.GLOW_TEMP_STEP`, `sprites.glow_fraction(kelvin) -> float`, `sprites.glow_key(kelvin) -> int`, and `SpriteCache.get(material, seed, px, lit_angle, is_star, temperature=units.CMB_TEMPERATURE_K)`.

The `temperature` argument is keyword-with-default so every existing call site and every existing test keeps working untouched.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sprites.py`:

```python
def test_blackbody_colour_runs_red_through_white_up_the_planck_locus():
    from src.render import palette
    ember = palette.blackbody_rgb(1000.0)
    sunlike = palette.blackbody_rgb(5772.0)
    assert ember[0] > ember[1] > ember[2]          # red, almost no blue
    assert min(sunlike) > 200                       # warm white
    assert palette.blackbody_rgb(1500.0)[2] < palette.blackbody_rgb(3500.0)[2]


def test_the_sun_comes_out_warm_white_not_yellow():
    """5772 K on the Planck locus is (255, 242, 230) -- the actual colour of
    sunlight above the atmosphere."""
    from src.render import palette
    r, g, b = palette.blackbody_rgb(5772.0)
    assert (r, g, b) == pytest.approx((255, 242, 230), abs=6)


def test_a_cold_body_has_no_glow_at_all(cache):
    from src.render import sprites
    assert sprites.glow_fraction(300.0) == 0.0
    assert sprites.glow_key(300.0) == 0


def test_glow_climbs_from_dull_red_to_fully_blackbody():
    from src.render import sprites
    assert sprites.glow_fraction(sprites.GLOW_MIN_K) == 0.0
    assert 0.0 < sprites.glow_fraction(1800.0) < 1.0
    assert sprites.glow_fraction(sprites.GLOW_FULL_K) == pytest.approx(1.0)
    assert sprites.glow_fraction(20000.0) == pytest.approx(1.0)


def test_a_glowing_body_looks_different_from_a_cold_one(cache):
    from src import materials
    rock = materials.get("rock")
    cold = cache.get(rock, seed=1, px=14, lit_angle=0.0, is_star=False)
    hot = cache.get(rock, seed=1, px=14, lit_angle=0.0, is_star=False,
                    temperature=2500.0)
    assert pygame.image.tostring(cold, "RGBA") != pygame.image.tostring(hot, "RGBA")


def test_temperature_is_bucketed_so_a_cooling_body_does_not_rerender_every_frame(cache):
    """A body cooling by a fraction of a kelvin a frame would miss the cache
    on every one of them -- exactly what the cache exists to prevent."""
    from src import materials
    rock = materials.get("rock")
    a = cache.get(rock, seed=1, px=9, lit_angle=0.0, is_star=False, temperature=2000.0)
    b = cache.get(rock, seed=1, px=9, lit_angle=0.0, is_star=False, temperature=2003.0)
    assert a is b


def test_a_cold_body_renders_exactly_as_it_did_before_temperature_existed(cache):
    """The default has to be a no-op, or every ordinary world changes colour."""
    from src import materials, units
    rock = materials.get("rock")
    a = cache.get(rock, seed=4, px=12, lit_angle=0.0, is_star=False)
    b = cache.get(rock, seed=4, px=12, lit_angle=0.0, is_star=False,
                  temperature=units.CMB_TEMPERATURE_K)
    assert a is b
```

Add `import pytest` and `import pygame` at the top of `tests/test_sprites.py` if they are not already there.

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_sprites.py -q`
Expected: `AttributeError: module 'src.render.palette' has no attribute 'blackbody_rgb'`.

- [ ] **Step 3: Add `blackbody_rgb` to `src/render/palette.py`**

Add `import math` at the top of the file (it currently has no imports), and append:

```python
# --- thermal emission ---------------------------------------------------------

def blackbody_rgb(kelvin):
    """The colour of a blackbody at ``kelvin``, as 0..255 RGB.

    A piecewise fit to the Planck locus -- the same curve a star's colour
    actually follows -- accurate to a few percent over 1000..40000 K, which is
    far tighter than a pixel-art sprite can show.  Below 1000 K a blackbody is
    too dim to have a useful colour at all, so it is clamped there and the
    caller's glow weight is what fades it in.

    This is one of the few colours in the sandbox that is *not* a theme choice:
    an ember is orange because of physics, not because the palette says so.
    """
    t = min(max(float(kelvin), 1000.0), 40000.0) / 100.0
    if t <= 66.0:
        red = 255.0
        green = 99.4708025861 * math.log(t) - 161.1195681661
    else:
        red = 329.698727446 * (t - 60.0) ** -0.1332047592
        green = 288.1221695283 * (t - 60.0) ** -0.0755148492
    if t >= 66.0:
        blue = 255.0
    elif t <= 19.0:
        blue = 0.0
    else:
        blue = 138.5177312231 * math.log(t - 10.0) - 305.0447927307
    return tuple(int(min(max(v, 0.0), 255.0)) for v in (red, green, blue))
```

- [ ] **Step 4: Blend the glow into the sprite**

In `src/render/sprites.py`, add to the constants block at the top:

```python
#: Below this a body's own thermal emission is far too faint to see against the
#: light it reflects; by GLOW_FULL_K there is nothing else left to see.  Real
#: numbers: iron starts to visibly glow around 800 K and is white by 3000 K.
GLOW_MIN_K = 800.0
GLOW_FULL_K = 3000.0
#: Temperature is quantised to this many kelvin before it reaches the cache
#: key.  A body cooling continuously would otherwise miss the cache on every
#: frame, which is precisely what the cache exists to prevent.
GLOW_TEMP_STEP = 100.0
```

Add, above `_make_body`:

```python
def glow_fraction(kelvin):
    """How far a body has gone from its own colours to its blackbody glow, 0..1."""
    if kelvin <= GLOW_MIN_K:
        return 0.0
    return min(1.0, (kelvin - GLOW_MIN_K) / (GLOW_FULL_K - GLOW_MIN_K))


def glow_key(kelvin):
    """The quantised temperature the cache is keyed on.  Zero means no glow,
    so every ordinary body shares one key however its temperature drifts."""
    if glow_fraction(kelvin) <= 0.0:
        return 0
    return int(round(float(kelvin) / GLOW_TEMP_STEP))


def _blend_toward(tone, target, weight):
    return tuple(int(round(tone[k] + (target[k] - tone[k]) * weight))
                 for k in range(3))
```

Change `_make_body`'s signature and its opening, so it reads:

```python
def _make_body(material, seed, px, lit_angle, is_star, glow_k=0.0):
    px = max(1, int(px))
    dark, mid, light = material.tones

    if glow_k > 0.0:
        # Its own light, mixed into the light it reflects.  Applied before the
        # star branch as well as after it, so a red dwarf really is red and a
        # blue giant really is blue -- the same Planck locus, the same helper.
        target = palette.blackbody_rgb(glow_k)
        weight = glow_fraction(glow_k)
        dark = _blend_toward(dark, target, weight)
        mid = _blend_toward(mid, target, weight)
        light = _blend_toward(light, target, weight)
```

and leave the rest of the function exactly as it is.

Add `from . import palette` to the imports at the top of `src/render/sprites.py` if it is not already there. `palette` imports only `math`, so this cannot cycle.

Change `SpriteCache.get` to:

```python
    def get(self, material, seed, px, lit_angle, is_star,
            temperature=units.CMB_TEMPERATURE_K):
        px = max(1, int(px))
        bucket = int(round(lit_angle / (2 * math.pi) * LIT_BUCKETS)) % LIT_BUCKETS
        heat = glow_key(temperature)
        key = (material.key, seed, px, bucket, bool(is_star), heat)
        hit = self._entries.get(key)
        if hit is not None:
            self._entries.move_to_end(key)
            return hit
        angle = bucket * 2 * math.pi / LIT_BUCKETS
        surface = _make_body(material, seed, px, angle, is_star,
                             heat * GLOW_TEMP_STEP)
        self._entries[key] = surface
        if len(self._entries) > self.limit:
            self._entries.popitem(last=False)
        return surface
```

- [ ] **Step 5: Pass the temperature from `draw.py`**

In `src/render/draw.py`, change the `self.cache.get(...)` call to:

```python
                sprite = self.cache.get(body.material, body.seed, int(px),
                                        nearest_star_angle(world, i), is_star,
                                        float(world.temperature[i]))
```

- [ ] **Step 6: Run to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_sprites.py tests/test_draw.py tests/test_black_hole_render.py tests/test_smoke.py -q`
Expected: all pass. If a pre-existing star-colour assertion now fails, that is the star tint working -- check the expected values against `blackbody_rgb` for that star's mass before changing anything, and report it rather than silently loosening the test.

- [ ] **Step 7: Commit**

```bash
git add src/render/palette.py src/render/sprites.py src/render/draw.py tests/test_sprites.py
git commit -m "feat: hot bodies glow their own blackbody colour"
```

---

### Task 11: Document the invariants and run the whole suite

**Files:**
- Modify: `CLAUDE.md` (the layer table, the Invariants list, the Design decisions list, the test count in Commands)

**Interfaces:**
- Consumes: everything.
- Produces: nothing code depends on.

- [ ] **Step 1: Run the whole suite and record the count**

Run: `./.venv/bin/python selftest.py`
Expected: every test passes. Note the total count printed -- the Commands block in `CLAUDE.md` quotes it.

- [ ] **Step 2: Update the layer table**

In `CLAUDE.md`, in the `src/*.py` row of the three-layer table, add `thermal.py` to the list of what it includes, in the same style as the existing mention of `collisions.py` and `destruction.py`:

```
Includes `collisions.py` (detection and outcome resolution), `destruction.py` (tidal
and thermal destruction near a star or a black hole), both emitting the plain-data
`ImpactEvent`, and `thermal.py` (every body's temperature).
```

- [ ] **Step 3: Add the invariants**

Add these three bullets to the **Invariants** list in `CLAUDE.md`:

```markdown
- **Temperature is state, not a derived quantity.** It lives in `World.temperature`
  because it changes every step and must survive being edited; `Body.spin` and
  `Body.disk` are on the Body because they are static appearance. Every path that
  creates a body -- `World.add`, every collision outcome, every tidal fragment --
  must give it one, or it starts at absolute zero, which is colder than the
  universe allows and produces a warm-up nobody asked for.
- **`thermal.pinned_temperature` decides who is free.** A star's temperature is
  `units.star_temperature(mass)` and a black hole's is its Hawking temperature;
  neither relaxes, neither is editable, and `edits.set_temperature` refuses both
  even though the panel already omits the row -- the panel is not the only caller.
  `edits._refresh_kind` and `edits.collapse` write the pinned value the moment a
  body changes kind, so the array is never a frame behind what the body is.
- **`thermal.advance` runs once a frame, never per substep.** The same rule
  `update_trails` follows, and for the same reason: the pass is O(n^2), and at ten
  years a second the integrator takes thousands of substeps per frame. Ablation
  inside the substep loop therefore reads a temperature that is up to one frame
  old, which is harmless because the analytic step has already snapped a body to
  equilibrium at any time scale where a frame is long.
```

- [ ] **Step 4: Add the design decisions**

Add these three bullets to the **Design decisions worth knowing before changing them** list:

```markdown
- **A body is heated by every other body, not by stars** (`thermal.py`). One sum:
  `T_eq^4 = (1-A)/4 * sum_j T_j^4 (R_j/d_ij)^2 + T_CMB^4`. The Sun heating Mercury
  and a molten Earth heating the Moon are the same term, which is why "a hot planet
  warms its neighbours" needed no code of its own, and why a black hole needs no
  exception -- 6e-8 K over a horizon a few km wide evaluates to nothing. Fluxes add;
  temperatures do not. There is deliberately **no greenhouse term**: it would be a
  tuned per-material constant rather than derived physics, so Earth reads -19 C,
  which is true.
- **Temperature relaxes on a one-metre skin, integrated analytically**
  (`units.THERMAL_SKIN_DEPTH_M`). Heating a whole planet gives a time constant near
  a hundred thousand years -- real, and invisible; real surfaces equilibrate in days
  because only a thin skin follows the sunlight. `tau = rho h c_p / (4 sigma T^3)`
  linearised at the *hotter* of the body and its target, so a body dropped to 3 K
  beside the Sun warms up instead of staying frozen. The step is
  `T <- T_eq + (T-T_eq) exp(-dt/tau)`, which is unconditionally stable; an explicit
  step would need `dt < tau`, and tau is under an hour for anything glowing.
- **A hot body's colour comes from the Planck locus, not the palette**
  (`render/palette.blackbody_rgb`). Temperature is quantised to 100 K before it
  reaches the sprite cache key, for the same reason the lighting angle is bucketed
  to 16 points: a body cooling continuously would miss the cache every frame. The
  blend runs before the star branch too, so a red dwarf is red and a blue giant is
  blue off the same helper.
```

- [ ] **Step 5: Update the test count**

In the Commands block, change `python selftest.py                      # whole suite (671 tests, ~3:40)` to the count recorded in Step 1.

- [ ] **Step 6: Run the whole suite once more and commit**

```bash
./.venv/bin/python selftest.py
git add CLAUDE.md
git commit -m "docs: temperature invariants and design decisions in CLAUDE.md"
```

---

## Self-Review

**Spec coverage.** Equilibrium sum -- Task 4. Pinned stars and holes -- Task 4. CMB floor -- Tasks 1, 4. Skin-depth relaxation and the analytic step -- Tasks 1, 5. Once-a-frame stepping -- Task 7. `settle` -- Tasks 5, 7. Albedo and heat capacity -- Task 2. Impact heating and every collision outcome -- Task 6. Ablation reading stored temperature -- Task 7. `world.temperature` plumbing -- Task 3. Edit row, star field list, clamping, ratio scrub -- Task 8. The `EDIT_FIELDS` bug -- Task 8. Inspector row -- Task 9. `blackbody_rgb` and the sprite tint -- Task 10. `test_layering` gaining `src.thermal` and `src.destruction` -- Task 4. Documentation -- Task 11.

**Types.** `pinned_temperature` returns `float | None` and is called that way in `radiator_temperatures`, `advance` and `settle`. `equilibrium_temperatures` returns a length-`world.n` array and is indexed as one in `equilibrium_temperature`, `advance` and `settle`. `impact_heating(material, specific_energy)` takes simulation units everywhere it is called (`merge`, `graze`, `shatter`) and is never handed J/kg. `World.add`'s `temperature` keyword is passed positionally nowhere. `SpriteCache.get`'s new argument is keyword-with-default at every existing call site and positional only in `draw.py`, which passes it last.

**Known follow-ups for the verification pass, not for the implementer:** whether a feeding black hole should light its surroundings (out of scope by decision, `Body.disk` is not an accretion rate); whether `scenarios.py` should seed temperatures itself rather than leaning on `Simulation.__init__` to settle them (it should not -- one owner is better); and whether the 1 m skin depth wants to vary by material (it does not until someone asks, and `Material.heat_capacity_j_kgk` already carries most of the variation).
