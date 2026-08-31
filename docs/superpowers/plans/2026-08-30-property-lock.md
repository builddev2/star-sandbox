# Property Lock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Any editable field on a body -- mass, density, temperature, material, spin, disk -- can be pinned with a padlock, and a pinned field then ignores every attempt to change it until it is explicitly freed.

**Architecture:** A `frozenset` of field names on `Body`, next to `spin` and `disk`, holds the locks for that body. Enforcement is a one-line guard at the top of each setter in `src/edits.py`, so no caller can route around it. The Edit panel grows a padlock button on every field row plus an "Unlock all" control beside `Collapse`, and two hotkeys (`K`, `U`) that call exactly the same tool methods those buttons do.

**Tech Stack:** Python 3.11, numpy, pygame 2.6.1. Tests with pytest.

**Spec:** `docs/superpowers/specs/2026-08-30-property-lock-design.md`

## Global Constraints

- **A lock blocks only *direct* edits to its own field.** Locking density does not stop a
  material swap from moving density, because `set_material` is editing material, not
  density. This was decided during design; do not invent cross-field bookkeeping.
- **Enforcement lives in `src/edits.py`, not in the panel.** The greyed-out button is a
  second, cosmetic line of defence. `edits.set_temperature` already refuses a star "even
  though the panel already omits the row -- the panel is not the only caller"; a lock gets
  the same treatment.
- **A hotkey does not inherit a control's `enabled` gate.** Keyboard events bypass
  `UiLayer.perform`'s `clickable` check, so a key path must duplicate any precondition its
  button carries -- the way `EditTool._collapse` re-checks `body.kind == KIND_STAR` for `C`.
- **Every action has a button, and every button carries its hotkey.** A control never
  reimplements a key: it names the same tool method the key calls. `tests/test_ui.py` reads
  the shortcut list out of `hud.KEY_HELP` and fails if any documented key has no button.
- **`src/*.py` must never import pygame.** `tests/test_layering.py` enforces it in a
  subprocess. Nothing in this feature gives `edits.py` or `body.py` a new import at all.
- **Pixel sizes come from `widgets.Metrics`, never hard-coded.** The `G` grid switch changes
  the font, and every size with it. Icons are drawn from lines, never typed as glyphs.
- **Nothing here needs `Simulation.refresh()`.** A lock changes no mass, no radius and no
  membership. Do not add a refresh call to the lock paths; it would be cargo cult.
- Run the tests with `python3` from the worktree root -- this worktree has no `.venv/` of
  its own, and the system Python has pygame 2.6.1 and numpy. Whole suite:
  `python3 selftest.py` (~3:40). One file: `python3 -m pytest tests/test_edits.py -q`.
- Comments and docstrings in this codebase explain **why**, not what. Match that. British
  spelling in prose (`colour`, `grey`), en-dash written as `--`.

---

### Task 1: `locked_fields` on `Body`, and the three lock helpers

**Files:**
- Modify: `src/body.py:72-73` (the `__slots__` tuple), `src/body.py:88-95` (the tail of
  `Body.__init__`)
- Modify: `src/edits.py` (append a new `# --- locking ---` section at the end of the file)
- Test: `tests/test_edits.py` (append; and widen the import on line 5)

**Interfaces:**
- Consumes: nothing.
- Produces: `Body.locked_fields: frozenset[str]`, `edits.is_locked(body, field) -> bool`,
  `edits.toggle_lock(body, field) -> None`, `edits.unlock_all(body) -> None`. All three
  take a **`Body`**, not a `(world, index)` pair -- the same shape `edits.edit_fields(body)`
  already has. Field names are exactly the strings `edits.edit_fields` returns: `"mass"`,
  `"density"`, `"temp"`, `"material"`, `"spin"`, `"disk"`.

- [ ] **Step 1: Widen the import at the top of `tests/test_edits.py`**

Change line 5 from:

```python
from src.body import KIND_PLANET, KIND_STAR
```

to:

```python
from src.body import KIND_BLACK_HOLE, KIND_PLANET, KIND_STAR
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_edits.py`:

```python
# --- locking ----------------------------------------------------------------------

def test_every_body_starts_with_nothing_locked(sim):
    for body in sim.world.bodies:
        assert body.locked_fields == frozenset()


def test_the_lock_set_is_a_frozenset(sim):
    """Rebinding is the only way to change it, so two bodies can never end up
    sharing one mutable set."""
    body = sim.world.bodies[_index(sim, "Earth")]
    assert isinstance(body.locked_fields, frozenset)


def test_a_body_added_to_the_world_starts_unlocked(sim):
    """Every fragment path -- shatter, graze, cratering ejecta, tidal debris --
    goes through World.add, so this is also what says debris never inherits its
    parent's locks."""
    body = sim.world.add("Pebble", materials.get("rock"), 1e-9, 1e-6,
                         (5.0, 0.0), (0.0, 1.0))
    assert body.locked_fields == frozenset()


def test_toggle_lock_flips_one_field_and_leaves_the_others(sim):
    body = sim.world.bodies[_index(sim, "Earth")]
    assert not edits.is_locked(body, "mass")
    edits.toggle_lock(body, "mass")
    assert edits.is_locked(body, "mass")
    assert not edits.is_locked(body, "density")
    edits.toggle_lock(body, "mass")
    assert not edits.is_locked(body, "mass")


def test_unlock_all_empties_the_set(sim):
    body = sim.world.bodies[_index(sim, "Earth")]
    for field in ("mass", "density", "temp", "material"):
        edits.toggle_lock(body, field)
    edits.unlock_all(body)
    assert body.locked_fields == frozenset()


def test_unlock_all_with_nothing_locked_is_harmless(sim):
    body = sim.world.bodies[_index(sim, "Earth")]
    edits.unlock_all(body)
    assert body.locked_fields == frozenset()


def test_a_lock_survives_the_body_changing_kind(sim):
    """A star that cools back into a planet finds its old temperature lock
    still in place.  A kind change is not a reset -- while the body is a star
    the lock is simply inert, because the panel offers no Temp row for one."""
    i = _index(sim, "Earth")
    body = sim.world.bodies[i]
    edits.toggle_lock(body, "temp")
    edits.set_mass(sim.world, i, units.STAR_MASS_THRESHOLD * 1.1)
    assert body.kind == KIND_STAR
    edits.set_mass(sim.world, i, units.EARTH_MASS)
    assert body.kind == KIND_PLANET
    assert edits.is_locked(body, "temp")
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_edits.py -q -k lock`
Expected: failures reading `AttributeError: 'Body' object has no attribute 'locked_fields'`
and `AttributeError: module 'src.edits' has no attribute 'is_locked'`.

- [ ] **Step 4: Add the slot to `Body`**

In `src/body.py`, replace the `__slots__` tuple:

```python
    __slots__ = ("id", "name", "kind", "material", "trail", "born_day",
                 "event_id", "no_collide_until", "seed", "rings", "spin", "disk")
```

with:

```python
    __slots__ = ("id", "name", "kind", "material", "trail", "born_day",
                 "event_id", "no_collide_until", "seed", "rings", "spin", "disk",
                 "locked_fields")
```

- [ ] **Step 5: Default it in `Body.__init__`**

In `src/body.py`, immediately after the two lines that set `self.disk = 0.0` (the last
statement in `__init__`), add:

```python
        #: Fields the child has pinned, named the way ``edits.edit_fields``
        #: names them.  On the Body rather than in World's arrays for the same
        #: reason ``spin`` and ``disk`` are: the O(n^2) force kernel never reads
        #: it.  A frozenset because rebinding is then the only way to change it,
        #: so two bodies can never end up sharing one mutable set -- and nothing
        #: inherits it: a fresh body, a collision fragment and a tidal fragment
        #: all arrive with nothing locked.
        self.locked_fields = frozenset()
```

- [ ] **Step 6: Add the helpers to `src/edits.py`**

Append to the end of `src/edits.py`:

```python


# --- locking ----------------------------------------------------------------------
#
# A lock is a standing preference for *this* body rather than a session flag,
# which is why it lives on the Body and not in this module: select away and back,
# or leave the Edit tool and return, and it is still there.
#
# It guards its **own** field and nothing else.  Locking density does not stop a
# material swap from moving density, because ``set_material`` is editing
# material, not density.  Chasing a value through every other field that touches
# it is bookkeeping nobody asked for, and it would make "locked" mean something
# slightly different in every row.


def is_locked(body, field):
    """True if ``field`` on this body refuses to be edited."""
    return field in body.locked_fields


def toggle_lock(body, field):
    """Lock ``field`` if it is free; free it if it is locked."""
    if field in body.locked_fields:
        body.locked_fields = body.locked_fields - {field}
    else:
        body.locked_fields = body.locked_fields | {field}


def unlock_all(body):
    """Free every field at once -- one press to undo a panel full of padlocks."""
    body.locked_fields = frozenset()
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_edits.py -q`
Expected: every test passes, including the seven new ones.

- [ ] **Step 8: Commit**

```bash
git add src/body.py src/edits.py tests/test_edits.py
git commit -m "feat: per-body locked_fields and the lock helpers in edits"
```

---

### Task 2: Every setter refuses its own locked field

**Files:**
- Modify: `src/edits.py:139` (`set_mass`), `src/edits.py:163` (`set_temperature`),
  `src/edits.py:181` (`set_spin`), `src/edits.py:185` (`set_disk`),
  `src/edits.py:213` (`set_density`), `src/edits.py:224` (`set_material`),
  `src/edits.py:231` (`cycle_material`)
- Test: `tests/test_edits.py` (append to the `# --- locking ---` section)

**Interfaces:**
- Consumes: `edits.is_locked` from Task 1.
- Produces: the enforcement every later task leans on. No new names.

Guard table -- each setter checks **only** its own field:

| setter | guarded field |
|---|---|
| `set_mass` | `"mass"` |
| `set_density` | `"density"` |
| `set_temperature` | `"temp"` |
| `set_spin` | `"spin"` |
| `set_disk` | `"disk"` |
| `set_material` | `"material"` |
| `cycle_material` | `"material"` |

`_refresh_kind` and `collapse` are **not** guarded. Collapse conserves mass, so it is not a
mass edit; and the pinned temperature `_refresh_kind` writes when a body ignites is not an
edit either -- it is what the body's own physics says its temperature *is*.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_edits.py`:

```python
def _black_hole(sim):
    """Ten solar masses of Earth, crushed.  Spin and disk take their defaults,
    0.7 and 0.6, so a test can set something else and see it."""
    i = _index(sim, "Earth")
    edits.set_mass(sim.world, i, 10.0)
    edits.collapse(sim.world, i)
    return i


def test_a_locked_mass_refuses_to_change(sim):
    i = _index(sim, "Earth")
    body = sim.world.bodies[i]
    mass = float(sim.world.mass[i])
    radius = float(sim.world.radius[i])
    edits.toggle_lock(body, "mass")
    edits.set_mass(sim.world, i, mass * 10.0)
    assert float(sim.world.mass[i]) == pytest.approx(mass)
    assert float(sim.world.radius[i]) == pytest.approx(radius)
    edits.toggle_lock(body, "mass")
    edits.set_mass(sim.world, i, mass * 10.0)
    assert float(sim.world.mass[i]) == pytest.approx(mass * 10.0)


def test_a_locked_mass_cannot_be_ignited(sim):
    """The guard is at the top of the setter, so ``_refresh_kind`` is never
    reached and the body does not quietly change kind on the way out."""
    i = _index(sim, "Jupiter")
    body = sim.world.bodies[i]
    edits.toggle_lock(body, "mass")
    edits.set_mass(sim.world, i, units.STAR_MASS_THRESHOLD * 2.0)
    assert body.kind == KIND_PLANET


def test_a_locked_density_refuses_to_change(sim):
    # Saturn: 0.687 g/cm3, so four times that is still inside the material range
    # and the clamp cannot be what holds the value still.
    i = _index(sim, "Saturn")
    body = sim.world.bodies[i]
    density = edits.body_density(sim.world, i)
    radius = float(sim.world.radius[i])
    material = body.material.key
    edits.toggle_lock(body, "density")
    edits.set_density(sim.world, i, density * 4.0)
    assert edits.body_density(sim.world, i) == pytest.approx(density, rel=1e-9)
    assert float(sim.world.radius[i]) == pytest.approx(radius)
    # set_density also relabels the composition; a refused edit relabels nothing.
    assert body.material.key == material
    edits.toggle_lock(body, "density")
    edits.set_density(sim.world, i, density * 4.0)
    assert edits.body_density(sim.world, i) == pytest.approx(density * 4.0, rel=1e-9)


def test_a_locked_temperature_refuses_to_change(sim):
    i = _index(sim, "Earth")
    body = sim.world.bodies[i]
    before = float(sim.world.temperature[i])
    edits.toggle_lock(body, "temp")
    edits.set_temperature(sim.world, i, 900.0)
    assert float(sim.world.temperature[i]) == pytest.approx(before)
    edits.toggle_lock(body, "temp")
    edits.set_temperature(sim.world, i, 900.0)
    assert float(sim.world.temperature[i]) == pytest.approx(900.0)


def test_a_locked_spin_refuses_to_change(sim):
    i = _black_hole(sim)
    body = sim.world.bodies[i]
    before = body.spin
    edits.toggle_lock(body, "spin")
    edits.set_spin(sim.world, i, 0.1)
    assert body.spin == pytest.approx(before)
    edits.toggle_lock(body, "spin")
    edits.set_spin(sim.world, i, 0.1)
    assert body.spin == pytest.approx(0.1)


def test_a_locked_disk_refuses_to_change(sim):
    i = _black_hole(sim)
    body = sim.world.bodies[i]
    before = body.disk
    edits.toggle_lock(body, "disk")
    edits.set_disk(sim.world, i, 0.2)
    assert body.disk == pytest.approx(before)
    edits.toggle_lock(body, "disk")
    edits.set_disk(sim.world, i, 0.2)
    assert body.disk == pytest.approx(0.2)


def test_a_locked_material_refuses_to_change(sim):
    i = _index(sim, "Earth")
    body = sim.world.bodies[i]
    key = body.material.key
    radius = float(sim.world.radius[i])
    edits.toggle_lock(body, "material")
    edits.set_material(sim.world, i, materials.get("iron"))
    assert body.material.key == key
    assert float(sim.world.radius[i]) == pytest.approx(radius)
    edits.cycle_material(sim.world, i, 1)
    assert body.material.key == key
    edits.toggle_lock(body, "material")
    edits.cycle_material(sim.world, i, 1)
    assert body.material.key != key


def test_a_lock_guards_only_its_own_field(sim):
    """The worked example the design is built around: pin the mass, then scrub
    density, temperature and composition freely."""
    i = _index(sim, "Earth")
    body = sim.world.bodies[i]
    mass = float(sim.world.mass[i])
    edits.toggle_lock(body, "mass")
    edits.set_density(sim.world, i, edits.body_density(sim.world, i) * 1.5)
    edits.set_temperature(sim.world, i, 500.0)
    edits.set_material(sim.world, i, materials.get("iron"))
    assert float(sim.world.mass[i]) == pytest.approx(mass)
    assert body.material.key == "iron"
    assert float(sim.world.temperature[i]) == pytest.approx(500.0)


def test_a_density_lock_does_not_chase_density_through_a_material_swap(sim):
    """Decided during design: a lock blocks direct edits to its own field and
    nothing else.  set_material is editing material, not density."""
    i = _index(sim, "Earth")
    body = sim.world.bodies[i]
    edits.toggle_lock(body, "density")
    edits.set_material(sim.world, i, materials.get("iron"))
    assert edits.body_density(sim.world, i) == pytest.approx(
        materials.get("iron").density, rel=1e-9)


def test_collapse_is_not_stopped_by_a_locked_mass(sim):
    """Collapse conserves mass, so it is not a mass edit -- and a lock reaches
    no further than the setters it guards."""
    i = _index(sim, "Earth")
    body = sim.world.bodies[i]
    edits.set_mass(sim.world, i, 10.0)
    edits.toggle_lock(body, "mass")
    edits.collapse(sim.world, i)
    assert body.kind == KIND_BLACK_HOLE
    assert float(sim.world.mass[i]) == pytest.approx(10.0)


def test_a_star_still_takes_its_pinned_temperature_with_temp_locked(sim):
    """The pin is not an edit: ``_refresh_kind`` writes the array directly, so a
    body that ignites reads as a star at once, whatever its lock says."""
    i = _index(sim, "Earth")
    body = sim.world.bodies[i]
    edits.toggle_lock(body, "temp")
    edits.set_mass(sim.world, i, units.STAR_MASS_THRESHOLD * 1.1)
    assert float(sim.world.temperature[i]) == pytest.approx(
        units.star_temperature(float(sim.world.mass[i])), rel=1e-9)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_edits.py -q -k lock`
Expected: the twelve new tests fail (values change when they should not); the Task 1 tests
still pass.

- [ ] **Step 3: Guard `set_mass`**

In `src/edits.py`, inside `set_mass`, replace:

```python
    body = world.bodies[index]
    mass = float(min(max(float(mass), MIN_MASS), MAX_MASS))
```

with:

```python
    body = world.bodies[index]
    if is_locked(body, "mass"):
        return
    mass = float(min(max(float(mass), MIN_MASS), MAX_MASS))
```

- [ ] **Step 4: Guard `set_temperature`**

Replace:

```python
    if world.bodies[index].kind in (KIND_STAR, KIND_BLACK_HOLE):
        return
```

with:

```python
    body = world.bodies[index]
    if body.kind in (KIND_STAR, KIND_BLACK_HOLE) or is_locked(body, "temp"):
        return
```

Also extend that function's docstring: after the existing paragraph beginning "Refuses on a
star or a black hole", add:

```
    Refuses a locked temperature for the same reason, and in the same place:
    the panel greys the row's buttons out, and this is what makes that mean
    something to every other caller.
```

- [ ] **Step 5: Guard `set_spin` and `set_disk`**

Replace:

```python
def set_spin(world, index, spin):
    world.bodies[index].spin = float(min(max(float(spin), 0.0), units.MAX_SPIN))


def set_disk(world, index, disk):
    world.bodies[index].disk = float(min(max(float(disk), 0.0), 1.0))
```

with:

```python
def set_spin(world, index, spin):
    body = world.bodies[index]
    if is_locked(body, "spin"):
        return
    body.spin = float(min(max(float(spin), 0.0), units.MAX_SPIN))


def set_disk(world, index, disk):
    body = world.bodies[index]
    if is_locked(body, "disk"):
        return
    body.disk = float(min(max(float(disk), 0.0), 1.0))
```

- [ ] **Step 6: Guard `set_density`**

Replace:

```python
def set_density(world, index, density):
    """Set density, holding mass constant, so the radius follows."""
    lo = units.from_gcm3(materials.MIN_DENSITY_GCM3)
```

with:

```python
def set_density(world, index, density):
    """Set density, holding mass constant, so the radius follows."""
    if is_locked(world.bodies[index], "density"):
        return
    lo = units.from_gcm3(materials.MIN_DENSITY_GCM3)
```

- [ ] **Step 7: Guard `set_material` and `cycle_material`**

Replace:

```python
def set_material(world, index, material):
    """Adopt a substance, snapping density to that substance's real value."""
    world.bodies[index].material = material
    world.radius[index] = units.radius_from_mass_density(
        float(world.mass[index]), material.density)


def cycle_material(world, index, step=1):
    body = world.bodies[index]
    if body.kind == KIND_BLACK_HOLE:
        return          # a hole is not made of anything to cycle through
```

with:

```python
def set_material(world, index, material):
    """Adopt a substance, snapping density to that substance's real value."""
    body = world.bodies[index]
    if is_locked(body, "material"):
        return
    body.material = material
    world.radius[index] = units.radius_from_mass_density(
        float(world.mass[index]), material.density)


def cycle_material(world, index, step=1):
    body = world.bodies[index]
    if body.kind == KIND_BLACK_HOLE:
        return          # a hole is not made of anything to cycle through
    if is_locked(body, "material"):
        return
```

- [ ] **Step 8: Note the guard in the module docstring**

In `src/edits.py`, add this paragraph to the module docstring, immediately before the last
line ("Imports numpy and nothing from pygame..."):

```
Any field may be **locked**, per body: a locked field's setter returns without
writing anything.  The check is here rather than in the panel because the panel
is not the only caller -- the same argument ``set_temperature`` already makes
about stars.
```

- [ ] **Step 9: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_edits.py -q`
Expected: all pass.

- [ ] **Step 10: Check nothing else in the physics went through these setters**

Run: `python3 -m pytest tests/test_collision_merge.py tests/test_collision_shatter.py tests/test_destruction.py tests/test_thermal.py tests/test_sim.py -q`
Expected: all pass. (`collisions.py` calls only `edits.collapse`, which is deliberately
unguarded, so this should be uneventful -- but a lock that broke a merge would be the worst
kind of bug to find later.)

- [ ] **Step 11: Commit**

```bash
git add src/edits.py tests/test_edits.py
git commit -m "feat: every editor setter refuses its own locked field"
```

---

### Task 3: The lock and unlock icons

**Files:**
- Modify: `src/render/widgets.py:483-485` (insert a new branch in `draw_icon` after the
  `"close"` branch)
- Test: `tests/test_widgets.py:98-101` (the `ICON_NAMES` tuple), and append one test

**Interfaces:**
- Consumes: nothing.
- Produces: `draw_icon(surface, "lock", rect, colour)` and
  `draw_icon(surface, "unlock", rect, colour)`.

Both are drawn from lines only -- **no `pygame.draw.arc`**. An arc inside a 4x5 box renders
nothing at all on some platforms, and the compact buttons these sit on are 11 pixels square
at the coarsest grid.

- [ ] **Step 1: Add the two names to the icon parametrisation**

In `tests/test_widgets.py`, replace:

```python
ICON_NAMES = ("look", "throw", "edit", "play", "pause", "slower", "faster", "reset",
              "trails", "sizes", "labels", "grid", "fullscreen", "follow", "flyto",
              "stop", "trash", "plus", "minus", "left", "right", "up", "down",
              "list", "close", "circle", "star", "blackhole", "help", "home", "power")
```

with:

```python
ICON_NAMES = ("look", "throw", "edit", "play", "pause", "slower", "faster", "reset",
              "trails", "sizes", "labels", "grid", "fullscreen", "follow", "flyto",
              "stop", "trash", "plus", "minus", "left", "right", "up", "down",
              "list", "close", "circle", "star", "blackhole", "help", "home", "power",
              "lock", "unlock")
```

- [ ] **Step 2: Write the failing test**

Append to `tests/test_widgets.py`:

```python
def test_the_lock_and_unlock_icons_read_as_a_pair_at_button_size():
    """These two are the whole readout on the padlock button, and they have to
    work at the size the +/- glyphs already draw at -- eight pixels, at the
    coarsest grid in the smallest window.  Which is also why neither is drawn
    with an arc: pygame renders nothing at all for an arc in a box this small.
    """
    from src.render import widgets
    drawn = {}
    for name in ("lock", "unlock"):
        surface = pygame.Surface((24, 24))
        surface.fill((0, 0, 0))
        rect = pygame.Rect(8, 8, 8, 8)
        widgets.draw_icon(surface, name, rect, (255, 255, 255))
        drawn[name] = {(x, y) for x in range(24) for y in range(24)
                       if surface.get_at((x, y))[:3] != (0, 0, 0)}
        assert len(drawn[name]) >= 8, "%s is invisible at eight pixels" % name
        for point in drawn[name]:
            assert rect.inflate(2, 2).collidepoint(point), \
                "%s escaped its box" % name
    assert drawn["lock"] != drawn["unlock"], \
        "locked and unlocked draw the same thing"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_widgets.py -q`
Expected: three failures -- the two parametrised `ICON_NAMES` cases ("lock drew nothing",
"unlock drew nothing") and the new pair test.

- [ ] **Step 4: Draw them**

In `src/render/widgets.py`, inside `draw_icon`, insert this branch immediately after the
`"close"` branch (the two crossing lines) and before `elif name == "circle":`:

```python
    elif name in ("lock", "unlock"):
        # A padlock: a body with a keyhole, and a shackle above it.  Drawn from
        # straight lines rather than an arc because an arc inside a box this
        # small renders nothing at all, and this button is eleven pixels square
        # at the coarsest grid.  The two states differ by where the shackle
        # sits -- seated in the body, or lifted and slid off to one side -- so
        # they read as one object in two states rather than two glyphs.
        body_box = pygame.Rect(0, 0, max(5, 2 * r - 1), max(4, r + 1))
        body_box.centerx = cx
        body_box.bottom = min(rect.bottom - 1, cy + r)
        pygame.draw.rect(surface, colour, body_box, 1)
        surface.set_at((body_box.centerx, body_box.centery), colour)
        top = max(y, body_box.top - max(2, r - 1))
        left = body_box.left + 2
        right = max(left + 2, body_box.right - 3)
        if name == "lock":
            pygame.draw.line(surface, colour, (left, top), (right, top), 1)
            pygame.draw.line(surface, colour, (left, top), (left, body_box.top), 1)
            pygame.draw.line(surface, colour, (right, top), (right, body_box.top), 1)
        else:
            shift = max(1, r // 2)
            lifted, hinge = left + shift, min(rect.right - 1, right + shift)
            pygame.draw.line(surface, colour, (lifted, top), (hinge, top), 1)
            pygame.draw.line(surface, colour, (hinge, top), (hinge, body_box.top), 1)
            pygame.draw.line(surface, colour, (lifted, top),
                             (lifted, top + max(1, (body_box.top - top) // 2)), 1)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_widgets.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/render/widgets.py tests/test_widgets.py
git commit -m "feat: lock and unlock icons, drawn from lines"
```

---

### Task 4: The padlock on every row, and Unlock all

**Files:**
- Modify: `src/app/tools.py:538-573` (`EditTool.controls`), `src/app/tools.py:575-588`
  (`EditTool.perform`), and append two private methods after `EditTool._collapse`
- Test: `tests/test_tools_edit.py` (append)

**Interfaces:**
- Consumes: `edits.is_locked`, `edits.toggle_lock`, `edits.unlock_all` from Task 1; the
  `"lock"` / `"unlock"` icons from Task 3.
- Produces: two new controls, and the two tool methods the hotkeys in Task 5 will call.
  - `Control("edit.lock", icon="lock"|"unlock", payload=<field name>, compact=True,
    hotkey="K", active=<locked>, enabled=True)` -- one per field row, in that row, after the
    `+` stepper.
  - `Control("edit.unlock_all", label="Unlock all", icon="unlock", hotkey="U",
    enabled=bool(body.locked_fields))` -- **sharing the `Collapse` row** for anything that is
    not a black hole, and alone on a row of its own for a black hole.
  - `EditTool._toggle_lock(state, field=None) -> bool` -- `field=None` means "whichever row
    is selected", which is what the `K` key passes.
  - `EditTool._unlock_all(state) -> bool`.

**Why `Unlock all` shares the Collapse row.** Measured at the interface floor
(`config.min_internal(3)` = 213x196, the coarsest grid in the smallest window): the Edit
panel's controls currently reach y=160 against a transport bar that starts at y=171. A row
is 13 pixels. A seventh row -- four fields, Collapse, Unlock all, status -- reaches 173 and
draws over the Play button. Sharing the row keeps a planet at six rows. The row costs no
width either: `Collapse` and `Unlock all` side by side measure 103 px against a panel that
the Material row already makes 115 px wide (183 against 238 at the finest grid). Task 7 adds
the test that keeps this true.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tools_edit.py`:

```python
# --- locking ----------------------------------------------------------------------

def _locks(rows):
    return [c for row in rows for c in row if c.id == "edit.lock"]


def _steps(rows, field):
    return [c for row in rows for c in row
            if c.id == "edit.step" and c.payload[0] == field]


def test_every_field_row_carries_a_lock_button(state):
    from src import edits
    fields = edits.edit_fields(state.sim.world.bodies[state.selected_index])
    locks = _locks(state.tool.controls(state, _metrics()))
    assert [c.payload for c in locks] == list(fields)


def test_the_lock_button_swaps_its_icon(state):
    """The same icon-swap-on-state the play/pause button already uses: the
    button is the readout."""
    from src import edits
    body = state.sim.world.bodies[state.selected_index]
    lock = next(c for c in _locks(state.tool.controls(state, _metrics()))
                if c.payload == "mass")
    assert lock.icon == "unlock"
    edits.toggle_lock(body, "mass")
    lock = next(c for c in _locks(state.tool.controls(state, _metrics()))
                if c.payload == "mass")
    assert lock.icon == "lock"
    assert lock.active


def test_a_locked_row_greys_out_both_its_steppers(state):
    from src import edits
    body = state.sim.world.bodies[state.selected_index]
    edits.toggle_lock(body, "mass")
    rows = state.tool.controls(state, _metrics())
    assert [c.enabled for c in _steps(rows, "mass")] == [False, False]
    assert all(c.enabled for c in _steps(rows, "density"))


def test_the_row_itself_stays_selectable_when_it_is_locked(state):
    """Only the two steppers go inert.  The value still reads out, and the row
    can still be picked."""
    from src import edits
    edits.toggle_lock(state.sim.world.bodies[state.selected_index], "mass")
    rows = state.tool.controls(state, _metrics())
    field = next(c for row in rows for c in row
                 if c.id == "edit.field" and c.payload == "mass")
    assert field.clickable
    assert field.label.startswith("Mass")


def test_the_lock_button_is_never_itself_disabled(state):
    """Otherwise locking a field would be one-way through the mouse."""
    from src import edits
    body = state.sim.world.bodies[state.selected_index]
    for field in edits.edit_fields(body):
        edits.toggle_lock(body, field)
    assert all(c.enabled for c in _locks(state.tool.controls(state, _metrics())))


def test_pressing_a_lock_toggles_it_and_picks_that_row(state):
    """A click on a row's padlock also selects the row, so Up/Down and
    Left/Right carry on from where the mouse left off -- the same thing
    edit.step already does."""
    from src import edits
    body = state.sim.world.bodies[state.selected_index]
    fields = edits.edit_fields(body)
    lock = next(c for c in _locks(state.tool.controls(state, _metrics()))
                if c.payload == fields[-1])
    assert state.tool.perform(lock, state)
    assert edits.is_locked(body, fields[-1])
    assert state.tool.field_index == len(fields) - 1
    assert state.tool.perform(lock, state)
    assert not edits.is_locked(body, fields[-1])


def test_unlock_all_is_disabled_until_something_is_locked(state):
    from src import edits
    rows = state.tool.controls(state, _metrics())
    unlock = next(c for row in rows for c in row if c.id == "edit.unlock_all")
    assert not unlock.enabled
    edits.toggle_lock(state.sim.world.bodies[state.selected_index], "density")
    rows = state.tool.controls(state, _metrics())
    unlock = next(c for row in rows for c in row if c.id == "edit.unlock_all")
    assert unlock.enabled


def test_unlock_all_clears_every_lock(state):
    from src import edits
    body = state.sim.world.bodies[state.selected_index]
    for field in edits.edit_fields(body):
        edits.toggle_lock(body, field)
    rows = state.tool.controls(state, _metrics())
    unlock = next(c for row in rows for c in row if c.id == "edit.unlock_all")
    assert state.tool.perform(unlock, state)
    assert body.locked_fields == frozenset()


def test_unlock_all_shares_the_collapse_row(state):
    """A seventh row would draw over the transport bar at the interface floor;
    tests/test_layout.py is what holds this in place."""
    rows = state.tool.controls(state, _metrics())
    row = next(row for row in rows
               if any(c.id == "edit.unlock_all" for c in row))
    assert {c.id for c in row} == {"edit.collapse", "edit.unlock_all"}


def test_a_black_hole_gets_unlock_all_on_a_row_of_its_own(state):
    """It has no Collapse button to share with."""
    from src import edits
    index = state.selected_index
    edits.set_mass(state.sim.world, index, 10.0)
    edits.collapse(state.sim.world, index)
    rows = state.tool.controls(state, _metrics())
    row = next(row for row in rows
               if any(c.id == "edit.unlock_all" for c in row))
    assert [c.id for c in row] == ["edit.unlock_all"]
    assert [c.payload for c in _locks(rows)] == ["mass", "spin", "disk"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_tools_edit.py -q -k "lock or unlock"`
Expected: failures -- `StopIteration` from the `next(...)` calls, because no `edit.lock` or
`edit.unlock_all` control exists yet.

- [ ] **Step 3: Build the controls**

In `src/app/tools.py`, in `EditTool.controls`, replace the row-building loop and the
Collapse row:

```python
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
```

with:

```python
        for k, field in enumerate(fields):
            step = self.CLICK_NOTCHES[field]
            # Material is a list of substances, not a quantity, so it gets
            # arrows rather than a minus and a plus, and it does not repeat.
            icons = ("left", "right") if field == "material" else ("minus", "plus")
            # Greying the steppers out is the *second* line of defence -- the
            # first is the guard in edits.py, which is what stops the drag and
            # the arrow keys, neither of which comes through a button.
            locked = edits.is_locked(body, field)
            rows.append([
                Control("edit.step", icon=icons[0], payload=(field, -step),
                        compact=True, repeat=field != "material",
                        enabled=not locked, tooltip="Less %s" % field),
                Control("edit.field", label="%s  %s" % (field.title(), values[field]),
                        payload=field, active=k == self.field_index,
                        tooltip="Or pick this row with Up/Down and change it "
                                "with Left/Right"),
                Control("edit.step", icon=icons[1], payload=(field, step),
                        compact=True, repeat=field != "material",
                        enabled=not locked, tooltip="More %s" % field),
                Control("edit.lock", icon="lock" if locked else "unlock",
                        payload=field, compact=True, hotkey="K", active=locked,
                        tooltip=("Unlock %s -- let it be changed again" % field)
                        if locked else ("Lock %s -- pin it where it is" % field)),
            ])
        # Unlock-all shares the Collapse row rather than taking one of its own:
        # at the interface floor a seventh row reaches past the transport bar.
        # A black hole has no Collapse to share with, so it gets the row.
        unlock_all = Control(
            "edit.unlock_all", label="Unlock all", icon="unlock", hotkey="U",
            enabled=bool(body.locked_fields),
            tooltip="Free every locked field on this body")
        if body.kind != KIND_BLACK_HOLE:
            rows.append([
                Control("edit.collapse", label="Collapse", icon="blackhole",
                        hotkey="C", enabled=body.kind == KIND_STAR,
                        tooltip="Crush this star into a black hole -- "
                                "there is no way back"),
                unlock_all,
            ])
        else:
            rows.append([unlock_all])
```

- [ ] **Step 4: Dispatch them**

In `EditTool.perform`, insert two branches immediately before `if control.id ==
"edit.collapse":`:

```python
        if control.id == "edit.lock":
            return self._toggle_lock(state, control.payload)
        if control.id == "edit.unlock_all":
            return self._unlock_all(state)
```

- [ ] **Step 5: Add the two methods**

In `src/app/tools.py`, immediately after `EditTool._collapse` and before `on_event`, add:

```python
    def _toggle_lock(self, state, field=None):
        """Lock or free one field on the selected body.

        ``field`` is the row whose padlock was pressed; the key path passes
        None and means "whichever row is selected", the way Left and Right
        already do.  A click also picks the row, so the arrows carry on from
        where the mouse left off.

        No ``sim.refresh()``: a lock changes no mass, no radius and no
        membership, so nothing the integrator caches has gone stale.
        """
        index = state.selected_index
        if index is None:
            return False
        world = state.sim.world
        fields = self._fields(world, index)
        if field is None:
            field = fields[min(self.field_index, len(fields) - 1)]
        else:
            self.field_index = fields.index(field)
        edits.toggle_lock(world.bodies[index], field)
        return True

    def _unlock_all(self, state):
        """Free every field on the selected body at once.

        Re-checks that there is something to free rather than leaning on the
        button being drawn disabled: a keyboard event never passes through
        ``UiLayer.perform``, so the control's ``enabled`` flag cannot gate it --
        the same rule ``_collapse`` follows for ``C``.
        """
        index = state.selected_index
        if index is None:
            return False
        body = state.sim.world.bodies[index]
        if not body.locked_fields:
            return False
        edits.unlock_all(body)
        return True
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_tools_edit.py -q`
Expected: all pass, old and new.

- [ ] **Step 7: Commit**

```bash
git add src/app/tools.py tests/test_tools_edit.py
git commit -m "feat: a padlock on every Edit row, and an Unlock all button"
```

---

### Task 5: The `K` and `U` hotkeys

**Files:**
- Modify: `src/app/tools.py:607-623` (`EditTool.on_event`, the `KEYDOWN` block),
  `src/app/tools.py:477-478` (`EditTool.hint`)
- Test: `tests/test_tools_edit.py` (append)

**Interfaces:**
- Consumes: `EditTool._toggle_lock` and `EditTool._unlock_all` from Task 4.
- Produces: `pygame.K_k` and `pygame.K_u` handled by `EditTool.on_event`. Neither key is
  taken anywhere else -- `src/app/input.py`'s global block has no `K_k` or `K_u` branch, and
  `TOOLS` is keyed on `K_1`/`K_2`/`K_3`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tools_edit.py`:

```python
def test_k_locks_the_selected_field_and_locks_it_again(state):
    from src import edits
    body = state.sim.world.bodies[state.selected_index]
    assert state.tool.on_event(_key(pygame.K_k), state)
    assert edits.is_locked(body, "mass")
    assert state.tool.on_event(_key(pygame.K_k), state)
    assert not edits.is_locked(body, "mass")


def test_k_locks_the_row_the_arrows_navigated_to(state):
    """It mirrors Left/Right, which act on self.field_index, not on row zero."""
    from src import edits
    body = state.sim.world.bodies[state.selected_index]
    fields = edits.edit_fields(body)
    state.tool.on_event(_key(pygame.K_DOWN), state)
    state.tool.on_event(_key(pygame.K_k), state)
    assert edits.is_locked(body, fields[1])
    assert not edits.is_locked(body, fields[0])


def test_a_locked_field_ignores_left_and_right(state):
    """The point of the whole feature: a stray arrow key cannot nudge a pinned
    value.  Enforced in edits.py, because a key never passes through the panel
    that greyed the buttons out."""
    index = state.selected_index
    before = float(state.sim.world.mass[index])
    state.tool.on_event(_key(pygame.K_k), state)
    for _ in range(20):
        state.tool.on_event(_key(pygame.K_RIGHT), state)
        state.tool.on_event(_key(pygame.K_LEFT), state)
    assert float(state.sim.world.mass[state.selected_index]) == pytest.approx(before)


def test_a_locked_field_ignores_a_drag(state):
    index = state.selected_index
    before = float(state.sim.world.mass[index])
    state.tool.on_event(_key(pygame.K_k), state)
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(2, 2),
                                           button=1), state)
    state.tool.on_event(pygame.event.Event(pygame.MOUSEMOTION, pos=(122, 2),
                                           rel=(120, 0), buttons=(1, 0, 0)), state)
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(122, 2),
                                           button=1), state)
    assert float(state.sim.world.mass[state.selected_index]) == pytest.approx(before)


def test_locking_one_field_leaves_the_others_scrubbable(state):
    index = state.selected_index
    mass = float(state.sim.world.mass[index])
    radius = float(state.sim.world.radius[index])
    state.tool.on_event(_key(pygame.K_k), state)        # lock mass
    state.tool.on_event(_key(pygame.K_DOWN), state)     # to density
    state.tool.on_event(_key(pygame.K_RIGHT), state)
    assert float(state.sim.world.mass[state.selected_index]) == pytest.approx(mass)
    assert float(state.sim.world.radius[state.selected_index]) != radius


def test_u_unlocks_everything_on_the_body(state):
    from src import edits
    body = state.sim.world.bodies[state.selected_index]
    for field in edits.edit_fields(body):
        edits.toggle_lock(body, field)
    assert state.tool.on_event(_key(pygame.K_u), state)
    assert body.locked_fields == frozenset()


def test_u_with_nothing_locked_does_nothing_and_says_so(state):
    """It matches its own disabled button: nothing to undo, so the event is not
    consumed either."""
    assert not state.tool.on_event(_key(pygame.K_u), state)


def test_k_and_u_with_nothing_selected_are_harmless(state):
    state.selected_id = None
    assert not state.tool.on_event(_key(pygame.K_k), state)
    assert not state.tool.on_event(_key(pygame.K_u), state)


def test_k_and_u_on_a_destroyed_body_are_harmless(state):
    state.sim.world.remove(state.selected_index)
    state.tool.on_event(_key(pygame.K_k), state)
    state.tool.on_event(_key(pygame.K_u), state)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_tools_edit.py -q -k "_k_ or _u_"`
Expected: failures -- `K` and `U` are not handled, so `on_event` returns False and nothing
gets locked.

- [ ] **Step 3: Handle the two keys**

In `src/app/tools.py`, in `EditTool.on_event`, insert two branches immediately after the
`K_c` branch:

```python
            if event.key == pygame.K_c:
                return self._collapse(state)
            if event.key == pygame.K_k:
                return self._toggle_lock(state)
            if event.key == pygame.K_u:
                return self._unlock_all(state)
```

- [ ] **Step 4: Teach the hint line**

Replace `EditTool.hint`:

```python
    hint = ("Click a body, then use the -- and + buttons  -  or Up/Down to pick a "
            "row and Left/Right to change it  -  hold a button to go faster")
```

with:

```python
    hint = ("Click a body, then use the -- and + buttons  -  Up/Down picks a row "
            "and Left/Right changes it  -  K locks a row so nothing can move it")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_tools_edit.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/app/tools.py tests/test_tools_edit.py
git commit -m "feat: K locks the selected field, U frees the whole body"
```

---

### Task 6: `KEY_HELP`, the parity test, and the panel end to end

**Files:**
- Modify: `src/render/hud.py:24-51` (the `KEY_HELP` table)
- Test: `tests/test_ui.py` (append; the existing parity test needs no edit)

**Interfaces:**
- Consumes: the `edit.lock` and `edit.unlock_all` controls from Task 4 and their hotkeys
  from Task 5.
- Produces: `K` and `U` documented, which is what makes
  `tests/test_ui.py::test_every_shortcut_is_also_a_button` demand the buttons.

**On the existing tests in `tests/test_ui.py`:** none of them need editing.
`test_every_shortcut_is_also_a_button` reads `KEY_HELP` and passes once the controls carry
`hotkey="K"` and `hotkey="U"` (a compact control still carries a hotkey -- it just shows it
in the tooltip through `Control.hint()` rather than as a key cap).
`test_every_button_says_what_it_does` passes because both new controls have tooltips.
`test_every_control_of_every_tool_can_be_pressed_without_raising` will now press the
padlocks; that is fine, and is worth having.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ui.py`, at the end of the file:

```python
# --- locking a property -----------------------------------------------------------

def _edit_earth(state, ui):
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    state.select(earth.id)
    click_id(ui, state, "tool.edit")
    return earth


def test_locking_a_row_greys_out_its_steppers(state, ui):
    """The second line of defence.  The first is in edits.py, and both have to
    be there: perform() is the door the held repeat and the tests come
    through."""
    _edit_earth(state, ui)
    assert find(ui, "edit.step", payload=("mass", 3)).enabled

    click_id(ui, state, "edit.lock", payload="mass")

    plus = find(ui, "edit.step", payload=("mass", 3))
    assert not plus.enabled
    before = float(state.sim.world.mass[state.selected_index])
    assert not ui.perform(plus, state, None)
    assert float(state.sim.world.mass[state.selected_index]) == pytest.approx(before)


def test_the_padlock_button_unlocks_as_well_as_locks(state, ui):
    from src import edits
    earth = _edit_earth(state, ui)
    click_id(ui, state, "edit.lock", payload="density")
    assert edits.is_locked(earth, "density")
    click_id(ui, state, "edit.lock", payload="density")
    assert not edits.is_locked(earth, "density")
    assert find(ui, "edit.step", payload=("density", 1)).enabled


def test_unlock_all_is_dead_until_something_is_locked(state, ui):
    earth = _edit_earth(state, ui)
    assert not find(ui, "edit.unlock_all").enabled

    click_id(ui, state, "edit.lock", payload="mass")
    click_id(ui, state, "edit.lock", payload="material")
    unlock = find(ui, "edit.unlock_all")
    assert unlock.enabled

    click(ui, state, unlock)
    assert earth.locked_fields == frozenset()
    assert not find(ui, "edit.unlock_all").enabled


def test_a_lock_outlives_leaving_the_tool_and_the_selection(state, ui):
    """It is a standing preference for this body, not a mode the Edit tool is
    in."""
    from src import edits
    earth = _edit_earth(state, ui)
    click_id(ui, state, "edit.lock", payload="mass")
    click_id(ui, state, "tool.look")
    state.select(state.sim.world.bodies[1].id)
    click_id(ui, state, "tool.edit")
    state.select(earth.id)
    ui.build(state, SIZE)
    assert edits.is_locked(earth, "mass")
    assert not find(ui, "edit.step", payload=("mass", 3)).enabled


def test_both_lock_keys_are_documented_and_have_buttons(state, ui):
    """The parity test above reads KEY_HELP; this one says which two keys this
    feature put in it, so a table edit that drops them is caught by name."""
    from src.render.hud import KEY_HELP
    documented = {combo for combo, _ in KEY_HELP}
    assert "K" in documented and "U" in documented
    _edit_earth(state, ui)
    assert {c.hotkey for c in ui.controls} >= {"K", "U"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_ui.py -q -k "lock or documented"`
Expected: `test_both_lock_keys_are_documented_and_have_buttons` fails on the `KEY_HELP`
assertion. The other four should already pass off Tasks 4 and 5 -- if any of them fails,
stop and fix the control wiring before going on.

- [ ] **Step 3: Document the two keys**

In `src/render/hud.py`, in `KEY_HELP`, insert two rows immediately after the `("C",
"collapse a star into a black hole (Edit)")` row:

```python
    ("C", "collapse a star into a black hole (Edit)"),
    ("K", "lock or unlock the selected field (Edit)"),
    ("U", "unlock every field on the selected body (Edit)"),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_ui.py tests/test_hud.py -q`
Expected: all pass, including `test_every_shortcut_is_also_a_button`.

If `test_every_shortcut_is_also_a_button` fails with `no button for ['K']` or `['U']`, the
cause is that the parity test builds the panel for `state.sim.world.bodies[3]` -- check that
the control genuinely carries `hotkey="K"` / `hotkey="U"` for **every** kind of body, since
`edit.lock` exists on every field row and `edit.unlock_all` on every panel.

- [ ] **Step 5: Commit**

```bash
git add src/render/hud.py tests/test_ui.py
git commit -m "feat: document K and U, and check the panel locks end to end"
```

---

### Task 7: The panel still fits at the interface floor

**Files:**
- Test: `tests/test_layout.py` (append)

**Interfaces:**
- Consumes: the whole panel from Tasks 4-6.
- Produces: nothing code depends on. This is the test that keeps the "Unlock all shares the
  Collapse row" decision from being quietly undone.

The two existing parametrised tests in this file (`test_every_control_is_inside_the_window`,
`test_every_panel_is_inside_the_window`) already cover the Edit tool at every window and grid
and **must keep passing unchanged**. In the worst case -- a body wearing the longest material
name, "Hydrogen & Helium" -- the fourth control per row widens the panel from 218 to 238 px
at the finest grid and from 102 to 115 at the coarsest, which pushes the body list's right
edge to 384 of 480 and to 194 of 213 respectively: still on screen, at the floor, with the
list open. What those tests do *not* check is that the panel stops above the transport bar,
which is the constraint a seventh row would break.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_layout.py`:

```python
def test_the_edit_panel_stays_clear_of_the_transport_bar():
    """The Edit panel grows a row per field, and now a padlock on each of them.

    Nothing else enforces that it stops above the transport bar: the two tests
    above only ask that every control is inside the buffer, and a panel that
    reaches over Play is inside the buffer.  At the floor there are six rows of
    room and no more, which is why "Unlock all" shares the Collapse row instead
    of taking one of its own.
    """
    from src import edits
    from src.render.widgets import Metrics

    for s in config.GRID_CHOICES:
        internal = config.min_internal(s)
        font = PixelFont(config.font_px(s))
        metrics = Metrics(font, *internal)
        state = AppState(Simulation(), Camera(*internal), ViewSettings())
        state.set_tool(EditTool)
        ui = UiLayer(font, Hud(font))

        def deepest(name):
            index = next(i for i, b in enumerate(state.sim.world.bodies)
                         if b.name == name)
            state.select(state.sim.world.bodies[index].id)
            ui.build(state, internal, None)
            return index, max(c.rect.bottom for c in ui.controls
                              if c.id.startswith("edit."))

        for name in ("Earth", "Sun"):          # a planet: four rows; a star: three
            _, bottom = deepest(name)
            assert bottom <= metrics.transport_y, \
                "the Edit panel reaches %d, past the transport bar at %d " \
                "(grid %d, %s)" % (bottom, metrics.transport_y, s, name)

        index, _ = deepest("Sun")
        edits.collapse(state.sim.world, index)
        _, bottom = deepest("Sun")
        assert bottom <= metrics.transport_y, \
            "the black hole panel reaches %d, past %d (grid %d)" % (
                bottom, metrics.transport_y, s)
```

- [ ] **Step 2: Run it**

Run: `python3 -m pytest tests/test_layout.py -q`
Expected: **all pass**, this one included. This test is written to pass against the layout
Task 4 built -- it is a regression guard, not a red test. If it fails, the cause is almost
certainly that "Unlock all" was given a row of its own; put it back on the Collapse row
rather than shrinking anything.

- [ ] **Step 3: Confirm it is actually load-bearing**

Temporarily change `EditTool.controls` so the non-black-hole branch appends two rows:

```python
        if body.kind != KIND_BLACK_HOLE:
            rows.append([Control("edit.collapse", label="Collapse", icon="blackhole",
                                 hotkey="C", enabled=body.kind == KIND_STAR,
                                 tooltip="Crush this star into a black hole -- "
                                         "there is no way back")])
            rows.append([unlock_all])
        else:
            rows.append([unlock_all])
```

Run: `python3 -m pytest tests/test_layout.py -q -k transport`
Expected: **FAIL**, reporting a bottom of 173 against a transport bar at 171 at grid 3.
Then revert the change (`git checkout -- src/app/tools.py`) and re-run to confirm it passes
again.

- [ ] **Step 4: Commit**

```bash
git add tests/test_layout.py
git commit -m "test: the Edit panel stops above the transport bar at the floor"
```

---

### Task 8: Document the invariant, and run the whole suite

**Files:**
- Modify: `CLAUDE.md` (the Invariants list, the "Adding a control" note, the test count in
  Commands)

**Interfaces:**
- Consumes: everything.
- Produces: nothing code depends on.

- [ ] **Step 1: Run the whole suite and record the count**

Run: `python3 selftest.py`
Expected: every test passes. Note the total printed -- the Commands block quotes it, and it
is currently 744.

- [ ] **Step 2: Add the invariant**

Add this bullet to the **Invariants** list in `CLAUDE.md`, immediately after the
"`thermal.pinned_temperature` decides who is free" bullet:

```markdown
- **A locked field is refused in `edits.py`, not in the panel.** `Body.locked_fields`
  is a frozenset of the same names `edits.edit_fields` returns, and every setter
  (`set_mass`, `set_density`, `set_temperature`, `set_spin`, `set_disk`,
  `set_material`/`cycle_material`) returns without writing when its **own** field is
  locked. The greyed-out stepper is the second line of defence, not the first: the
  arrow keys and the horizontal drag never pass through a button. A lock guards only
  its own field -- locking density does not stop a material swap from moving density
  -- and it reaches no further than those setters, so `collapse` and the pinned
  temperature `_refresh_kind` writes are unaffected. Locks live on the Body beside
  `spin` and `disk` because the force kernel never reads them; nothing inherits them,
  and a kind change leaves them alone, so a lock on a field the current kind does not
  offer is inert rather than lost.
```

- [ ] **Step 3: Note the layout constraint where a control is added**

In the **Adding a control** section of `CLAUDE.md`, append this paragraph:

```markdown
A tool panel row is not free: `tests/test_layout.py` checks that the Edit panel stops
above the transport bar at `config.min_internal(3)`, where there is room for six rows
and no more. That is why "Unlock all" shares the Collapse row rather than taking one of
its own.
```

- [ ] **Step 4: Update the test count**

In the Commands block, change

```
python selftest.py                      # whole suite (744 tests, ~3:35)
```

to the count recorded in Step 1.

- [ ] **Step 5: Run the whole suite once more and commit**

```bash
python3 selftest.py
git add CLAUDE.md
git commit -m "docs: the property lock invariant in CLAUDE.md"
```

---

## Self-Review

**Spec coverage.**
- Data model, `Body.locked_fields` defaulting to an empty frozenset, values matching
  `edit_fields()` -- Task 1.
- Fresh bodies and fragments start unlocked; a kind change does not touch the set; a lock on
  a field the current kind does not offer is inert -- Tasks 1 and 2.
- `is_locked` / `toggle_lock` / `unlock_all` -- Task 1.
- The guard table over all six setters, `set_temperature`'s existing early return joined
  rather than replaced, `_refresh_kind` / `collapse` / the density-follows-mass paths
  unaffected -- Task 2.
- Steppers `enabled=not is_locked(...)`; `edit.lock` always enabled with the icon swap;
  `edit.unlock_all` beside Collapse, disabled when nothing is locked; `perform` setting
  `field_index`; the hint clause -- Tasks 4 and 5.
- `K` and `U` in `on_event`, guarded on `state.selected_index is not None`, with unlock-all
  re-checking `body.locked_fields` so its key path matches its disabled button -- Task 5.
- Two `KEY_HELP` rows -- Task 6.
- `"lock"` and `"unlock"` in `draw_icon`, drawn from lines, reading as a pair -- Task 3.
- Testing section: `tests/test_edits.py` (Tasks 1, 2), `tests/test_tools_edit.py`
  (Tasks 4, 5), `tests/test_ui.py` (Task 6 -- the table update plus five end-to-end tests),
  `tests/test_layout.py` (Task 7). `tests/test_widgets.py` is additionally touched because
  the feature adds two icons, which the spec's `draw_icon` section implies.
- Both risks the spec names are answered concretely: row width at the smallest grid is
  measured in Task 4 and pinned by the test in Task 7, and icon legibility at compact size is
  a test in Task 3 rather than a hope.

**Deviations from a literal reading of the spec, and why.** The spec's Architecture section
says "a new panel-level control, `edit.unlock_all`, placed next to `Collapse`", while its
Interface section says "a new 'Unlock all' row appears under Collapse". This plan takes the
first reading -- same row -- because the second does not fit: measured at
`config.min_internal(3)` a seventh row reaches y=173 against a transport bar at y=171. The
spec's own risk note says the fix for a floor that does not fit is compaction, not a second
layout, and this is the compaction. The black hole panel, which has no Collapse row, gets the
row-of-its-own the Interface section describes.

**Types.** `is_locked`, `toggle_lock` and `unlock_all` take a `Body` in every call site
(`edits.py`'s setters read `world.bodies[index]` first; `tools.py` reads
`state.sim.world.bodies[index]`). `Body.locked_fields` is a `frozenset` at every
assignment -- `frozenset()`, `| {field}` and `- {field}` all return one.
`EditTool._toggle_lock(state, field=None)` and `EditTool._unlock_all(state)` both return
`bool`, which is what `perform` and `on_event` both need. `Control.payload` on `edit.lock` is
the bare field string, distinct from `edit.step`'s `(field, notches)` tuple and the same as
`edit.field`'s -- `find(ui, cid, payload)` in `tests/test_ui.py` matches on both id and
payload, so the two never collide.

**Known follow-ups for the verification pass, not for the implementer:** whether the
inspector should show a padlock beside a locked value as well as the panel (out of scope --
the spec changes no HUD readout); whether a lock should survive a `Restart` (it cannot -- the
scenario rebuilds every Body); and whether locking mass ought to block `collapse` as well
(decided no in Task 2, and tested there, because collapse conserves mass).
