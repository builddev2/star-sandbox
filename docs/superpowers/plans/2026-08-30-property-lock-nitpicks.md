# Property Lock -- Nitpicks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the five non-blocking nitpicks the evaluation pass raised against the
shipped property-lock feature (`docs/superpowers/plans/2026-08-30-property-lock.md`,
merged at `d615034`). Two are wording, one is a crash guard, two are missing tests and
one missing sentence of documentation. Nothing here changes what the feature *does*.

**Tech Stack:** Python 3.11, numpy, pygame 2.6.1. Tests with pytest.

**Spec:** `docs/superpowers/specs/2026-08-30-property-lock-design.md` (unchanged -- none
of this alters the design).

## What this plan covers

| | nitpick | verdict |
|---|---|---|
| N1 | the padlock tooltip overstates what `K` does | fixed, Task 1 |
| N2 | `EditTool.hint` lost "hold a button to go faster" | fixed, Task 2 |
| N3 | `fields.index(field)` unguarded at four sites | **fixed, all four**, Task 3 -- see below |
| N4 | untested: a merge survivor keeps its own locks | test added, Task 4 |
| N5 | Restart drops every lock, undocumented and untested | documented + tested, Task 5 |
| N6 | function-local `from src import edits` in the new tests | **not a defect -- do not touch** |

### N3: the scope call, and why

**Fix it, at all four sites.** The nitpick was raised as a theoretical risk in
pre-existing code, but it is a demonstrable crash today, and the new `edit.lock` branch
has it too, so "fix only the new site" would be an inconsistent patch and "fix nothing"
leaves a reachable `ValueError` that this feature widened.

Measured against this HEAD -- select Earth, build the Edit panel, then let the body
ignite underneath the panel that was already built, then press a control from it:

```
kind now star   fields ('mass', 'density', 'material')
edit.step -> ValueError: tuple.index(x): x not in tuple
edit.lock -> ValueError: tuple.index(x): x not in tuple
no selection  edit.step -> TypeError: list indices must be integers or slices, not NoneType
no selection  edit.lock -> ok
```

The window is not hypothetical. `UiLayer._pressed` (`src/app/ui.py:487-509`,
`src/app/ui.py:537-551`) holds the `Control` **object** across frames and re-performs it
from `update()` while a stepper is held, while `ui.build` throws that object away and
builds a fresh panel every frame. So a child holding the Temp `+` on a planet that then
merges past `STAR_MASS_THRESHOLD` is performing a control for a row `edit_fields` no
longer returns. A second route is two presses in one event drain: the body list picks a
star, then the click behind it lands on the planet panel's Temp row.

**But the fix stays narrow.** One small private helper, `EditTool._pick_row`, used at
all four sites; it returns `False` and does nothing when the row is gone. No method is
restructured, no control is redesigned, and `_scrub` (`src/app/tools.py:714-720`) already
guards `index is None` and clamps `field_index`, so nothing downstream needs touching.
A helper rather than the same four-line guard copied four times is what this codebase
does elsewhere -- "Drawing, picking and the contact refresh all read the one helper so
they cannot disagree."

### N6: investigated, and it is not a defect

The evaluation claimed the new tests broke convention with function-local
`from src import edits` imports. Verified false: `tests/test_tools_edit.py:36`, `:54`,
`:70`, `:109` are all **pre-existing** tests written before this feature and all use
exactly that pattern, as does the file's `state` fixture (`:19-24`) and its `_metrics`
helper (`:163-166`). It is the established convention in that file. **Do not change any
import in any test file.**

## Global Constraints

- **Nothing in this plan changes behaviour a user can see, except two strings.** N3's
  guard turns a crash into a no-op; N4 and N5 are tests; N5 also adds one sentence to
  `CLAUDE.md`. If a source change grows past that, stop.
- **Every action has a button, and every button carries its hotkey.** `Control.hint()`
  (`src/render/widgets.py:94-99`) appends `   [K]` to any control that carries a hotkey.
  Task 1 must not remove `hotkey="K"` from `edit.lock` -- `tests/test_ui.py`'s parity
  test reads `hud.KEY_HELP` and fails if `K` has no button.
- **A hotkey does not inherit a control's `enabled` gate.** Unchanged by this plan; the
  key paths (`_toggle_lock(state)` with `field=None`, `_unlock_all`) keep their own
  guards.
- **A merge is not an edit.** `collisions.merge` writes `world.mass`, `world.radius`,
  `world.contact_radius` and `world.temperature` directly and never consults
  `locked_fields`. Task 4 pins that as *correct behaviour*; it is not a bug report. If
  the Task 4 test fails, do not "fix" `collisions.py` -- the test is wrong.
- **Tooltips are one line and are clipped at the window edge** (`draw_tooltip`,
  `src/render/widgets.py:315-340`: `x = max(2, min(x, surface.get_width() - width - 2))`
  does not shrink a box wider than the buffer). Task 1's replacement was measured at
  every grid floor and fits; do not lengthen it further.
- **`src/*.py` must never import pygame.** Nothing here gives any module a new import.
- Run the tests with `python3` from the worktree root -- this worktree has no `.venv/`.
  Whole suite: `python3 selftest.py` (~3:35, currently **803** tests). One file:
  `python3 -m pytest tests/test_tools_edit.py -q`.
- Comments and docstrings explain **why**, not what. British spelling in prose; en-dash
  written as `--`.

---

### Task 1 (N1): the padlock tooltip stops promising `K` acts on *this* row

**Files:**
- Modify: `src/app/tools.py:569-572` (the `edit.lock` `Control`)
- Test: `tests/test_tools_edit.py` (append to the `# --- locking ---` section)

**Interfaces:** consumes nothing, produces nothing. Two strings.

**The problem.** `Control.hint()` renders the padlock's tooltip as
`"Lock density -- pin it where it is   [K]"`. `K` does **not** act on the hovered row --
`EditTool._toggle_lock(state)` with `field=None` acts on `self.field_index`, the row that
is *selected*. Hover Density's padlock while Mass is selected and the tooltip promises
one thing and `K` does another.

**The replacement.** Both states get the same second clause, because `K` toggles and the
old wording's flavour half ("pin it where it is" / "let it be changed again") is what the
padlock icon already says:

```
Lock density -- K locks or frees the row that is picked      [K]
Unlock density -- K locks or frees the row that is picked    [K]
```

Measured with `draw_tooltip` at `config.min_internal(s)` for every grid, with the longest
field name ("material"): box widths 348/360 px at grid 1 (buffer 480), 178/184 at grid 2
(buffer 240), 147/152 at grid 3 (buffer 213). The existing longest Edit tooltip, `"Or
pick this row with Up/Down and change it with Left/Right"`, is 351/190/150. So this stays
at or under what the panel already draws, and fits at the floor.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tools_edit.py`:

```python
def test_the_padlock_tooltip_does_not_promise_K_acts_on_this_row(state):
    """Control.hint() appends "[K]" to anything carrying a hotkey, and K acts on
    the *selected* row, not the hovered one.  So hovering the Density padlock
    while Mass is picked must not read as a promise that K will lock density.
    """
    from src import edits
    body = state.sim.world.bodies[state.selected_index]
    for field in edits.edit_fields(body):
        edits.toggle_lock(body, field)          # check the locked wording too
    for lock in _locks(state.tool.controls(state, _metrics())):
        assert "the row that is picked" in lock.tooltip, lock.tooltip
    edits.unlock_all(body)
    for lock in _locks(state.tool.controls(state, _metrics())):
        assert "the row that is picked" in lock.tooltip, lock.tooltip
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python3 -m pytest tests/test_tools_edit.py -q -k promise`
Expected: fails on the first assertion, reporting the old
`"Lock mass -- pin it where it is"`.

- [ ] **Step 3: Reword the tooltip**

In `src/app/tools.py`, replace:

```python
                Control("edit.lock", icon="lock" if locked else "unlock",
                        payload=field, compact=True, hotkey="K", active=locked,
                        tooltip=("Unlock %s -- let it be changed again" % field)
                        if locked else ("Lock %s -- pin it where it is" % field)),
```

with:

```python
                # The hotkey is appended by Control.hint(), and K acts on
                # whichever row is *selected* -- so the text must not read as a
                # promise about this row, which may not be the selected one.
                Control("edit.lock", icon="lock" if locked else "unlock",
                        payload=field, compact=True, hotkey="K", active=locked,
                        tooltip="%s %s -- K locks or frees the row that is picked"
                                % ("Unlock" if locked else "Lock", field)),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_tools_edit.py tests/test_ui.py -q`
Expected: all pass. `test_every_button_says_what_it_does` still passes (the control still
has a tooltip) and `test_every_shortcut_is_also_a_button` still passes (`hotkey="K"` was
kept).

- [ ] **Step 5: Commit**

```bash
git add src/app/tools.py tests/test_tools_edit.py
git commit -m "fix: the padlock tooltip says which row K acts on"
```

---

### Task 2 (N2): the Edit hint says again that a button repeats when held

**Files:**
- Modify: `src/app/tools.py:477-478` (`EditTool.hint`)
- Test: none. No test in the suite asserts hint prose; `tests/test_state.py:150` only
  asserts `state.hint` is a non-empty string, and `tests/test_tools_select.py:51` that it
  is the tool's own. Both keep passing. This is a one-string documentation fix and a
  prose-matching test would be brittle without buying anything.

**The problem.** The pre-property-lock hint ended `"hold a button to go faster"`. That
clause was dropped when the `K` clause was added, and it is a real, still-true
affordance: `ui.REPEAT_DELAY_MS` / `REPEAT_INTERVAL_MS` fire a held stepper repeatedly,
and mass scrubs by ratio at 1.06 a notch, so igniting Jupiter is ~80 notches. One click
at a time is unusable, and now nothing on screen says so.

**Length.** Four clauses at 160 characters, against `SelectTool.hint`
(`src/app/tools.py:62-63`, four clauses, 136 chars) and `LaunchTool.hint`
(`src/app/tools.py:159-161`, four clauses, 183 chars). The hint draws on one line at the
bottom of the window (`hud._draw_hint`, `src/render/hud.py:346-351`) and is clipped at
the edge, so it must not become the longest of the three -- it does not.

- [ ] **Step 1: Restore the clause**

In `src/app/tools.py`, replace:

```python
    hint = ("Click a body, then use the -- and + buttons  -  Up/Down picks a row "
            "and Left/Right changes it  -  K locks a row so nothing can move it")
```

with:

```python
    hint = ("Click a body, then use the -- and + buttons  -  hold one to go faster "
            " -  Up/Down picks a row and Left/Right changes it  -  K locks a row "
            "so nothing can move it")
```

Note the clause separator this codebase uses is two spaces, a hyphen, two spaces; the
line break above splits it, so the assembled string must read
`"...buttons  -  hold one to go faster  -  Up/Down picks..."`. Check the joined value,
not the source lines.

- [ ] **Step 2: Verify the assembled string and its length**

Run:

```bash
python3 -c "from src.app.tools import EditTool, LaunchTool, SelectTool; \
h = EditTool.hint; print(repr(h)); print(len(h), len(SelectTool.hint), len(LaunchTool.hint)); \
assert '  -  hold one to go faster  -  ' in h; \
assert len(h) <= len(LaunchTool.hint)"
```

Expected: the separators are intact and the Edit hint is no longer than the Throw hint.

- [ ] **Step 3: Run the tests**

Run: `python3 -m pytest tests/test_tools_edit.py tests/test_tools_select.py tests/test_state.py tests/test_hud.py -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/app/tools.py
git commit -m "fix: the Edit hint says again that a stepper repeats when held"
```

---

### Task 3 (N3): a control for a row the body no longer offers is ignored

**Files:**
- Modify: `src/app/tools.py` -- add `_pick_row` after `_field_count`
  (`src/app/tools.py:496-511`, immediately before `field_values`); rewrite the
  `edit.field` and `edit.step` branches of `perform` (`src/app/tools.py:596-604`); use it
  in `_toggle_lock` (`src/app/tools.py:645-651`)
- Test: `tests/test_tools_edit.py` (append)

**Interfaces:**
- Consumes: `EditTool._fields`.
- Produces: `EditTool._pick_row(state, field) -> bool`. Four call sites, no new control,
  no new key.

**Behaviour change, precisely.** `perform` returns `False` instead of raising when the
selected body no longer offers the pressed control's field, or when nothing is selected
at all. `UiLayer.perform` (`src/app/ui.py:661`) does `return bool(tool.perform(...))` and
its own caller discards that, so a `False` is indistinguishable from an unrecognised
control -- nothing downstream changes. No existing test asserts the return value of
`perform` for `edit.field` or `edit.step`; the only lock test that does
(`test_pressing_a_lock_toggles_it_and_picks_that_row`) presses a control from a panel
built the same instant, which still returns `True`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tools_edit.py`:

```python
# --- a control from a panel the body has outgrown ---------------------------------

def _stale_planet_controls(state):
    """The Edit panel for a planet, and then a body that is no longer one.

    The panel is built at the end of a frame from the body as it was then; the
    press it answers arrives at the start of the next one, or -- with a held
    stepper, which UiLayer keeps a reference to across frames -- many frames
    later.  A merge in between can ignite the body, and edit_fields then stops
    offering Temp.
    """
    from src import edits, units
    rows = state.tool.controls(state, _metrics())
    stale = {
        "step": next(c for row in rows for c in row
                     if c.id == "edit.step" and c.payload[0] == "temp"),
        "field": next(c for row in rows for c in row
                      if c.id == "edit.field" and c.payload == "temp"),
        "lock": next(c for row in rows for c in row
                     if c.id == "edit.lock" and c.payload == "temp"),
    }
    edits.set_mass(state.sim.world, state.selected_index,
                   units.STAR_MASS_THRESHOLD * 1.1)
    assert "temp" not in edits.edit_fields(
        state.sim.world.bodies[state.selected_index])
    return stale


def test_a_stale_control_is_ignored_rather_than_raising(state):
    """fields.index() raises ValueError on a row that has gone.  Saying "that
    row is gone" and doing nothing is right: the panel is rebuilt from the new
    kind before the next press anyway."""
    stale = _stale_planet_controls(state)
    # Read *after* igniting: _refresh_kind pins a star's temperature the moment
    # it lights up, so the pre-ignition value is not what a no-op must preserve.
    temperature = float(state.sim.world.temperature[state.selected_index])
    for control in stale.values():
        assert state.tool.perform(control, state) is False
    assert float(state.sim.world.temperature[state.selected_index]) == \
        pytest.approx(temperature)


def test_a_stale_control_leaves_the_selected_row_alone(state):
    """A press that does nothing must not move the cursor either."""
    state.tool.field_index = 1
    for control in _stale_planet_controls(state).values():
        state.tool.perform(control, state)
    assert state.tool.field_index == 1


def test_a_stale_lock_does_not_lock_the_wrong_field(state):
    """The worst outcome of guessing: silently pinning whatever row happens to
    sit at that index in the new tuple."""
    stale = _stale_planet_controls(state)
    body = state.sim.world.bodies[state.selected_index]
    state.tool.perform(stale["lock"], state)
    assert body.locked_fields == frozenset()


def test_pressing_an_edit_control_with_nothing_selected_is_harmless(state):
    """selected_index is None once the selected body is destroyed, and
    world.bodies[None] raises TypeError -- which edit.step and edit.field did."""
    rows = state.tool.controls(state, _metrics())
    controls = [c for row in rows for c in row
                if c.id in ("edit.step", "edit.field", "edit.lock")]
    state.selected_id = None
    for control in controls:
        assert state.tool.perform(control, state) is False
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python3 -m pytest tests/test_tools_edit.py -q -k "stale or nothing_selected_is_harmless"`
Expected: `ValueError: tuple.index(x): x not in tuple` from the three stale tests and
`TypeError: list indices must be integers or slices, not NoneType` from the fourth.

- [ ] **Step 3: Add `_pick_row`**

In `src/app/tools.py`, immediately after `_field_count` and before `field_values`, add:

```python
    def _pick_row(self, state, field):
        """Point ``field_index`` at ``field``'s row.  False if there is no such row.

        The panel is built at the end of a frame from the body as it was then,
        and the press it answers arrives at the start of the next one -- or,
        for a held stepper, many frames later, because ``UiLayer`` keeps the
        pressed ``Control`` object while ``ui.build`` replaces the panel around
        it.  In between the body can change kind, and ``edit_fields`` then
        returns a different tuple: a planet that merges past the ignition
        threshold loses its Temp row, and ``fields.index("temp")`` on a star
        raises.  A destroyed body is the same story with ``selected_index``
        gone.

        Doing nothing is the whole fix.  Guessing a row by position would pin
        or scrub whichever field happened to land at that index instead, and
        the panel is rebuilt from the new kind before the next press anyway.
        """
        index = state.selected_index
        if index is None:
            return False
        fields = self._fields(state.sim.world, index)
        if field not in fields:
            return False
        self.field_index = fields.index(field)
        return True
```

- [ ] **Step 4: Use it in `perform`**

Replace the first two branches of `EditTool.perform`:

```python
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
```

with:

```python
        if control.id == "edit.field":
            return self._pick_row(state, control.payload)
        if control.id == "edit.step":
            field, notches = control.payload
            if not self._pick_row(state, field):
                return False
            self._scrub(state, notches)
            return True
```

- [ ] **Step 5: Use it in `_toggle_lock`**

In `EditTool._toggle_lock`, replace:

```python
        if field is None:
            field = fields[min(self.field_index, len(fields) - 1)]
        else:
            self.field_index = fields.index(field)
        edits.toggle_lock(world.bodies[index], field)
        return True
```

with:

```python
        if field is None:
            field = fields[min(self.field_index, len(fields) - 1)]
        elif not self._pick_row(state, field):
            return False
        edits.toggle_lock(world.bodies[index], field)
        return True
```

The `field is None` arm is the key path and needs no guard of its own: it reads the row
out of the tuple it just fetched, clamped, so it cannot be stale.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_tools_edit.py tests/test_ui.py tests/test_layout.py -q`
Expected: all pass. In particular
`test_ui.py::test_every_control_of_every_tool_can_be_pressed_without_raising` and the
existing `test_pressing_a_lock_toggles_it_and_picks_that_row` (which still asserts
`perform` returns `True` on a fresh panel).

- [ ] **Step 7: Confirm the guard is actually load-bearing**

Temporarily change `_pick_row`'s `if field not in fields:` to `if False:` and run
`python3 -m pytest tests/test_tools_edit.py -q -k stale`. Expected: **FAIL** with
`ValueError`. Revert (`git checkout -- src/app/tools.py` would also throw away Steps 3-5
-- undo the two characters by hand instead) and re-run to confirm green.

- [ ] **Step 8: Commit**

```bash
git add src/app/tools.py tests/test_tools_edit.py
git commit -m "fix: an Edit control for a row the body no longer offers does nothing"
```

---

### Task 4 (N4): a merge survivor keeps its own locks

**Files:**
- Test only: `tests/test_collision_merge.py` (widen the import on line 4, append one test)

**Interfaces:** consumes `edits.toggle_lock`. Produces nothing. **No source change.**

**Why this file.** `tests/test_collision_merge.py` already owns merge outcomes and
already builds a two-body `World` by hand (`_pair`, lines 10-15), which is exactly the
setup this needs; `tests/test_edits.py` would have to import `collisions` to say the same
thing. This file's convention is top-level `from src import ...`, unlike
`tests/test_tools_edit.py` -- match the file you are in.

**What it pins.** Two things at once, and both matter:

1. A merge is **not** an edit. `collisions.merge` (`src/collisions.py:222-282`) writes
   `world.mass[i]`, `world.radius[i]`, `world.contact_radius[i]` and
   `world.temperature[i]` straight into the arrays, never through `edits.set_*`, so a
   locked mass does not stop a body from growing by eating another. That is correct and
   `CLAUDE.md` already says so ("`collapse`, a merge, ablation ... all still touch a
   locked field, because none of them is an edit"). If this assertion fails, the bug is
   in whatever started routing merges through the setters.
2. The survivor is the same `Body` object throughout (`survivor = world.bodies[i]`,
   returned via `world.index_of(survivor.id)` after the swap-remove), so its
   `locked_fields` must come out the far side untouched -- neither cleared nor
   inheriting the victim's. The existing tests only cover a *fresh* body starting
   unlocked; nothing covered a survivor.

Verified against this HEAD: locks survive, mass grows from 3.0035e-06 to 3.0404e-06 solar
masses, and a 500 K survivor eating a 100 K Moon comes out at 495.1 K.

- [ ] **Step 1: Widen the import**

In `tests/test_collision_merge.py`, change line 4 from:

```python
from src import collisions, materials, units
```

to:

```python
from src import collisions, edits, materials, units
```

- [ ] **Step 2: Write the test**

Append to `tests/test_collision_merge.py`:

```python
def test_the_survivor_keeps_its_own_locks_and_the_merge_ignores_them():
    """A merge is not an edit.

    It writes mass, radius and temperature into the arrays directly rather than
    through edits.set_*, so a locked mass does not stop a body growing by
    eating another -- and the survivor is the same Body object throughout, so
    the locks it was wearing must come out the far side untouched.  Every
    other lock test covers a *fresh* body starting unlocked; this is the one
    body that is neither fresh nor destroyed.
    """
    r = units.EARTH_RADIUS
    world = _pair(units.EARTH_MASS, r, (0.0, 0.0),
                  units.MOON_MASS, r * 0.27, (0.0, 0.0), r)
    world.temperature[0] = 500.0
    world.temperature[1] = 100.0
    survivor = world.bodies[0]
    edits.toggle_lock(survivor, "mass")
    edits.toggle_lock(survivor, "temp")
    mass, temperature = float(world.mass[0]), float(world.temperature[0])

    idx = collisions.merge(world, collisions.Collision(0, 1, 0.5, 0.0))

    assert world.bodies[idx] is survivor
    assert survivor.locked_fields == frozenset({"mass", "temp"})
    assert float(world.mass[idx]) > mass
    assert float(world.temperature[idx]) != pytest.approx(temperature)


def test_a_merge_does_not_hand_the_victims_locks_to_the_survivor():
    """Nothing inherits a lock -- not a fragment, and not the body that ate
    its owner."""
    r = units.EARTH_RADIUS
    world = _pair(units.EARTH_MASS, r, (0.0, 0.0),
                  units.MOON_MASS, r * 0.27, (0.0, 0.0), r)
    edits.toggle_lock(world.bodies[1], "material")

    idx = collisions.merge(world, collisions.Collision(0, 1, 0.5, 0.0))

    assert world.bodies[idx].locked_fields == frozenset()
```

- [ ] **Step 3: Run it**

Run: `python3 -m pytest tests/test_collision_merge.py -q`
Expected: **all pass**, these two included. They are regression guards, not red tests --
`collisions.py` already behaves this way and this plan changes no source. If either
fails, **do not change `src/collisions.py`**: re-read the two assertions against what
`merge` actually does and fix the test.

- [ ] **Step 4: Commit**

```bash
git add tests/test_collision_merge.py
git commit -m "test: a merge survivor keeps its own locks, and inherits none"
```

---

### Task 5 (N5): a restart clears every lock -- written down and tested

**Files:**
- Modify: `CLAUDE.md:111-126` (the property-lock invariant paragraph)
- Test: `tests/test_state.py` (append, next to `test_reset_restores_everything`)

**Interfaces:** consumes `edits.toggle_lock` and `AppState.reset`. **No source change.**

**Why it is already true.** `AppState.reset` (`src/app/state.py:246-250`) calls
`Simulation.reset` (`src/sim.py:43-51`), which calls the scenario builder, which begins
with `world.clear()` (`src/scenarios.py:62`; `World.clear`, `src/world.py:111-115`,
empties `self.bodies`) and then rebuilds every body through `world.add`. Every `Body` is
a new object with `locked_fields = frozenset()`. Verified against this HEAD: after
locking Earth's mass and calling `sim.reset("solar_system")`, no body carries a lock and
no body is the object that did.

It is the right behaviour -- Restart means "put the solar system back how it started",
and a lock is a property of a body that no longer exists -- but nothing said so, and
nothing would catch a future scenario builder that reused `Body` objects instead of
rebuilding them.

**Why `tests/test_state.py`.** It owns `AppState.reset` already
(`test_reset_restores_everything`, `tests/test_state.py:133-141`) and its fixture is a
plain `AppState` over a real `Simulation`. `tests/test_ui.py`'s
`test_restart_puts_the_solar_system_back` goes through the `sim.reset` button, which is
the same path one layer up; one test is enough.

- [ ] **Step 1: Write the test**

Append to `tests/test_state.py`:

```python
def test_a_restart_clears_every_lock(state):
    """Restart rebuilds every Body from the scenario, so a lock cannot survive
    it -- the body it belonged to does not.  Right, but only accidentally so
    until something says it out loud: a scenario builder that reused Body
    objects instead of world.clear()-ing them would carry stale locks into a
    world the child asked to have put back."""
    from src import edits
    body = next(b for b in state.sim.world.bodies if b.name == "Earth")
    for field in edits.edit_fields(body):
        edits.toggle_lock(body, field)
    assert body.locked_fields

    state.reset()

    assert all(b.locked_fields == frozenset() for b in state.sim.world.bodies)
    assert all(b is not body for b in state.sim.world.bodies)
```

- [ ] **Step 2: Run it**

Run: `python3 -m pytest tests/test_state.py -q`
Expected: **all pass**, this one included -- another regression guard rather than a red
test.

- [ ] **Step 3: Document it**

In `CLAUDE.md`, in the property-lock invariant bullet, replace the closing sentence:

```markdown
  live on the Body beside `spin` and `disk` because the force kernel never reads them;
  nothing inherits them, and a kind change leaves them alone, so a lock on a field the
  current kind does not offer is inert rather than lost.
```

with:

```markdown
  live on the Body beside `spin` and `disk` because the force kernel never reads them;
  nothing inherits them, and a kind change leaves them alone, so a lock on a field the
  current kind does not offer is inert rather than lost. **A Restart (`F5`) drops every
  lock**, because the scenario `world.clear()`s and rebuilds every `Body` from scratch,
  and a lock is a property of a body that no longer exists.
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md tests/test_state.py
git commit -m "docs: a restart clears every lock, said out loud and tested"
```

---

### Task 6: the whole suite, and the test count

**Files:**
- Modify: `CLAUDE.md:8` (the `selftest.py` line in the Commands block)

**Interfaces:** consumes everything.

- [ ] **Step 1: Run the whole suite**

Run: `python3 selftest.py`
Expected: every test passes. Note the total printed. It is 803 at the start of this plan
and this plan adds eight tests (one in Task 1, four in Task 3, two in Task 4, one in
Task 5), so expect **811**.

- [ ] **Step 2: Update the count**

In `CLAUDE.md`, change

```
python selftest.py                      # whole suite (803 tests, ~3:35)
```

to the count recorded in Step 1.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: the test count after the property-lock nitpicks"
```

---

## Self-Review

**Every item from the brief, against what this plan actually does.**

- **N1 -- padlock tooltip.** Task 1 replaces both strings so neither implies `K` acts on
  the hovered row, keeps `hotkey="K"` (so the parity test and `Control.hint()`'s `[K]`
  cap are untouched), matches the file's `--`-and-lowercase-field house style, and is
  short: measured at every grid floor with `draw_tooltip` and the longest field name, it
  is at or under the panel's existing longest tooltip. One test, asserting the
  disambiguating clause is present in both the locked and unlocked wording.
- **N2 -- the hint.** Task 2 works "hold one to go faster" back in as its own clause,
  positioned immediately after "the -- and + buttons" so it attaches to the right noun.
  Four clauses, 160 characters, checked against `SelectTool.hint` (136) and
  `LaunchTool.hint` (183) -- it does not become the longest, which matters because
  `hud._draw_hint` clips at the window edge. No test: nothing in the suite asserts hint
  prose, and a prose assertion would be brittle. Step 2 verifies the assembled string's
  separators survive the source line break, which is the one thing that can silently go
  wrong here.
- **N3 -- `fields.index`.** **Decision: fix it, at all four sites**, with the reasoning
  in the header rather than buried. It is not theoretical -- both `ValueError` (a row the
  kind no longer offers) and `TypeError` (nothing selected) were reproduced against this
  HEAD, and `UiLayer._pressed` holding a `Control` across frames while `ui.build` replaces
  the panel is a real route to it. Fixing only the new `edit.lock` site would leave two
  identical bugs behind an inconsistent patch. The fix is narrow: one helper,
  `return False` when the row is gone, no method restructured, no downstream change
  (`ui.perform`'s return is discarded, and no test asserts it for these controls). Step 7
  mutates the guard to prove the tests fail without it.
- **N4 -- merge survivor.** Task 4, `tests/test_collision_merge.py`, chosen over
  `tests/test_edits.py` because that file already owns merge outcomes and has the
  two-body `World` helper. Two tests: the survivor keeps its own locks while the merge
  ignores them (mass grows, temperature mixes), and the survivor does not inherit the
  victim's. Explicitly flagged as test-only, with an instruction **not** to "fix"
  `collisions.py` if it fails. The file's top-level import convention is matched, not
  `test_tools_edit.py`'s.
- **N5 -- restart.** Task 5. One sentence appended to the existing property-lock
  invariant paragraph in `CLAUDE.md` (not a new bullet -- it belongs with "nothing
  inherits them"), naming `F5` and giving the mechanism (`world.clear()` and a rebuild).
  One test in `tests/test_state.py`, beside the existing `test_reset_restores_everything`,
  which asserts both that no lock survives *and* that no `Body` object survives -- the
  second is what would catch a future builder that reused bodies.
- **N6 -- not a defect.** Called out in the header table and in a section of its own,
  with the exact pre-existing line numbers (`tests/test_tools_edit.py:36`, `:54`, `:70`,
  `:109`) that disprove it, and an explicit "do not change any import in any test file".

**What this plan deliberately does not do.** It does not touch `src/edits.py`,
`src/body.py`, `src/collisions.py`, `src/thermal.py`, `src/render/widgets.py` or
`src/render/hud.py`. It adds no control, no hotkey and no `KEY_HELP` row. It does not
revisit any decision the design settled (a lock guards only its own field; a merge and
`collapse` are not edits; `thermal.advance` is the one exception).

**Ordering.** Tasks 1 and 2 both edit `src/app/tools.py` within twenty lines of each
other but in different methods, and Task 3 edits a third and fourth region of the same
file; done in order with a commit each, none of them conflicts. Tasks 4 and 5 touch no
source at all and could run in any order. Task 6 must be last -- it counts the tests the
other five added.

**Types.** `_pick_row(state, field) -> bool`, matching `_collapse`, `_toggle_lock` and
`_unlock_all`, all of which `perform` and `on_event` consume as bools. `field` is always
one of the bare strings `edits.edit_fields` returns, which is what `edit.field` and
`edit.lock` carry as `payload` and what `edit.step` carries as `payload[0]`. The tooltip
in Task 1 is built with a single `%`-format over a two-element tuple, so the
`"Unlock" if locked else "Lock"` expression must be parenthesised inside the tuple as
written.
