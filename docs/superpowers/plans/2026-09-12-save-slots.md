# Save Slots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three save slots, a New / Load / Quit start menu, and an in-game Saves dialog that saves, loads and renames.

**Architecture:** Serialisation is a pure physics-layer module (`src/save.py`: world + a `Session` record ↔ plain dict). File handling is an app-layer `SlotStore` (`src/app/saves.py`). Dialog state and every action live on `AppState`; `ui.py` lays the dialogs out from that state like any other panel, using one new control flavour (a text field) and one new icon. While a dialog is up, `input.py` routes every event to it and nothing else.

**Tech Stack:** Python 3.11, numpy, pygame 2.6, pytest. JSON files.

**Spec:** `docs/superpowers/specs/2026-09-12-save-slots-design.md`

## Global Constraints

- `src/*.py` (including the new `src/save.py`) **must never import pygame**; `tests/test_layering.py` runs a subprocess to assert it.
- Any test that touches pygame sets `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")` at module top, before importing pygame.
- Never hold an array index across a frame; store `body.id`, resolve with `World.index_of`.
- Every mutation of mass, radius or membership is followed by `Simulation.refresh()`.
- No pixel constants in the interface: every size comes from `widgets.Metrics` or the font.
- Every action has a button, and every button carries its hotkey. New documented key: `S` only. `N`, `L`, `Q`, `Enter`, `Esc` are not added to `hud.KEY_HELP`.
- Exactly three slots: `SLOTS = 3`. Names are at most `NAME_MAX = 24` printable ASCII characters.
- Save files: `saves/slot1.json` … `saves/slot3.json` at the repository root, written via `slotN.json.tmp` + `os.replace`. `saves/` is gitignored.
- Loading a save does **not** call `thermal.settle`. It does call `sim.refresh()`.
- Camera zoom is saved as `au_across = camera.width / camera.zoom`, never as `log_zoom`.
- Run the full suite with `python selftest.py` — it takes ~4 minutes, so use a Bash timeout of at least 400000 ms. Single files: `python -m pytest tests/test_x.py -q`.
- Commit after every task. Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01FV8ZQSwAFEVGapwMH9gxZN
  ```

---

## File map

| File | Responsibility |
|---|---|
| `src/save.py` (new) | `VERSION`, `SaveFormatError`, `Session`, `to_data`, `from_data`. Pure data. |
| `src/world.py` | `next_id` property and `rekey(next_id)` so the loader never touches privates. |
| `src/app/saves.py` (new) | `SLOTS`, `SlotInfo`, `default_directory`, `SlotStore` (list/read/write/rename/writable). |
| `src/app/state.py` | `DialogState`, dialog constants, every save/load/rename/menu action. |
| `src/render/effects.py` | `EffectField.clear()`. |
| `src/render/widgets.py` | `Control.text_field`, `draw_text_field`, the `save` icon. |
| `src/render/hud.py` | `KEY_HELP` gains `S`. |
| `src/app/ui.py` | `game.saves` button, `_build_dialog` / `_build_start` / `_build_slots`, `perform` branches, dialog mouse rules, veil + labels drawing. |
| `src/app/input.py` | dialog routing, `dialog_key`, global `S`. |
| `src/app/main.py` | `App(saves_dir=...)`, start paused on the menu, skip HUD/overlay while a dialog is up. |
| `.gitignore` | `saves/`. |
| `CLAUDE.md` | document the new rules. |
| Tests | `tests/test_save.py`, `tests/test_saves_store.py`, additions to `test_state.py`, `test_widgets.py`, `test_ui.py`, `test_layout.py`, `test_smoke.py`, `test_layering.py`. |

---

### Task 1: `src/save.py` — world and session to plain data and back

**Files:**
- Create: `src/save.py`
- Modify: `src/world.py` (add `next_id` property and `rekey`)
- Modify: `tests/test_layering.py:12-16` (add `"src.save"` to `PHYSICS_MODULES`)
- Test: `tests/test_save.py`

**Interfaces:**
- Consumes: `World.add(name, material, mass, radius, pos, vel, kind, seed, temperature)`, `World.clear()`, `Body` fields, `maneuver.Maneuver(due_time, prograde, radial)`, `materials.BY_KEY`, `materials.ROCK`.
- Produces:
  - `save.VERSION = 1`
  - `class save.SaveFormatError(ValueError)`
  - `@dataclass(frozen=True) class save.Session(scenario: str, time_index: int, paused: bool, show_trails: bool, real_size: bool, show_labels: bool, explain_enabled: bool, show_body_list: bool, selected_id, centre: tuple, follow_id, au_across: float)`
  - `save.to_data(world, session, name, scenario_label, now=None) -> dict`
  - `save.from_data(data, world) -> Session` (validates everything before touching `world`; raises `SaveFormatError`)
  - `World.next_id` (property), `World.rekey(next_id)`

- [ ] **Step 1: Add `next_id` and `rekey` to `World`**

In `src/world.py`, after `body_by_id`:

```python
    @property
    def next_id(self):
        """The id the next ``add`` will hand out.  Saved with the world so a
        body added after a load cannot collide with one that was saved."""
        return self._next_id

    def rekey(self, next_id):
        """Rebuild the id index from the bodies' own ids, and set the counter.

        For the loader: it rewrites ``Body.id`` after ``add`` so a saved
        selection still names the same planet, and the index built by ``add``
        is then wrong.  Raises ValueError on a duplicate id, before anything
        is changed.
        """
        ids = [body.id for body in self.bodies]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate body id")
        self._by_id = {body.id: i for i, body in enumerate(self.bodies)}
        self._next_id = max([int(next_id)] + [i + 1 for i in ids])
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_save.py`:

```python
"""A world goes to plain data and comes back the same, to the last bit."""

import json

import numpy as np
import pytest

from src import edits, materials, save, scenarios, units
from src.body import KIND_BLACK_HOLE, KIND_CRAFT, KIND_STAR
from src.maneuver import Maneuver
from src.world import World


def session(**over):
    base = dict(scenario="solar_system", time_index=5, paused=False,
                show_trails=True, real_size=False, show_labels=True,
                explain_enabled=True, show_body_list=True, selected_id=None,
                centre=(0.25, -1.5), follow_id=None, au_across=13.0)
    base.update(over)
    return save.Session(**base)


def busy_world():
    """Everything a save has to carry: a hole with spin, a craft with a plan,
    a locked planet, rings, a body with collision immunity."""
    world = scenarios.solar_system()
    sun = world.bodies[0]
    assert sun.kind == KIND_STAR
    edits.collapse(world, 0)
    sun.spin, sun.disk = 0.7, 0.6
    earth = next(i for i, b in enumerate(world.bodies) if b.name == "Earth")
    world.bodies[earth].locked_fields = frozenset({"mass", "temp"})
    world.temperature[earth] = 1234.5
    world.bodies[earth].event_id = 7
    world.bodies[earth].no_collide_until = world.time + 3.0
    saturn = next(b for b in world.bodies if b.name == "Saturn")
    assert saturn.rings
    craft = world.add("Shuttle", materials.SPACECRAFT, units.EARTH_MASS * 1e-18,
                      units.CRAFT_RADIUS, world.pos[earth] + (0.01, 0.0),
                      world.vel[earth] + (0.0, 0.001), kind=KIND_CRAFT,
                      temperature=250.0)
    craft.fuel_dv = 0.0123
    craft.plan = (Maneuver(10.0, prograde=0.001), Maneuver(20.0, radial=-0.0005))
    world.time = 812.5
    return world


def roundtrip(world, sess=None):
    data = json.loads(json.dumps(save.to_data(world, sess or session(),
                                               "Test", "Solar")))
    target = World()
    got = save.from_data(data, target)
    return data, target, got


def test_every_array_survives_to_the_last_bit():
    world = busy_world()
    _, target, _ = roundtrip(world)
    n = world.n
    assert target.n == n
    for name in ("pos", "vel", "mass", "radius", "temperature"):
        assert np.array_equal(getattr(world, name)[:n], getattr(target, name)[:n]), name
    assert np.array_equal(target.contact_radius[:n], target.radius[:n])
    assert target.time == world.time


def test_every_body_field_survives():
    world = busy_world()
    _, target, _ = roundtrip(world)
    for a, b in zip(world.bodies, target.bodies):
        assert (a.id, a.name, a.kind, a.material.key) == (b.id, b.name, b.kind, b.material.key)
        assert (a.born_day, a.event_id, a.no_collide_until, a.seed) == \
            (b.born_day, b.event_id, b.no_collide_until, b.seed)
        assert (a.rings, a.spin, a.disk, a.fuel_dv) == (b.rings, b.spin, b.disk, b.fuel_dv)
        assert a.locked_fields == b.locked_fields
        assert a.plan == b.plan
        assert isinstance(b.locked_fields, frozenset)
        assert isinstance(b.plan, tuple)
    hole = target.bodies[0]
    assert hole.kind == KIND_BLACK_HOLE and hole.spin == 0.7
    craft = target.bodies[-1]
    assert craft.kind == KIND_CRAFT and craft.plan[1].radial == -0.0005


def test_ids_and_the_counter_are_preserved():
    world = busy_world()
    _, target, _ = roundtrip(world)
    for body in world.bodies:
        assert target.index_of(body.id) is not None
        assert target.body_by_id(body.id).name == body.name
    assert target.next_id == world.next_id
    fresh = target.add("New", materials.ROCK, 1e-6, 1e-5, (5.0, 0.0), (0.0, 0.01))
    assert fresh.id not in {b.id for b in world.bodies}


def test_the_session_comes_back_and_the_camera_targets_resolve():
    world = busy_world()
    earth_id = next(b.id for b in world.bodies if b.name == "Earth")
    _, _, got = roundtrip(world, session(selected_id=earth_id, follow_id=earth_id,
                                         paused=True, time_index=8))
    assert got.selected_id == earth_id and got.follow_id == earth_id
    assert got.paused is True and got.time_index == 8
    assert got.centre == (0.25, -1.5) and got.au_across == 13.0


def test_a_camera_target_that_is_not_in_the_file_loads_as_none():
    world = busy_world()
    _, _, got = roundtrip(world, session(selected_id=999999, follow_id=999999))
    assert got.selected_id is None and got.follow_id is None


def test_the_data_is_plain_json_with_a_header():
    world = busy_world()
    from datetime import datetime
    data = save.to_data(world, session(), "Mars trip", "Solar",
                        now=datetime(2026, 9, 12, 14, 3))
    text = json.dumps(data)          # raises on any numpy scalar
    assert data["version"] == save.VERSION
    assert data["name"] == "Mars trip"
    assert data["saved_at"] == "2026-09-12T14:03"
    assert data["summary"] == {"scenario_label": "Solar", "days": 812.5,
                               "bodies": world.n}
    assert "numpy" not in text


def test_an_unknown_version_raises():
    world = busy_world()
    data = save.to_data(world, session(), "x", "Solar")
    data["version"] = 99
    with pytest.raises(save.SaveFormatError):
        save.from_data(data, World())


def test_an_unknown_kind_raises_and_leaves_the_target_untouched():
    world = busy_world()
    data = save.to_data(world, session(), "x", "Solar")
    data["world"]["bodies"][3]["kind"] = "teapot"
    target = scenarios.solar_system()
    before = target.pos[:target.n].copy()
    n = target.n
    with pytest.raises(save.SaveFormatError):
        save.from_data(data, target)
    assert target.n == n
    assert np.array_equal(target.pos[:n], before)


def test_an_unknown_material_falls_back_to_rock():
    world = busy_world()
    data = save.to_data(world, session(), "x", "Solar")
    data["world"]["bodies"][3]["material"] = "unobtainium"
    target = World()
    save.from_data(data, target)
    assert target.bodies[3].material is materials.ROCK


def test_too_many_bodies_raises():
    world = busy_world()
    data = save.to_data(world, session(), "x", "Solar")
    rec = data["world"]["bodies"][3]
    data["world"]["bodies"] = [dict(rec, id=1000 + k) for k in range(World().capacity + 1)]
    with pytest.raises(save.SaveFormatError):
        save.from_data(data, World())


def test_a_duplicate_id_raises():
    world = busy_world()
    data = save.to_data(world, session(), "x", "Solar")
    data["world"]["bodies"][2]["id"] = data["world"]["bodies"][1]["id"]
    with pytest.raises(save.SaveFormatError):
        save.from_data(data, World())


def test_garbage_raises_rather_than_crashing():
    for garbage in (None, [], {}, {"version": 1}, {"version": 1, "world": 3, "session": {}}):
        with pytest.raises(save.SaveFormatError):
            save.from_data(garbage, World())
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_save.py -q`
Expected: FAIL — `ImportError: cannot import name 'save'`.

- [ ] **Step 3b: Name the fallback material**

There is no module-level `ROCK` in `src/materials.py`. Add one directly after the `BY_KEY[SPACECRAFT.key] = SPACECRAFT` line:

```python
#: What a save names when its material key is one this build no longer has.
ROCK = BY_KEY["rock"]
```

- [ ] **Step 4: Write `src/save.py`**

```python
"""A world and a session as plain data, and back again.

The physics layer's half of saving: numpy arrays and Body objects in, a dict
of lists, floats, strings and bools out, and the reverse.  Nothing here
touches a file -- ``app/saves.py`` does that -- and nothing here imports
pygame; ``tests/test_layering.py`` asserts it.

**``from_data`` validates everything before it touches the world.**  A load
that fails halfway would leave the child with half a solar system, so every
record is parsed into a checked tuple first and the world is only cleared
once all of them are good.

**Kind is loaded as saved, never re-derived from mass.**  A black hole and a
craft are one-way kinds (``edits._refresh_kind``) that mass alone cannot
reproduce.

**Temperature is loaded as saved and not settled.**  It is state, and a
planet he heated on purpose must still be hot when he comes back.
"""

from dataclasses import asdict, dataclass, replace
from datetime import datetime

from . import materials
from .body import KIND_BLACK_HOLE, KIND_CRAFT, KIND_DEBRIS, KIND_PLANET, KIND_STAR
from .maneuver import Maneuver

#: Bump when the file layout changes.  A file with any other version is
#: refused rather than guessed at.
VERSION = 1

KINDS = frozenset((KIND_PLANET, KIND_STAR, KIND_DEBRIS, KIND_BLACK_HOLE, KIND_CRAFT))


class SaveFormatError(ValueError):
    """The data is not a save this build can read."""


@dataclass(frozen=True)
class Session:
    """Everything the application saves that is not the world.

    Defined here rather than in ``app/state.py`` so the physics layer never
    has to see ``AppState``.  ``au_across`` is the camera's view width in AU
    -- never ``log_zoom``, which is pixels per AU and would show a different
    stretch of sky in a different window.
    """

    scenario: str
    time_index: int
    paused: bool
    show_trails: bool
    real_size: bool
    show_labels: bool
    explain_enabled: bool
    show_body_list: bool
    selected_id: object
    centre: tuple
    follow_id: object
    au_across: float


# --- writing ------------------------------------------------------------------------

def to_data(world, session, name, scenario_label, now=None):
    """The whole save as a JSON-able dict: header, world, session."""
    bodies = []
    for i in range(world.n):
        b = world.bodies[i]
        bodies.append({
            "id": int(b.id),
            "name": str(b.name),
            "kind": b.kind,
            "material": b.material.key,
            "mass": float(world.mass[i]),
            "radius": float(world.radius[i]),
            "pos": [float(world.pos[i, 0]), float(world.pos[i, 1])],
            "vel": [float(world.vel[i, 0]), float(world.vel[i, 1])],
            "temperature": float(world.temperature[i]),
            "born_day": float(b.born_day),
            "event_id": int(b.event_id),
            "no_collide_until": float(b.no_collide_until),
            "seed": int(b.seed),
            "rings": bool(b.rings),
            "spin": float(b.spin),
            "disk": float(b.disk),
            "locked_fields": sorted(b.locked_fields),
            "fuel_dv": float(b.fuel_dv),
            "plan": [{"due_time": float(n.due_time), "prograde": float(n.prograde),
                      "radial": float(n.radial)} for n in b.plan],
        })
    when = datetime.now() if now is None else now
    sess = asdict(session)
    sess["centre"] = [float(session.centre[0]), float(session.centre[1])]
    return {
        "version": VERSION,
        "name": str(name),
        "saved_at": when.isoformat(timespec="minutes"),
        "summary": {"scenario_label": str(scenario_label),
                    "days": float(world.time), "bodies": int(world.n)},
        "world": {"time": float(world.time), "next_id": int(world.next_id),
                  "bodies": bodies},
        "session": sess,
    }


# --- reading ------------------------------------------------------------------------

def _opt_int(value):
    return None if value is None else int(value)


def _parse_body(rec):
    """One record into a checked dict of the exact types ``World.add`` and
    ``Body`` want.  Raises SaveFormatError on anything off."""
    try:
        kind = rec["kind"]
        if kind not in KINDS:
            raise SaveFormatError("unknown kind %r" % (kind,))
        material = materials.BY_KEY.get(rec["material"], materials.ROCK)
        pos, vel = rec["pos"], rec["vel"]
        return {
            "id": int(rec["id"]),
            "name": str(rec["name"]),
            "kind": kind,
            "material": material,
            "mass": float(rec["mass"]),
            "radius": float(rec["radius"]),
            "pos": (float(pos[0]), float(pos[1])),
            "vel": (float(vel[0]), float(vel[1])),
            "temperature": float(rec["temperature"]),
            "born_day": float(rec["born_day"]),
            "event_id": int(rec["event_id"]),
            "no_collide_until": float(rec["no_collide_until"]),
            "seed": int(rec["seed"]),
            "rings": bool(rec["rings"]),
            "spin": float(rec["spin"]),
            "disk": float(rec["disk"]),
            "locked_fields": frozenset(str(f) for f in rec["locked_fields"]),
            "fuel_dv": float(rec["fuel_dv"]),
            "plan": tuple(Maneuver(float(n["due_time"]), float(n["prograde"]),
                                   float(n["radial"])) for n in rec["plan"]),
        }
    except SaveFormatError:
        raise
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise SaveFormatError("bad body record: %s" % exc) from exc


def _parse_session(s):
    try:
        centre = s["centre"]
        return Session(
            scenario=str(s["scenario"]),
            time_index=int(s["time_index"]),
            paused=bool(s["paused"]),
            show_trails=bool(s["show_trails"]),
            real_size=bool(s["real_size"]),
            show_labels=bool(s["show_labels"]),
            explain_enabled=bool(s["explain_enabled"]),
            show_body_list=bool(s["show_body_list"]),
            selected_id=_opt_int(s["selected_id"]),
            centre=(float(centre[0]), float(centre[1])),
            follow_id=_opt_int(s["follow_id"]),
            au_across=float(s["au_across"]),
        )
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise SaveFormatError("bad session: %s" % exc) from exc


def from_data(data, world):
    """Replace ``world``'s contents from ``data`` and return the Session.

    Validates first, mutates second: on any SaveFormatError the world is
    exactly as it was.  Does not settle temperature and does not refresh
    the integrator -- the caller owns ``Simulation.refresh()``.
    """
    if not isinstance(data, dict) or data.get("version") != VERSION:
        raise SaveFormatError("not a version %d save" % VERSION)
    try:
        w = data["world"]
        records = w["bodies"]
        time = float(w["time"])
        next_id = int(w["next_id"])
        session_data = data["session"]
        if not isinstance(records, list):
            raise SaveFormatError("bodies is not a list")
    except (KeyError, TypeError, ValueError) as exc:
        raise SaveFormatError("bad save: %s" % exc) from exc
    if len(records) > world.capacity:
        raise SaveFormatError("%d bodies, capacity %d" % (len(records), world.capacity))
    parsed = [_parse_body(rec) for rec in records]
    ids = [p["id"] for p in parsed]
    if len(set(ids)) != len(ids):
        raise SaveFormatError("duplicate body id")
    session = _parse_session(session_data)

    world.clear()
    world.time = time
    for p in parsed:
        body = world.add(p["name"], p["material"], p["mass"], p["radius"],
                         p["pos"], p["vel"], kind=p["kind"], seed=p["seed"],
                         temperature=p["temperature"])
        body.id = p["id"]
        body.born_day = p["born_day"]
        body.event_id = p["event_id"]
        body.no_collide_until = p["no_collide_until"]
        body.rings = p["rings"]
        body.spin = p["spin"]
        body.disk = p["disk"]
        body.locked_fields = p["locked_fields"]
        body.fuel_dv = p["fuel_dv"]
        body.plan = p["plan"]
    world.rekey(next_id)

    present = set(ids)
    return replace(session,
                   selected_id=session.selected_id if session.selected_id in present else None,
                   follow_id=session.follow_id if session.follow_id in present else None)
```

Note `world.time` is set before the `add` loop so `born_day` written by `add` is then overwritten by the saved one — both are set explicitly, so order does not matter for correctness, but `clear()` zeroes `time`, so it must be set *after* `clear()`.

- [ ] **Step 5: Add `src.save` to the layering test**

In `tests/test_layering.py`, `PHYSICS_MODULES` gains `"src.save"` at the end of the list.

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_save.py tests/test_layering.py tests/test_world.py -q`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/save.py src/world.py src/materials.py tests/test_save.py tests/test_layering.py
git commit -m "feat: a world and a session as plain data, and back"
```

---

### Task 2: `src/app/saves.py` — the three slot files

**Files:**
- Create: `src/app/saves.py`
- Modify: `.gitignore`
- Test: `tests/test_saves_store.py`

**Interfaces:**
- Consumes: `save.VERSION`, `save.SaveFormatError`.
- Produces:
  - `saves.SLOTS = 3`
  - `@dataclass(frozen=True) class saves.SlotInfo(slot: int, name: str, saved_at: str, summary: dict, readable: bool)`
  - `saves.UNREADABLE_NAME = "Can't read"`
  - `saves.default_directory() -> str` (the repo-root `saves/`)
  - `class saves.SlotStore(directory)`: `.directory`, `path(slot)`, `list() -> list[SlotInfo | None]` (length 3), `read(slot) -> dict`, `write(slot, data)`, `rename(slot, name)`, `writable() -> bool`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_saves_store.py`:

```python
"""Three slot files on disk: written whole or not at all, listed by header."""

import json
import os

import pytest

from src import save
from src.app import saves


def document(name="Solar Year 1.0", version=save.VERSION):
    return {"version": version, "name": name, "saved_at": "2026-09-12T14:03",
            "summary": {"scenario_label": "Solar", "days": 365.25, "bodies": 9},
            "world": {"time": 365.25, "next_id": 20, "bodies": []},
            "session": {}}


@pytest.fixture
def store(tmp_path):
    return saves.SlotStore(str(tmp_path / "saves"))


def test_there_are_exactly_three_slots(store):
    assert saves.SLOTS == 3
    assert store.list() == [None, None, None]
    with pytest.raises(ValueError):
        store.path(0)
    with pytest.raises(ValueError):
        store.path(4)


def test_the_directory_is_created_on_first_write(store):
    assert not os.path.isdir(store.directory)
    store.write(2, document())
    assert os.path.isdir(store.directory)
    assert os.path.basename(store.path(2)) == "slot2.json"
    assert os.path.exists(store.path(2))


def test_a_write_leaves_no_temporary_file_behind(store):
    store.write(1, document())
    assert sorted(os.listdir(store.directory)) == ["slot1.json"]


def test_listing_reads_the_header(store):
    store.write(3, document(name="Mars trip"))
    listed = store.list()
    assert listed[0] is None and listed[1] is None
    info = listed[2]
    assert info.slot == 3 and info.name == "Mars trip"
    assert info.saved_at == "2026-09-12T14:03"
    assert info.summary["bodies"] == 9
    assert info.readable


def test_a_corrupt_file_lists_as_unreadable_and_does_not_raise(store):
    os.makedirs(store.directory)
    with open(store.path(1), "w") as f:
        f.write("{not json")
    info = store.list()[0]
    assert info is not None and not info.readable
    assert info.name == saves.UNREADABLE_NAME
    with pytest.raises(ValueError):
        store.read(1)


def test_a_wrong_version_lists_as_unreadable(store):
    store.write(1, document(version=99))
    assert not store.list()[0].readable
    with pytest.raises(save.SaveFormatError):
        store.read(1)


def test_rename_changes_only_the_name(store):
    store.write(1, document())
    store.rename(1, "Kaboom")
    data = store.read(1)
    expected = document(name="Kaboom")
    assert data == expected
    assert store.list()[0].name == "Kaboom"


def test_read_returns_what_was_written(store):
    doc = document()
    store.write(2, doc)
    assert store.read(2) == json.loads(json.dumps(doc))


def test_writable_creates_and_reports(store):
    assert store.writable()
    assert os.path.isdir(store.directory)


def test_writable_is_false_where_it_cannot_create(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    store = saves.SlotStore(str(blocker / "saves"))
    assert not store.writable()


def test_default_directory_is_beside_run_command():
    root = saves.default_directory()
    assert os.path.basename(root) == "saves"
    assert os.path.exists(os.path.join(os.path.dirname(root), "run.command"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_saves_store.py -q`
Expected: FAIL — `ImportError: cannot import name 'saves'`.

- [ ] **Step 3: Write `src/app/saves.py`**

```python
"""The three slot files.

``saves/slot1.json`` .. ``slot3.json`` in a ``saves/`` directory beside
``run.command``, so a parent can find them and back them up.  A write goes
to ``slotN.json.tmp`` and is moved into place with ``os.replace``, so a
crash mid-write leaves the old save rather than half of a new one.

Listing reads each file's header -- name, when, a summary -- and never
deserialises the world; a file that cannot be read is listed as unreadable
rather than raised, because the dialog still has to draw the other two.
"""

import json
import os
from dataclasses import dataclass

from .. import save

SLOTS = 3
UNREADABLE_NAME = "Can't read"


@dataclass(frozen=True)
class SlotInfo:
    """What the dialog shows for a filled slot."""

    slot: int
    name: str
    saved_at: str
    summary: dict
    readable: bool


def default_directory():
    """``saves/`` at the repository root: two levels up from this package."""
    here = os.path.dirname(os.path.abspath(__file__))       # src/app
    return os.path.join(os.path.dirname(os.path.dirname(here)), "saves")


class SlotStore:
    def __init__(self, directory):
        self.directory = directory

    def path(self, slot):
        if not 1 <= int(slot) <= SLOTS:
            raise ValueError("slot %r is not 1..%d" % (slot, SLOTS))
        return os.path.join(self.directory, "slot%d.json" % int(slot))

    def writable(self):
        """Can a save land here?  Creates the directory if it can."""
        try:
            os.makedirs(self.directory, exist_ok=True)
        except OSError:
            return False
        return os.access(self.directory, os.W_OK)

    def list(self):
        out = []
        for slot in range(1, SLOTS + 1):
            if not os.path.exists(self.path(slot)):
                out.append(None)
                continue
            try:
                data = self.read(slot)
                out.append(SlotInfo(slot, str(data["name"]), str(data["saved_at"]),
                                    dict(data["summary"]), True))
            except (OSError, ValueError, KeyError, TypeError):
                out.append(SlotInfo(slot, UNREADABLE_NAME, "", {}, False))
        return out

    def read(self, slot):
        """The whole document.  Raises OSError, or a ValueError subclass
        (json's decode error, or SaveFormatError for a version we do not know)."""
        with open(self.path(slot), encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or data.get("version") != save.VERSION:
            raise save.SaveFormatError("not a version %d save" % save.VERSION)
        return data

    def write(self, slot, data):
        os.makedirs(self.directory, exist_ok=True)
        final = self.path(slot)
        tmp = final + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, final)

    def rename(self, slot, name):
        data = self.read(slot)
        data["name"] = str(name)
        self.write(slot, data)
```

- [ ] **Step 4: Add `saves/` to `.gitignore`**

Append a line `saves/` to `.gitignore`.

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_saves_store.py -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/app/saves.py tests/test_saves_store.py .gitignore
git commit -m "feat: three slot files, written whole or not at all"
```

---

### Task 3: `AppState` — dialog state and every save/load/rename action

**Files:**
- Modify: `src/app/state.py`
- Modify: `src/render/effects.py` (add `EffectField.clear`)
- Test: `tests/test_state.py` (append)

**Interfaces:**
- Consumes: `save.Session`, `save.to_data`, `save.from_data`, `save.SaveFormatError`, `saves.SlotStore`, `saves.SlotInfo`, `saves.SLOTS`, `saves.UNREADABLE_NAME`.
- Produces on `src/app/state.py`:
  - constants `DIALOG_START = "start"`, `DIALOG_LOAD = "load"`, `DIALOG_SAVES = "saves"`, `NAME_MAX = 24`
  - `class DialogState`: `kind`, `slots` (list of `SlotInfo | None`), `armed` (slot int or None), `renaming` (slot int or None), `text` (str), `resume_paused` (bool)
  - `AppState.__init__(self, sim, camera, view, store=None)`; attribute `store`; attribute `dialog` (`DialogState | None`)
  - methods: `open_start()`, `start_new()`, `quit_now()`, `open_load()`, `open_saves()`, `toggle_saves()`, `close_dialog()`, `disarm()`, `save_slot(slot) -> bool`, `load_slot(slot) -> bool`, `begin_rename(slot)`, `rename_type(text)`, `rename_backspace()`, `rename_commit()`, `rename_cancel()`, `default_save_name() -> str`, `describe_slot(info) -> str`
- `EffectField.clear()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_state.py`:

```python
# --- saving, loading, the menus ------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    from src.app.saves import SlotStore
    return SlotStore(str(tmp_path / "saves"))


@pytest.fixture
def saving_state(store):
    from src.app.state import AppState
    from src.app.tools import SelectTool
    st = AppState(Simulation(), Camera(1000, 700), ViewSettings(), store=store)
    st.set_tool(SelectTool)
    return st


def test_the_start_menu_freezes_time(saving_state):
    from src.app.state import DIALOG_START
    st = saving_state
    st.open_start()
    assert st.dialog.kind == DIALOG_START and st.sim.paused
    st.start_new()
    assert st.dialog is None and not st.sim.paused


def test_quit_from_the_start_menu_needs_no_confirmation(saving_state):
    saving_state.open_start()
    saving_state.quit_now()
    assert saving_state.running is False


def test_load_from_the_start_menu_and_back(saving_state):
    from src.app.state import DIALOG_LOAD, DIALOG_START
    st = saving_state
    st.open_start()
    st.open_load()
    assert st.dialog.kind == DIALOG_LOAD
    assert st.dialog.slots == [None, None, None]
    st.close_dialog()
    assert st.dialog.kind == DIALOG_START


def test_opening_the_saves_dialog_pauses_and_closing_restores(saving_state):
    st = saving_state
    for was_paused in (False, True):
        st.sim.paused = was_paused
        st.open_saves()
        assert st.sim.paused is True
        st.close_dialog()
        assert st.dialog is None
        assert st.sim.paused is was_paused


def test_toggle_saves_opens_and_closes(saving_state):
    from src.app.state import DIALOG_SAVES
    st = saving_state
    st.toggle_saves()
    assert st.dialog.kind == DIALOG_SAVES
    st.toggle_saves()
    assert st.dialog is None


def test_saving_into_an_empty_slot_takes_one_press(saving_state, store):
    st = saving_state
    st.open_saves()
    assert st.save_slot(2) is True
    assert store.list()[1].name == st.default_save_name()
    assert st.dialog.slots[1] is not None
    assert st.dialog.armed is None


def test_default_name_is_scenario_and_year(saving_state):
    saving_state.sim.world.time = 3.2 * 365.25
    assert saving_state.default_save_name() == "Solar Year 3.2"


def test_saving_over_a_full_slot_takes_two_presses_and_keeps_the_name(saving_state, store):
    st = saving_state
    st.open_saves()
    st.save_slot(1)
    store.rename(1, "Mars trip")
    st.dialog.slots = store.list()
    st.sim.world.time = 999.0
    assert st.save_slot(1) is False
    assert st.dialog.armed == 1
    assert store.read(1)["world"]["time"] != 999.0
    assert st.save_slot(1) is True
    assert st.dialog.armed is None
    assert store.read(1)["world"]["time"] == 999.0
    assert store.read(1)["name"] == "Mars trip"


def test_anything_else_disarms_an_overwrite(saving_state):
    st = saving_state
    st.open_saves()
    st.save_slot(1)
    st.save_slot(1)
    assert st.dialog.armed == 1
    st.disarm()
    assert st.dialog.armed is None
    st.save_slot(1)
    st.save_slot(3)                       # a different slot is "something else"
    assert st.dialog.armed is None
    assert st.dialog.slots[2] is not None


def test_save_over_an_unreadable_slot_uses_the_default_name(saving_state, store):
    import os
    os.makedirs(store.directory, exist_ok=True)
    with open(store.path(1), "w") as f:
        f.write("junk")
    st = saving_state
    st.open_saves()
    assert not st.dialog.slots[0].readable
    st.save_slot(1)
    assert st.save_slot(1) is True
    assert store.list()[0].name == st.default_save_name()


def test_without_a_store_nothing_is_saved(state):
    state.open_saves()
    assert state.dialog.slots == [None, None, None]
    assert state.save_slot(1) is False
    assert state.dialog.slots == [None, None, None]


def test_loading_restores_the_world_the_selection_and_the_camera(saving_state, store):
    st = saving_state
    world = st.sim.world
    earth_id = next(b.id for b in world.bodies if b.name == "Earth")
    st.select(earth_id)
    st.camera.follow_id = earth_id
    st.camera.centre[:] = (1.0, 2.0)
    st.camera.log_zoom = np.log(1000 / 4.0)          # 4 AU across
    st.time_index = 8
    st.view.show_trails = False
    st.sim.paused = True
    st.open_saves()
    st.save_slot(1)
    st.close_dialog()

    st.reset()
    st.sim.paused = False
    st.time_index = 2
    st.view.show_trails = True
    st.camera.log_zoom = np.log(1000 / 50.0)
    st.sim.world.time = 5000.0
    st.open_saves()
    assert st.load_slot(1) is True
    assert st.dialog is None
    assert st.sim.paused is True
    assert st.time_index == 8 and st.view.show_trails is False
    assert st.selected_id == earth_id and st.camera.follow_id == earth_id
    assert st.selected_body.name == "Earth"
    assert tuple(st.camera.centre) == (1.0, 2.0)
    assert abs(st.camera.width / st.camera.zoom - 4.0) < 1e-9
    assert st.sim.world.time != 5000.0


def test_the_same_save_shows_the_same_width_of_sky_in_a_narrower_window(saving_state, store):
    from src.app.state import AppState
    st = saving_state
    st.camera.log_zoom = np.log(1000 / 4.0)
    st.open_saves()
    st.save_slot(1)
    narrow = AppState(Simulation(), Camera(400, 300), ViewSettings(), store=store)
    narrow.open_saves()
    assert narrow.load_slot(1)
    assert abs(narrow.camera.width / narrow.camera.zoom - 4.0) < 1e-9


def test_loading_refreshes_the_integrator_and_drops_the_stale_report(saving_state):
    st = saving_state
    st.open_saves()
    st.save_slot(1)
    st.explain.report = object()
    st.load_slot(1)
    assert st.explain.report is None
    st.sim.advance(1.0)                      # would blow up on stale accelerations


def test_a_slot_that_cannot_be_loaded_is_marked_and_the_world_survives(saving_state, store):
    import os
    st = saving_state
    st.open_saves()
    st.save_slot(1)
    data = store.read(1)
    data["world"]["bodies"][0]["kind"] = "teapot"
    store.write(1, data)
    st.dialog.slots = store.list()           # still lists as readable: header is fine
    n = st.sim.world.n
    assert st.load_slot(1) is False
    assert st.dialog is not None
    assert not st.dialog.slots[0].readable
    assert st.sim.world.n == n


def test_rename_commit_writes_and_cancel_does_not(saving_state, store):
    st = saving_state
    st.open_saves()
    st.save_slot(2)
    st.begin_rename(2)
    assert st.dialog.renaming == 2 and st.dialog.text == st.default_save_name()
    for ch in "!!":
        st.rename_type(ch)
    st.rename_backspace()
    st.rename_commit()
    assert st.dialog.renaming is None
    assert store.list()[1].name == st.default_save_name() + "!"
    st.begin_rename(2)
    st.rename_type("z")
    st.rename_cancel()
    assert store.list()[1].name == st.default_save_name() + "!"


def test_an_empty_name_keeps_the_old_one(saving_state, store):
    st = saving_state
    st.open_saves()
    st.save_slot(1)
    st.begin_rename(1)
    while st.dialog.text:
        st.rename_backspace()
    st.rename_type(" ")
    st.rename_commit()
    assert store.list()[0].name == st.default_save_name()


def test_names_are_capped_and_printable_ascii_only(saving_state):
    from src.app.state import NAME_MAX
    st = saving_state
    st.open_saves()
    st.save_slot(1)
    st.begin_rename(1)
    while st.dialog.text:
        st.rename_backspace()
    for _ in range(NAME_MAX + 10):
        st.rename_type("a")
    assert len(st.dialog.text) == NAME_MAX
    st.rename_backspace()
    st.rename_type("\t")
    st.rename_type("é")
    st.rename_type("\x1b")
    assert st.dialog.text == "a" * (NAME_MAX - 1)


def test_rename_and_arm_cancel_each_other(saving_state):
    st = saving_state
    st.open_saves()
    st.save_slot(1)
    st.save_slot(1)                          # armed
    st.begin_rename(1)
    assert st.dialog.armed is None and st.dialog.renaming == 1
    st.save_slot(1)                          # arms again, and leaves the field
    assert st.dialog.renaming is None and st.dialog.armed == 1


def test_describe_slot_reads_well(saving_state, store):
    from src.app.saves import SlotInfo, UNREADABLE_NAME
    st = saving_state
    assert st.describe_slot(None) == "Nothing saved here yet"
    bad = SlotInfo(1, UNREADABLE_NAME, "", {}, False)
    assert "can't be read" in st.describe_slot(bad)
    good = SlotInfo(1, "x", "2026-09-12T14:03",
                    {"scenario_label": "Solar", "days": 3.2 * 365.25, "bodies": 9}, True)
    assert st.describe_slot(good) == "Solar  Year 3.2  9 bodies  Sat 12 Sep 14:03"
    odd = SlotInfo(1, "x", "yesterday", {"scenario_label": "Solar", "days": 0, "bodies": 1}, True)
    assert st.describe_slot(odd).endswith("yesterday")


def test_restart_closes_nothing_but_loading_resets_the_tool(saving_state):
    from src.app.tools import FlyTool
    st = saving_state
    st.set_tool(FlyTool)
    st.open_saves()
    st.save_slot(1)
    old_tool = st.tool
    st.load_slot(1)
    assert isinstance(st.tool, FlyTool) and st.tool is not old_tool
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_state.py -q -k "start_menu or saves or slot or rename or describe or loading"`
Expected: FAIL — `TypeError: AppState.__init__() got an unexpected keyword argument 'store'` and `AttributeError`s.

- [ ] **Step 3: Add `EffectField.clear`**

In `src/render/effects.py`, inside `class EffectField` after `count`:

```python
    def clear(self):
        """Drop every particle.  A loaded world has no impact to show."""
        self.particles = []
```

- [ ] **Step 4: Implement the state**

In `src/app/state.py`:

Imports — add `from datetime import datetime`, `import numpy as np`, and:

```python
from .. import explain, save, units
from ..render.camera import MAX_ZOOM, MIN_ZOOM
from .saves import SLOTS, SlotInfo, UNREADABLE_NAME
```

Constants, after `SETTLE_MS`:

```python
#: Which modal is up.  ``"start"`` is the opening New / Load / Quit menu,
#: ``"load"`` the slot list reached from it (load only, Back), ``"saves"``
#: the in-game list with Load, Save and Rename per slot.
DIALOG_START = "start"
DIALOG_LOAD = "load"
DIALOG_SAVES = "saves"

#: Longest name a slot can have.  Wide enough for "Jupiter eats everything",
#: narrow enough that the name row still fits beside three buttons at the
#: smallest window (tests/test_layout.py).
NAME_MAX = 24


class DialogState:
    """What the open dialog knows: which one it is, the listed slots, and the
    two things a slot can be in the middle of -- an armed overwrite, or a
    name being typed.  Only one slot can be in either state at once, and
    starting one ends the other."""

    def __init__(self, kind, slots):
        self.kind = kind
        self.slots = slots
        #: Slot whose Save button reads "Overwrite?" and is waiting for the
        #: second press, or None.
        self.armed = None
        #: Slot whose name is a text field right now, or None.
        self.renaming = None
        self.text = ""
        #: Whether the sim was paused before the in-game dialog opened, so
        #: closing it puts time back how it was.
        self.resume_paused = False
```

`AppState.__init__` signature becomes `def __init__(self, sim, camera, view, store=None)` and gains, after `self.scenario = SCENARIOS[0]`:

```python
        #: Where saves go, or None when there is nowhere (tests).  With no
        #: store every Save button is disabled and every slot lists empty.
        self.store = store
        #: The open modal, or None.  While one is up, ``ui.build`` lays out
        #: only it and ``input.handle_event`` routes everything to it.
        self.dialog = None
```

New section at the end of the class, before `cycle_grid`:

```python
    # --- the start menu and the saves dialog ------------------------------------

    def _list_slots(self):
        return [None] * SLOTS if self.store is None else self.store.list()

    def open_start(self):
        """The opening menu.  Time is frozen behind it until New or Load."""
        self.dialog = DialogState(DIALOG_START, [])
        self.sim.paused = True

    def start_new(self):
        self.dialog = None
        self.sim.paused = False

    def quit_now(self):
        """Quit from the start menu: nothing has happened yet, so no arming."""
        self.running = False

    def open_load(self):
        self.dialog = DialogState(DIALOG_LOAD, self._list_slots())

    def open_saves(self):
        """The in-game list.  Pauses, remembering whether it was, so what he
        saves is exactly what is on screen."""
        if self.dialog is not None:
            return
        d = DialogState(DIALOG_SAVES, self._list_slots())
        d.resume_paused = self.sim.paused
        self.sim.paused = True
        self.dialog = d
        self.cancel_quit()

    def toggle_saves(self):
        """The S key and the disk button: open the list, or put it away."""
        if self.dialog is None:
            self.open_saves()
        elif self.dialog.kind == DIALOG_SAVES:
            self.close_dialog()

    def close_dialog(self):
        """Esc and the close button.  The in-game list restores the pause
        state it found; the load list goes back to the start menu; the start
        menu has no way out but its own three buttons."""
        d = self.dialog
        if d is None:
            return
        self.rename_cancel()
        if d.kind == DIALOG_SAVES:
            self.sim.paused = d.resume_paused
            self.dialog = None
        elif d.kind == DIALOG_LOAD:
            self.open_start()

    def disarm(self):
        """Any press that is not the second press is an answer of "no"."""
        if self.dialog is not None:
            self.dialog.armed = None

    def default_save_name(self):
        years = self.sim.world.time / units.DAYS_PER_YEAR
        return "%s Year %.1f" % (self.scenario_label, years)

    def _session(self):
        camera = self.camera
        return save.Session(
            scenario=self.scenario, time_index=self.time_index,
            paused=self.sim.paused, show_trails=self.view.show_trails,
            real_size=self.view.real_size, show_labels=self.view.show_labels,
            explain_enabled=self.explain.enabled,
            show_body_list=self.show_body_list, selected_id=self.selected_id,
            centre=(float(camera.centre[0]), float(camera.centre[1])),
            follow_id=camera.follow_id,
            au_across=camera.width / camera.zoom)

    def _apply_session(self, session):
        camera = self.camera
        if session.scenario in SCENARIOS:
            self.scenario = session.scenario
        self.time_index = max(0, min(len(TIME_SCALES) - 1, session.time_index))
        self.view.show_trails = session.show_trails
        self.view.real_size = session.real_size
        self.view.show_labels = session.show_labels
        self.explain.enabled = session.explain_enabled
        self.show_body_list = session.show_body_list
        self.selected_id = session.selected_id
        self.inspector_expanded = False
        camera.centre[:] = session.centre
        camera.follow_id = session.follow_id
        au_across = max(session.au_across, 1e-12)
        camera.log_zoom = min(max(math.log(camera.width / au_across),
                                  math.log(MIN_ZOOM)), math.log(MAX_ZOOM))

    def save_slot(self, slot):
        """Save into ``slot``.  Returns True only when a file was written.

        An empty slot takes one press.  A full one -- readable or not --
        needs a second press on the same slot while it is armed, and the
        name it had is kept: the name is his label for that game.
        """
        d = self.dialog
        if d is None or d.kind != DIALOG_SAVES or self.store is None:
            return False
        self.rename_cancel()
        info = d.slots[slot - 1]
        if info is not None and d.armed != slot:
            d.armed = slot
            return False
        d.armed = None
        name = info.name if (info is not None and info.readable) \
            else self.default_save_name()
        data = save.to_data(self.sim.world, self._session(), name, self.scenario_label)
        try:
            self.store.write(slot, data)
        except OSError:
            return False
        d.slots = self.store.list()
        return True

    def load_slot(self, slot):
        """Replace the running world with ``slot``.  Returns True on success.

        A file the loader refuses is marked unreadable in the list and the
        running world is left exactly as it was -- ``save.from_data``
        validates before it touches anything.
        """
        d = self.dialog
        if d is None or self.store is None:
            return False
        info = d.slots[slot - 1]
        if info is None or not info.readable:
            return False
        self.rename_cancel()
        d.armed = None
        try:
            data = self.store.read(slot)
            session = save.from_data(data, self.sim.world)
        except (OSError, ValueError):
            d.slots[slot - 1] = SlotInfo(slot, UNREADABLE_NAME, "", {}, False)
            return False
        self.sim.refresh()
        self._apply_session(session)
        # Loading is a restart as far as everything that remembers a body
        # goes: the report names bodies that no longer exist, the effects are
        # someone else's impacts, and a tool's target id could collide with
        # a loaded id by accident.
        self.explain.clear()
        self.explain_rect = None
        self.effects.clear()
        self.body_list_offset = 0
        self.quit_armed = False
        if self.tool is not None:
            self.set_tool(type(self.tool))
        self.dialog = None
        self.sim.paused = session.paused
        return True

    def begin_rename(self, slot):
        d = self.dialog
        if d is None or self.store is None:
            return
        info = d.slots[slot - 1]
        if info is None or not info.readable:
            return
        d.armed = None
        d.renaming = slot
        d.text = info.name

    def rename_type(self, text):
        """Append typed characters: printable ASCII, up to NAME_MAX."""
        d = self.dialog
        if d is None or d.renaming is None:
            return
        for ch in text:
            if 32 <= ord(ch) <= 126 and len(d.text) < NAME_MAX:
                d.text += ch

    def rename_backspace(self):
        d = self.dialog
        if d is not None and d.renaming is not None:
            d.text = d.text[:-1]

    def rename_commit(self):
        """Enter.  Writes the name; an empty one keeps the old name."""
        d = self.dialog
        if d is None or d.renaming is None:
            return
        slot, d.renaming = d.renaming, None
        info = d.slots[slot - 1]
        name = d.text.strip() or (info.name if info is not None else self.default_save_name())
        d.text = ""
        try:
            self.store.rename(slot, name[:NAME_MAX])
        except (OSError, ValueError):
            return
        d.slots = self.store.list()

    def rename_cancel(self):
        d = self.dialog
        if d is not None:
            d.renaming = None
            d.text = ""

    def describe_slot(self, info):
        """The dim second line under a slot's name."""
        if info is None:
            return "Nothing saved here yet"
        if not info.readable:
            return "This file can't be read -- saving over it will fix it"
        summary = info.summary
        try:
            when = datetime.fromisoformat(info.saved_at).strftime("%a %d %b %H:%M")
        except (TypeError, ValueError):
            when = info.saved_at
        years = float(summary.get("days", 0.0)) / units.DAYS_PER_YEAR
        return "%s  Year %.1f  %d bodies  %s" % (
            summary.get("scenario_label", "?"), years, int(summary.get("bodies", 0)), when)
```

Check that `math` is already imported at the top of `state.py` (it is: `import math`). `MIN_ZOOM`/`MAX_ZOOM` are module constants in `src/render/camera.py`.

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_state.py tests/test_effects.py -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/app/state.py src/render/effects.py tests/test_state.py
git commit -m "feat: dialog state, and every save, load and rename action"
```

---

### Task 4: Widgets — a text field and a disk icon

**Files:**
- Modify: `src/render/widgets.py`
- Test: `tests/test_widgets.py` (append)

**Interfaces:**
- Produces: `Control.text_field: bool = False` (a `text_field` control is clickable and draws as a focused input); `widgets.draw_text_field(surface, font, control, rect, now_ms=None)`; icon name `"save"` in `draw_icon`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_widgets.py`:

```python
def test_a_text_field_draws_its_text_and_a_cursor(font):
    from src.render import palette, widgets
    surface = pygame.Surface((200, 40))
    surface.fill(palette.SPACE)
    control = widgets.Control("slot.name", label="Mars trip", text_field=True,
                              rect=pygame.Rect(4, 4, 190, 30))
    widgets.draw_button(surface, font, control)
    with_text = _ink(surface)
    assert with_text > 100
    # The focus ring is the selection colour: a text field is the thing on
    # screen you are editing, the same way a selected planet is.
    assert surface.get_at((4, 20))[:3] == palette.SELECT
    empty = widgets.Control("slot.name", label="", text_field=True,
                            rect=pygame.Rect(4, 4, 190, 30))
    surface.fill(palette.SPACE)
    widgets.draw_text_field(surface, font, empty, empty.rect, now_ms=0)
    lit = _ink(surface)
    surface.fill(palette.SPACE)
    widgets.draw_text_field(surface, font, empty, empty.rect, now_ms=500)
    dark = _ink(surface)
    assert lit > dark, "the cursor blinks: on at 0 ms, off at 500 ms"


def test_a_text_field_is_clickable_but_a_readout_is_not():
    from src.render.widgets import Control
    assert Control("x", text_field=True).clickable
    assert not Control("x", readout=True).clickable


def test_the_save_icon_draws_something(font):
    from src.render import palette, widgets
    surface = pygame.Surface((24, 24))
    surface.fill(palette.SPACE)
    widgets.draw_icon(surface, "save", pygame.Rect(2, 2, 20, 20), palette.TEXT)
    assert _ink(surface) > 20
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_widgets.py -q -k "text_field or save_icon"`
Expected: FAIL — `TypeError: Control.__init__() got an unexpected keyword argument 'text_field'`.

- [ ] **Step 3: Implement**

In `src/render/widgets.py`:

Add to `Control`, after `value: float = 0.0`:

```python
    #: A text field is drawn as an input with a cursor and takes keystrokes
    #: (routed by ``app/input.py``); its ``label`` is the text so far.
    text_field: bool = False
```

In `_draw_button_content`, before the `if control.readout:` branch:

```python
    if control.text_field:
        draw_text_field(surface, font, control, rect)
        return
```

After `draw_slider` (or wherever the other `draw_*` helpers sit), add:

```python
#: The cursor's on/off period, ms.  Wall clock: a cursor must blink at the
#: same rate at one hour a second and at ten years a second.
CURSOR_BLINK_MS = 500


def draw_text_field(surface, font, control, rect, now_ms=None):
    """An input box: panel fill, selection-colour ring, the text, a cursor.

    ``now_ms`` is for tests; the frame loop leaves it to the clock.
    """
    pad = max(3, font.height // 3)
    box = pygame.Surface(rect.size, pygame.SRCALPHA)
    box.fill(tuple(palette.PANEL) + (238,))
    surface.blit(box, rect.topleft)
    pygame.draw.rect(surface, palette.SELECT, rect, 1)
    y = rect.y + (rect.height - font.height) // 2
    advance = font.draw(surface, rect.x + pad, y, control.label, palette.TEXT)
    now = pygame.time.get_ticks() if now_ms is None else now_ms
    if (now // CURSOR_BLINK_MS) % 2 == 0:
        cx = min(rect.x + pad + advance + 1, rect.right - 2)
        pygame.draw.line(surface, palette.TEXT, (cx, y), (cx, y + font.height - 1), 1)
```

In `draw_icon`, add a branch (anywhere among the `elif`s, e.g. before `"power"`):

```python
    elif name == "save":
        # A floppy disk: the one icon every child still knows means "save".
        pygame.draw.rect(surface, colour, pygame.Rect(cx - r, cy - r, 2 * r, 2 * r), 1)
        pygame.draw.rect(surface, colour,
                         pygame.Rect(cx - r // 2, cy - r + 1, r, max(1, r // 2)))
        pygame.draw.rect(surface, colour,
                         pygame.Rect(cx - r // 2, cy + 1, r, max(1, r - 1)), 1)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_widgets.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/render/widgets.py tests/test_widgets.py
git commit -m "feat: a text field control and a save icon"
```

---

### Task 5: The interface — Saves button, start menu, slot list

**Files:**
- Modify: `src/app/ui.py`
- Modify: `src/render/hud.py:24-57` (`KEY_HELP`)
- Test: `tests/test_ui.py` (append), `tests/test_layout.py` (append)

**Interfaces:**
- Consumes: everything Task 3 put on `AppState`; `Control.text_field`; icon `"save"`; `saves.SLOTS`.
- Produces control ids: `game.saves` (hotkey `S`), `start.new` (`N`), `start.load` (`L`), `start.quit` (`Q`), `dialog.close`, `slot.name` (payload slot; readout, or text field while renaming), `slot.load`, `slot.save`, `slot.rename` (payload slot). `UiLayer.labels`: list of `(x, y, text, colour)` drawn inside the modal. `ui.build` builds only the dialog while `state.dialog` is set.

- [ ] **Step 1: Write the failing UI tests**

Append to `tests/test_ui.py`:

```python
# --- saving and loading ---------------------------------------------------------------

@pytest.fixture
def saving(tmp_path):
    from src.app.saves import SlotStore
    from src.app.state import AppState
    from src.app.tools import SelectTool
    from src.app.ui import UiLayer
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.render.hud import Hud
    from src.render.pixelfont import PixelFont
    from src.sim import Simulation
    store = SlotStore(str(tmp_path / "saves"))
    st = AppState(Simulation(), Camera(*SIZE), ViewSettings(), store=store)
    st.set_tool(SelectTool)
    font = PixelFont(18)
    layer = UiLayer(font, Hud(font))
    layer.build(st, SIZE)
    return st, layer, store


def test_the_saves_button_carries_the_s_key_and_opens_the_list(saving):
    st, ui, _ = saving
    button = find(ui, "game.saves")
    assert button is not None and button.hotkey == "S"
    click(ui, st, button)
    assert st.dialog is not None and st.dialog.kind == "saves"
    assert st.sim.paused


def test_with_a_dialog_open_nothing_else_is_built(saving):
    st, ui, _ = saving
    st.open_saves()
    ui.build(st, SIZE)
    built = ids(ui)
    assert "dialog.close" in built and "slot.save" in built
    assert not built & {"tool.look", "sim.reset", "app.quit", "game.saves",
                        "body.follow", "view.zoom"}
    assert ui.modal is not None


def test_an_empty_slot_offers_save_only(saving):
    st, ui, _ = saving
    st.open_saves()
    ui.build(st, SIZE)
    assert find(ui, "slot.save", 1) is not None
    assert find(ui, "slot.load", 1) is None
    assert find(ui, "slot.rename", 1) is None
    assert "Empty" in find(ui, "slot.name", 1).label


def test_saving_by_button_fills_the_slot_and_offers_load_and_rename(saving):
    st, ui, store = saving
    st.open_saves()
    ui.build(st, SIZE)
    click_id(ui, st, "slot.save", 2)
    assert store.list()[1] is not None
    assert find(ui, "slot.load", 2) is not None
    assert find(ui, "slot.rename", 2) is not None
    assert st.default_save_name() in find(ui, "slot.name", 2).label


def test_overwrite_reads_as_a_question_and_a_stray_press_disarms(saving):
    st, ui, store = saving
    st.open_saves()
    ui.build(st, SIZE)
    click_id(ui, st, "slot.save", 1)
    click_id(ui, st, "slot.save", 1)
    armed = find(ui, "slot.save", 1)
    assert armed.label == "Overwrite?" and armed.danger and armed.active
    # A press on the panel background, off every button.
    pos = (ui.modal.x + 2, ui.modal.bottom - 2)
    ui.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1), st)
    ui.on_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=1), st)
    ui.build(st, SIZE)
    assert find(ui, "slot.save", 1).label == "Save"
    assert st.dialog is not None, "a stray press disarms, it does not close"


def test_load_by_button_replaces_the_world_and_closes(saving):
    st, ui, _ = saving
    st.open_saves()
    ui.build(st, SIZE)
    click_id(ui, st, "slot.save", 3)
    st.close_dialog()
    st.sim.world.time = 4321.0
    st.open_saves()
    ui.build(st, SIZE)
    click_id(ui, st, "slot.load", 3)
    assert st.dialog is None
    assert st.sim.world.time != 4321.0
    assert "game.saves" in ids(ui)


def test_rename_by_button_turns_the_name_into_a_field(saving):
    st, ui, _ = saving
    st.open_saves()
    ui.build(st, SIZE)
    click_id(ui, st, "slot.save", 1)
    click_id(ui, st, "slot.rename", 1)
    field = find(ui, "slot.name", 1)
    assert field.text_field and field.label == st.default_save_name()


def test_close_button_restores_the_pause_state(saving):
    st, ui, _ = saving
    st.sim.paused = False
    st.open_saves()
    ui.build(st, SIZE)
    click_id(ui, st, "dialog.close")
    assert st.dialog is None and st.sim.paused is False


def test_the_start_menu_has_three_buttons_with_keys(saving):
    st, ui, _ = saving
    st.open_start()
    ui.build(st, SIZE)
    assert {c.hotkey for c in ui.controls if c.id.startswith("start.")} == {"N", "L", "Q"}
    assert ids(ui) == {"start.new", "start.load", "start.quit"}


def test_new_starts_time_and_load_shows_slots_with_back(saving):
    st, ui, _ = saving
    st.open_start()
    ui.build(st, SIZE)
    click_id(ui, st, "start.load")
    assert st.dialog.kind == "load"
    assert "slot.save" not in ids(ui) and "slot.rename" not in ids(ui)
    assert "Back" in find(ui, "dialog.close").tooltip
    click_id(ui, st, "dialog.close")
    assert st.dialog.kind == "start"
    click_id(ui, st, "start.new")
    assert st.dialog is None and not st.sim.paused


def test_quit_from_the_start_menu_is_immediate(saving):
    st, ui, _ = saving
    st.open_start()
    ui.build(st, SIZE)
    click_id(ui, st, "start.quit")
    assert st.running is False


def test_without_a_store_save_is_disabled_and_says_why(state, ui):
    state.open_saves()
    ui.build(state, SIZE)
    button = find(ui, "slot.save", 1)
    assert not button.enabled and "folder" in button.tooltip.lower()


def test_an_unreadable_slot_disables_load_and_rename(saving):
    import os
    st, ui, store = saving
    os.makedirs(store.directory, exist_ok=True)
    with open(store.path(2), "w") as f:
        f.write("junk")
    st.open_saves()
    ui.build(st, SIZE)
    assert not find(ui, "slot.load", 2).enabled
    assert not find(ui, "slot.rename", 2).enabled
    assert find(ui, "slot.save", 2).enabled


def test_the_wheel_does_nothing_under_a_dialog(saving):
    st, ui, _ = saving
    st.show_body_list = True
    st.open_saves()
    ui.build(st, SIZE)
    ui.mouse = ui.modal.center
    assert ui.on_event(pygame.event.Event(pygame.MOUSEWHEEL, y=-3), st)
    assert st.body_list_offset == 0


def test_the_dialog_draws_over_a_veil(saving):
    from src.render import palette
    st, ui, _ = saving
    st.open_saves()
    ui.build(st, SIZE)
    surface = pygame.Surface(SIZE)
    surface.fill(palette.SPACE)
    ui.draw(surface, st)
    corner = surface.get_at((2, 2))[:3]
    assert corner != palette.SPACE[:3], "the world behind the dialog is dimmed"
    inside = sum(1 for x in range(ui.modal.left, ui.modal.right, 3)
                 for y in range(ui.modal.top, ui.modal.bottom, 3)
                 if surface.get_at((x, y))[:3] == palette.TEXT_DIM)
    assert inside > 10, "the slot descriptions are drawn"
```

Append to `tests/test_layout.py`:

```python
@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """One slot at the longest name allowed, one corrupt, one empty."""
    import os
    from src import save, scenarios
    from src.app.saves import SlotStore
    from src.app.state import NAME_MAX
    store = SlotStore(str(tmp_path_factory.mktemp("saves")))
    world = scenarios.solar_system()
    session = save.Session("solar_system", 5, False, True, False, True, True,
                           True, None, (0.0, 0.0), None, 13.0)
    store.write(1, save.to_data(world, session, "W" * NAME_MAX, "Solar"))
    os.makedirs(store.directory, exist_ok=True)
    with open(store.path(2), "w") as f:
        f.write("junk")
    return store


DIALOGS = ("start", "load", "saves")


def build_dialog(size, s, dialog, store):
    from src.app.tools import SelectTool
    font = PixelFont(config.font_px(s))
    hud = Hud(font)
    internal = config.internal_size(size[0], size[1], s)
    state = AppState(Simulation(), Camera(*internal), ViewSettings(), store=store)
    state.set_tool(SelectTool)
    if dialog == "start":
        state.open_start()
    elif dialog == "load":
        state.open_start()
        state.open_load()
    else:
        state.open_saves()
        state.begin_rename(1)
    ui = UiLayer(font, hud)
    ui.build(state, internal, None)
    return ui, pygame.Rect(0, 0, *internal)


@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("s", config.GRID_CHOICES)
@pytest.mark.parametrize("dialog", DIALOGS)
def test_the_dialogs_fit_inside_the_window(window, s, dialog, store):
    if not config.grid_fits(window[0], window[1], s):
        pytest.skip("grid does not fit")
    ui, screen = build_dialog(window, s, dialog, store)
    assert ui.modal is not None and screen.contains(ui.modal)
    outside = sorted({c.id for c in ui.controls if not screen.contains(c.rect)})
    assert not outside, "%s off screen at %s grid %d (%s)" % (outside, screen.size, s, dialog)
    for i, a in enumerate(ui.controls):
        for b in ui.controls[i + 1:]:
            assert not a.rect.colliderect(b.rect), \
                "%s overlaps %s at %s grid %d (%s)" % (a.id, b.id, window, s, dialog)
        assert ui.modal.contains(a.rect), "%s hangs off the dialog" % a.id
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ui.py -q -k "saves or slot or start_menu or dialog or store" ; python -m pytest tests/test_layout.py -q -k dialogs`
Expected: FAIL — `assert button is not None` for `game.saves`, and `AttributeError: 'UiLayer' object has no attribute 'labels'` / assertion on `ui.modal`.

- [ ] **Step 3: Add `S` to the help table**

In `src/render/hud.py`, `KEY_HELP`, insert after `("F5", "start the solar system over"),`:

```python
    ("S", "save or load a game"),
```

- [ ] **Step 4: Implement the interface**

In `src/app/ui.py`:

Imports: add `from .saves import SLOTS` and `from .state import DIALOG_LOAD, DIALOG_SAVES, DIALOG_START` (check for an import cycle: `state.py` does not import `ui.py`, so this is fine).

`UiLayer.__init__`: add after `self.modal = None`:

```python
        #: Whether ``modal`` is the help panel (click-away closes it) or a
        #: dialog (click-away only disarms an overwrite).
        self._modal_is_help = False
        #: (x, y, text, colour) lines drawn inside the modal, after the
        #: panels and before the controls -- the slot descriptions.
        self.labels = []
```

`build`: replace the body after `self.panels = []` with:

```python
        self.labels = []
        self.modal = None
        self._modal_control = None
        self._modal_is_help = False

        if state.dialog is not None:
            # A dialog is the only thing on screen: nothing else is built, so
            # nothing else can be clicked, and the layout test checks only it.
            self._build_dialog(state, m)
            return self.controls

        self._build_bottom_hotbar(state, m, ctx)
        ... (the existing calls, unchanged)
        self._build_help(state, m)
        return self.controls
```

`_build_help`: remove its first three lines (`self.modal = None`, `self._modal_control = None` are now done in `build`) and set `self._modal_is_help = True` right after `self.modal = self.hud.help_rect(...)`.

`_build_view_buttons`: after `x -= m.gap * 3` (just after placing the quit button) insert:

```python
        saves = Control("game.saves", icon="save", hotkey="S", compact=True,
                        tooltip="Save or load a game",
                        active=state.dialog is not None)
        x -= m.icon_btn_w
        self._place([(saves, pygame.Rect(x, y, m.icon_btn_w, m.btn_h))])
        x -= m.gap * 3
```

New methods, after `_build_help`:

```python
    # --- the start menu and the slot list ---------------------------------------

    def _build_dialog(self, state, m):
        if state.dialog.kind == DIALOG_START:
            self._build_start(state, m)
        else:
            self._build_slots(state, m)

    def _centred_panel(self, m, title, inner_w, inner_h):
        """A modal panel sized to its contents and clamped to the window."""
        panel_w = min(m.width - 2 * m.pad, inner_w + 2 * m.pad)
        panel_h = min(m.height - 2 * m.pad, m.line + m.gap + inner_h + 2 * m.pad)
        rect = pygame.Rect((m.width - panel_w) // 2, (m.height - panel_h) // 2,
                           panel_w, panel_h)
        self.modal = self._panel(rect, title)
        return rect

    def _build_start(self, state, m):
        rows = [
            Control("start.new", label="New", hotkey="N",
                    tooltip="Start a fresh solar system"),
            Control("start.load", label="Load", hotkey="L",
                    tooltip="Pick up a saved game"),
            Control("start.quit", label="Quit", hotkey="Q", danger=True,
                    tooltip="Close the sandbox"),
        ]
        btn_w = max(m.button_width(c) for c in rows) + 4 * m.pad
        title = "STAR SANDBOX"
        inner_w = max(btn_w, m.font.width(title)) + 2 * m.pad
        inner_h = len(rows) * (m.btn_h + m.gap) + m.pad
        rect = self._centred_panel(m, title, inner_w, inner_h)
        btn_w = min(btn_w, rect.width - 2 * m.pad)
        x = rect.x + (rect.width - btn_w) // 2
        y = rect.y + m.line + m.gap + m.pad
        for control in rows:
            self._place([(control, pygame.Rect(x, y, btn_w, m.btn_h))])
            y += m.btn_h + m.gap

    def _build_slots(self, state, m):
        """Three rows: name (or the field being typed into), the buttons the
        slot offers, and a dim line saying what is in it."""
        d = state.dialog
        loading = d.kind == DIALOG_LOAD
        can_save = state.store is not None
        char = m.font.width("X")
        from .state import NAME_MAX
        row_h = m.btn_h + m.line + m.gap
        widest_buttons = (m.button_width(Control("x", label="Load"))
                          + m.button_width(Control("x", label="Overwrite?"))
                          + m.button_width(Control("x", label="Rename")) + 2 * m.gap)
        inner_w = char * (NAME_MAX + 3) + 2 * m.pad + m.gap + widest_buttons
        inner_h = SLOTS * (row_h + m.gap) + m.pad
        rect = self._centred_panel(m, "LOAD A GAME" if loading else "SAVED GAMES",
                                   inner_w, inner_h)
        close = Control("dialog.close", icon="close", compact=True,
                        tooltip="Back to the start menu  [Esc]" if loading
                        else "Close  [Esc]")
        self._place([(close, pygame.Rect(rect.right - m.icon_btn_w - m.gap,
                                         rect.y + m.gap, m.icon_btn_w, m.btn_h))])
        x0 = rect.x + m.pad
        inner = rect.width - 2 * m.pad
        y = rect.y + m.line + m.gap + m.pad
        for slot, info in enumerate(d.slots, start=1):
            buttons = []
            if info is not None:
                buttons.append(Control("slot.load", label="Load", payload=slot,
                                       enabled=info.readable,
                                       tooltip="Load this game"))
            if not loading:
                armed = d.armed == slot
                if can_save:
                    tip = ("Click again to save over this game" if armed
                           else "Save the game into this slot")
                else:
                    tip = "No saves folder to write to"
                buttons.append(Control("slot.save",
                                       label="Overwrite?" if armed else "Save",
                                       payload=slot, danger=armed, active=armed,
                                       enabled=can_save, tooltip=tip))
                if info is not None:
                    buttons.append(Control("slot.rename", label="Rename",
                                           payload=slot, enabled=info.readable,
                                           active=d.renaming == slot,
                                           tooltip="Give this game a name"))
            widths = [m.button_width(c) for c in buttons]
            name_w = max(char * 4, inner - sum(widths) - m.gap * len(buttons))
            if d.renaming == slot:
                name = Control("slot.name", label=d.text, text_field=True,
                               payload=slot,
                               tooltip="Type a name -- Enter keeps it, Esc cancels")
            else:
                text = "%d  %s" % (slot, "Empty" if info is None else info.name)
                name = Control("slot.name", label=text, readout=True, payload=slot)
            self._place([(name, pygame.Rect(x0, y, name_w, m.btn_h))])
            x = x0 + name_w + m.gap
            for control, w in zip(buttons, widths):
                self._place([(control, pygame.Rect(x, y, w, m.btn_h))])
                x += w + m.gap
            self.labels.append((x0 + m.pad, y + m.btn_h + m.gap // 2,
                                state.describe_slot(info), palette.TEXT_DIM))
            y += row_h + m.gap
```

Move `from .state import NAME_MAX` up to the module imports alongside the `DIALOG_*` names (it is shown inline above only to make the dependency obvious).

`on_event`, in the `MOUSEBUTTONDOWN` branch: change

```python
            if self.modal is not None and control is not self._modal_control:
```

to

```python
            if self._modal_is_help and control is not self._modal_control:
```

and after `state.cancel_quit()` (before `if self.over_panel(event.pos):`) insert:

```python
            if state.dialog is not None:
                # Off every button while a dialog is up: an answer of "no" to
                # a waiting Overwrite?, and never a pan of the world behind.
                state.disarm()
                self._press_on_ui = True
                return True
```

In the `MOUSEWHEEL` branch, make it:

```python
        if event.type == pygame.MOUSEWHEEL:
            if state.dialog is not None:
                return True
            if self.over_panel(self.mouse):
                state.scroll_body_list(-event.y)
                return True
```

`perform`: after the `cancel_quit` line, add:

```python
        # Doing anything but the second press is an answer of "no" to a
        # waiting Overwrite? as well.
        if cid != "slot.save":
            state.disarm()

        if cid == "game.saves":
            state.toggle_saves()
            return True
        if cid == "start.new":
            state.start_new()
            return True
        if cid == "start.load":
            state.open_load()
            return True
        if cid == "start.quit":
            state.quit_now()
            return True
        if cid == "dialog.close":
            state.close_dialog()
            return True
        if cid == "slot.load":
            return state.load_slot(payload)
        if cid == "slot.save":
            state.save_slot(payload)
            return True
        if cid == "slot.rename":
            state.begin_rename(payload)
            return True
        if cid == "slot.name":
            return False
```

`draw`: at the top, before the panels loop:

```python
        if state.dialog is not None:
            # The world is dimmed under a dialog, the way it is under the help:
            # one thing to deal with, and nothing behind it is clickable.
            veil = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            veil.fill((0, 0, 0, 128))
            surface.blit(veil, (0, 0))
```

After the panels loop and before the controls loop, draw the labels clipped to the modal:

```python
        if self.labels:
            previous = surface.get_clip()
            clip = self.modal if self.modal is not None else previous
            surface.set_clip(clip.clip(previous) if previous else clip)
            try:
                for x, y, text, colour in self.labels:
                    self.font.draw(surface, x, y, text, colour)
            finally:
                surface.set_clip(previous)
```

And change the help veil condition from `if self.modal is not None:` to `if self._modal_is_help:`.

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_ui.py tests/test_layout.py tests/test_hud.py -q`
Expected: all PASS, including the parity test (`S` now has a button) and the dialog layout test at every window and grid. If `test_the_dialogs_fit_inside_the_window` fails at `config.MIN_WINDOW` grid 3 because a row overflows, reduce `inner_w` by dropping the `+ 3` on `NAME_MAX` — do not add pixel constants.

- [ ] **Step 6: Commit**

```bash
git add src/app/ui.py src/render/hud.py tests/test_ui.py tests/test_layout.py
git commit -m "feat: the Saves button, the start menu and the slot list"
```

---

### Task 6: Input routing, the app boots on the menu, smoke tests

**Files:**
- Modify: `src/app/input.py`
- Modify: `src/app/main.py`
- Test: `tests/test_ui.py` (append key-routing tests), `tests/test_smoke.py`

**Interfaces:**
- Consumes: `AppState.dialog`, the Task 3 actions, `saves.SlotStore`, `saves.default_directory`.
- Produces: `input.dialog_key(event, state, ctx)`; `App.__init__(self, fullscreen=False, saves_dir=None)`; `App.store`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ui.py`:

```python
def key(code, unicode="", mod=0):
    return pygame.event.Event(pygame.KEYDOWN, key=code, unicode=unicode, mod=mod)


def test_the_s_key_opens_and_closes_the_list(saving):
    from src.app.input import handle_event
    st, ui, _ = saving
    handle_event(key(pygame.K_s, "s"), st, None, ui)
    assert st.dialog is not None and st.dialog.kind == "saves"
    handle_event(key(pygame.K_s, "s"), st, None, ui)
    assert st.dialog is None


def test_keys_go_to_the_field_while_renaming(saving):
    from src.app.input import handle_event
    st, ui, _ = saving
    st.open_saves()
    st.save_slot(1)
    st.begin_rename(1)
    handle_event(key(pygame.K_s, "s"), st, None, ui)
    assert st.dialog is not None and st.dialog.renaming == 1
    assert st.dialog.text.endswith("s")
    handle_event(key(pygame.K_BACKSPACE), st, None, ui)
    assert not st.dialog.text.endswith("s")
    handle_event(key(pygame.K_SPACE, " "), st, None, ui)
    assert st.sim.paused, "Space is a character here, not the pause key"
    handle_event(key(pygame.K_ESCAPE), st, None, ui)
    assert st.dialog is not None and st.dialog.renaming is None
    handle_event(key(pygame.K_ESCAPE), st, None, ui)
    assert st.dialog is None


def test_enter_commits_a_name(saving):
    from src.app.input import handle_event
    st, ui, store = saving
    st.open_saves()
    st.save_slot(1)
    st.begin_rename(1)
    for ch in "!?":
        handle_event(key(pygame.K_1, ch), st, None, ui)
    handle_event(key(pygame.K_RETURN), st, None, ui)
    assert st.dialog.renaming is None
    assert store.list()[0].name.endswith("!?")


def test_global_keys_are_swallowed_by_a_dialog(saving):
    from src.app.input import handle_event
    st, ui, _ = saving
    st.open_saves()
    before = st.time_index
    handle_event(key(pygame.K_RIGHTBRACKET), st, None, ui)
    handle_event(key(pygame.K_F5), st, None, ui)
    handle_event(key(pygame.K_t), st, None, ui)
    assert st.time_index == before and st.view.show_trails
    handle_event(key(pygame.K_DELETE), st, None, ui)
    assert st.dialog is not None


def test_escape_disarms_before_it_closes(saving):
    from src.app.input import handle_event
    st, ui, _ = saving
    st.open_saves()
    st.save_slot(1)
    st.save_slot(1)
    assert st.dialog.armed == 1
    handle_event(key(pygame.K_ESCAPE), st, None, ui)
    assert st.dialog is not None and st.dialog.armed is None
    handle_event(key(pygame.K_ESCAPE), st, None, ui)
    assert st.dialog is None


def test_the_start_menu_keys(saving):
    from src.app.input import handle_event
    st, ui, _ = saving
    st.open_start()
    handle_event(key(pygame.K_l, "l"), st, None, ui)
    assert st.dialog.kind == "load"
    handle_event(key(pygame.K_ESCAPE), st, None, ui)
    assert st.dialog.kind == "start"
    handle_event(key(pygame.K_n, "n"), st, None, ui)
    assert st.dialog is None and not st.sim.paused
    st.open_start()
    handle_event(key(pygame.K_q, "q"), st, None, ui)
    assert st.running is False


def test_command_q_still_quits_under_a_dialog(saving):
    from src.app.input import handle_event
    st, ui, _ = saving
    st.open_saves()
    handle_event(key(pygame.K_q, "q", pygame.KMOD_META), st, None, ui)
    assert st.running is False


def test_a_click_on_the_world_under_a_dialog_does_not_reach_the_tool(saving):
    from src.app.input import handle_event
    from src.app.tools import LaunchTool
    st, ui, _ = saving
    st.set_tool(LaunchTool)
    st.open_saves()
    ui.build(st, SIZE)
    n = st.sim.world.n
    for kind in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
        handle_event(pygame.event.Event(kind, pos=(5, 5), button=1), st, None, ui)
    assert st.sim.world.n == n and st.dialog is not None
```

Modify `tests/test_smoke.py`: change the fixture and add tests.

```python
@pytest.fixture
def app(tmp_path):
    from src.app.main import App
    instance = App(fullscreen=False, saves_dir=str(tmp_path / "saves"))
    # The real app boots onto the start menu; most of these tests want a
    # running world, so press New.
    instance.state.start_new()
    yield instance
    pygame.quit()


def test_the_app_boots_onto_the_start_menu(tmp_path):
    from src.app.main import App
    instance = App(fullscreen=False, saves_dir=str(tmp_path / "saves"))
    try:
        assert instance.state.dialog is not None
        assert instance.state.dialog.kind == "start"
        assert instance.state.sim.paused
        for _ in range(5):
            instance.step_once()
        assert instance.state.sim.world.time == 0.0
        assert {c.id for c in instance.ui.controls} == {"start.new", "start.load", "start.quit"}
    finally:
        pygame.quit()


def test_new_from_the_menu_starts_time(app):
    app.state.open_start()
    app.step_once()
    app.ui.perform(next(c for c in app.ui.controls if c.id == "start.new"), app.state)
    for _ in range(5):
        app.step_once()
    assert app.state.sim.world.time > 0.0


def test_save_then_load_through_real_frames(app):
    state = app.state
    for _ in range(10):
        app.step_once()
    state.toggle_saves()
    app.step_once()
    assert state.sim.paused
    app.ui.perform(next(c for c in app.ui.controls
                        if c.id == "slot.save" and c.payload == 1), state)
    app.step_once()
    saved_time = state.sim.world.time
    state.close_dialog()
    for _ in range(10):
        app.step_once()
    assert state.sim.world.time > saved_time
    state.toggle_saves()
    app.step_once()
    app.ui.perform(next(c for c in app.ui.controls
                        if c.id == "slot.load" and c.payload == 1), state)
    assert state.dialog is None
    assert state.sim.world.time == saved_time
    for _ in range(5):
        app.step_once()
    assert state.sim.world.time > saved_time


def test_load_from_the_start_menu(app):
    state = app.state
    state.toggle_saves()
    app.step_once()
    state.save_slot(2)
    state.close_dialog()
    state.open_start()
    app.step_once()
    state.open_load()
    app.step_once()
    ids = {c.id for c in app.ui.controls}
    assert "slot.load" in ids and "slot.save" not in ids
    app.ui.perform(next(c for c in app.ui.controls
                        if c.id == "slot.load" and c.payload == 2), state)
    assert state.dialog is None
    for _ in range(5):
        app.step_once()


def test_frames_run_while_renaming(app):
    state = app.state
    state.toggle_saves()
    state.save_slot(3)
    state.begin_rename(3)
    for _ in range(5):
        app.step_once()
    assert state.dialog.renaming == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ui.py -q -k "key or dialog or start_menu" ; python -m pytest tests/test_smoke.py -q`
Expected: FAIL — `S` does nothing yet (`assert st.dialog is not None`), `TypeError: App.__init__() got an unexpected keyword argument 'saves_dir'`.

- [ ] **Step 3: Route events in `src/app/input.py`**

After the `VIDEORESIZE` block and before `if ui is not None and ui.on_event(...)`:

```python
    # A dialog owns every event.  The interface still gets the mouse, so its
    # buttons work as buttons; the keys go to the dialog and stop there --
    # the throw tool must not fire through the slot list, and a name being
    # typed must not toggle trails with every T.
    if state.dialog is not None:
        if event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN,
                          pygame.MOUSEBUTTONUP, pygame.MOUSEWHEEL):
            if ui is not None:
                ui.on_event(event, state, ctx)
        elif event.type == pygame.KEYDOWN:
            dialog_key(event, state, ctx)
        return
```

Add the global `S` to the key chain, e.g. after the `F5` branch:

```python
    elif key == pygame.K_s:
        state.toggle_saves()
```

Add the function at module bottom:

```python
def dialog_key(event, state, ctx):
    """Keys while a dialog is up.

    Cmd+Q and Cmd+F still mean what they mean everywhere.  A text field
    with focus takes everything else: Enter keeps the name, Esc drops the
    edit (and only the edit -- a second Esc closes the dialog), Backspace
    deletes, and any printable character is typed.  Otherwise Esc disarms
    a waiting Overwrite? before it closes anything, the start menu answers
    to its three letters, and S puts the in-game list away.
    """
    key = event.key
    command = event.mod & (pygame.KMOD_META | pygame.KMOD_CTRL)
    d = state.dialog
    if command and key == pygame.K_q:
        state.running = False
        return
    if command and key == pygame.K_f:
        if ctx is not None:
            ctx.toggle_fullscreen()
            state.on_resize(ctx)
        return
    if d.renaming is not None:
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            state.rename_commit()
        elif key == pygame.K_ESCAPE:
            state.rename_cancel()
        elif key == pygame.K_BACKSPACE:
            state.rename_backspace()
        elif not command and getattr(event, "unicode", ""):
            state.rename_type(event.unicode)
        return
    if key == pygame.K_ESCAPE:
        if d.armed is not None:
            state.disarm()
        else:
            state.close_dialog()
    elif d.kind == "start":
        if key == pygame.K_n:
            state.start_new()
        elif key == pygame.K_l:
            state.open_load()
        elif key == pygame.K_q:
            state.quit_now()
    elif d.kind == "saves" and key == pygame.K_s:
        state.toggle_saves()
```

Use the `DIALOG_START` / `DIALOG_SAVES` constants from `.state` instead of the string literals (import them at the top: `from .state import DIALOG_SAVES, DIALOG_START`).

- [ ] **Step 4: Boot onto the menu in `src/app/main.py`**

Imports: add `from .saves import SlotStore, default_directory`.

`App.__init__(self, fullscreen=False, saves_dir=None)`:

```python
        self.store = SlotStore(default_directory() if saves_dir is None else saves_dir)
        self.state = AppState(Simulation(), self.camera, ViewSettings(), store=self.store)
        self.state.set_tool(SelectTool)
        # Opens onto New / Load / Quit, with the solar system frozen behind it.
        self.state.open_start()
```

`step_once`: wrap the explain panel, the tool overlay and the HUD so a dialog is the only chrome:

```python
        if state.dialog is None:
            report = state.explain.report if state.explain.enabled else None
            if report is not None:
                m = self.hud.metrics(surface.get_width(), surface.get_height())
                state.explain_rect = self.explain_panel.draw(
                    surface, report, m,
                    widgets.overlay_x(m, state.show_body_list), state.show_maths)
            else:
                state.explain_rect = None
            state.tool.draw_overlay(surface, state, self.font)
            self.hud.draw(surface, state)
```

(`ui.build` / `ui.draw` stay as they are; they build and draw the dialog.)

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_ui.py tests/test_smoke.py tests/test_state.py -q`
Expected: all PASS.

- [ ] **Step 6: Run the whole suite**

Run: `python selftest.py` (Bash timeout 400000 ms)
Expected: everything passes. `tests/test_layering.py` in particular: `src.save` imports no pygame.

- [ ] **Step 7: Run the real app once**

Run: `timeout 5 ./.venv/bin/python -m src.app.main || true` (or the system python if pygame is installed there). It should open on the start menu and exit on the timeout without a traceback.

- [ ] **Step 8: Commit**

```bash
git add src/app/input.py src/app/main.py tests/test_ui.py tests/test_smoke.py
git commit -m "feat: boot onto New / Load / Quit; a dialog owns every event"
```

---

### Task 7: Document the rules in `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Architecture table**

In the `src/*.py` row, after `explain.py (...)`, add: `, and `save.py` (a world and a session as plain data, and back)`. In the `src/app/` row add: `, `saves.py` (the three slot files)`.

- [ ] **Step 2: Invariants**

Append these bullets to the Invariants list:

```markdown
- **A dialog owns every event** (`src/app/input.py`). While `state.dialog` is set,
  `ui.build` lays out only the dialog, mouse events go to `ui.on_event` and keys to
  `input.dialog_key`, and nothing reaches a tool or a global key -- the throw tool
  would otherwise fire through the slot list, and a name being typed would toggle
  trails on every `T`. Cmd+Q and Cmd+F are the two exceptions.
- **`save.from_data` validates before it mutates.** Every record is parsed and
  checked first and the world is cleared only once all of them are good, so a
  refused file leaves the running world exactly as it was. Kind is loaded as
  saved, never re-derived from mass (a hole and a craft are one-way); temperature
  is loaded as saved and **not** settled (it is state, and a planet he heated on
  purpose must still be hot). `sim.refresh()` follows, as after any membership
  change. `World.rekey` is how the loader restores ids and the id counter.
- **The camera is saved as `au_across`, never `log_zoom`.** `log_zoom` is pixels
  per AU, so the same save would show a different stretch of sky in a different
  window.
- **A save file is written whole or not at all** (`app/saves.py`): `slotN.json.tmp`
  then `os.replace`. A file that cannot be read lists as unreadable with Load and
  Rename disabled; Save stays enabled because overwriting is the way out.
```

- [ ] **Step 3: Commands**

Add to the Commands block: `# saves live in saves/slot{1,2,3}.json beside run.command (gitignored)`.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: the save-slot rules"
```

---

## Self-review

**Spec coverage.** Start menu (Task 5 `_build_start`, Task 6 boot + keys); New = solar system (`start_new` leaves the booted world); Load-only list with Back (`DIALOG_LOAD`, `close_dialog` → `open_start`); Quit immediate (`quit_now`); in-game `S` button beside Quit (Task 5); pause while open and restore (`open_saves` / `close_dialog`); Save one press empty / two press full, keeps name (`save_slot`); Load restores saved pause (`load_slot`); Rename with field, 24 printable ASCII, Enter/Esc/Backspace, empty keeps old (`rename_*`, `dialog_key`); focused field swallows `S` (`dialog_key` checks `renaming` first); Close/Esc (`dialog.close`, `dialog_key`); unreadable slot shows Can't read, Load/Rename disabled, Save enabled (`SlotStore.list`, `_build_slots`); what is saved (`to_data`); `au_across` (`_session` / `_apply_session`); header with summary (`to_data`); not saved list — nothing in `Session` covers window/grid/effects/report; loading details (`from_data`, `load_slot`: refresh, no settle, explain.clear, effects.clear, offset, quit_armed, tool reset); files, atomic write, gitignore (Task 2); error handling: unknown version/kind/material/too many/bad ids (Task 1), unwritable directory (Save disabled only when `store is None` — see gap below), dangling selection/follow (`from_data`); tests as listed in the spec across Tasks 1–6; layering (Task 1 step 5).

**Gap found and fixed here:** the spec says an uncreatable saves directory disables Save "with the tooltip saying why". Task 3's `save_slot` returns False on `OSError`, and Task 5 disables Save only when `store is None`. To meet the spec, `_build_slots` computes `can_save = state.store is not None and state.store.writable()` and uses the tooltip `"No saves folder to write to"` in both cases — apply that in Task 5 Step 4 (the `can_save` line) and the existing test `test_without_a_store_save_is_disabled_and_says_why` still passes. `writable()` is one `os.makedirs` + `os.access` per build frame while the dialog is open, which is fine.

**Placeholders.** None: every step carries its code.

**Type consistency.** `Session` field names match between `save.py`, `_session`, `_apply_session` and the layout fixture's positional construction (order: scenario, time_index, paused, show_trails, real_size, show_labels, explain_enabled, show_body_list, selected_id, centre, follow_id, au_across). `SlotInfo(slot, name, saved_at, summary, readable)` is constructed identically in `saves.py`, `state.py` and the tests. `save_slot` returns True only on a written file; `perform` for `slot.save` returns True regardless (an arming is an action). Control ids are identical across `ui.py`, `test_ui.py`, `test_smoke.py`, `test_layout.py`.
