# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python selftest.py                      # whole suite (1363 tests, ~3:40) -- give Bash timeout: 300000
python -m pytest tests/test_orbits.py    # one file
python -m pytest tests/test_sim.py::test_paused_simulation_does_not_move   # one test
python -m src.app.main                  # run the app
python -c "from src.app.main import probe; probe()"   # report what SDL actually gives us
./run.command                           # double-clickable launcher; creates .venv on first run
# saves live in saves/slot{1,2,3}.json beside run.command (gitignored)
```

The full suite outruns the Bash tool's 120 s default; `.claude/settings.json` raises the
default to 300 s, and any subagent brief that ends in "run the suite" should still say so.
Use single test files for the inner loop (`tests/test_maneuver.py` runs in under a second).

The venv at `.venv/` is created by `run.command`; use `./.venv/bin/python` if the system
Python lacks pygame/numpy. The dependency is **pygame-ce** (`import pygame` is unchanged);
it cannot share a venv with the original `pygame`, so `run.command` uninstalls that first
whenever it finds it.

Tests that touch pygame set `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")` at module
top, before importing pygame. Any new render/app test must do the same or it will try to
open a window.

**Reviewers verify by mutation**: break the guarded line and confirm a test fails. Tests
written into a plan before the code exists have repeatedly passed against broken code.

## Architecture

Three layers, enforced by `tests/test_layering.py`:

| | |
|---|---|
| `src/*.py` | physics. Pure numpy. **Must never import pygame** (a subprocess test asserts this). `collisions.py` and `destruction.py` (tidal/thermal destruction) emit the plain-data `ImpactEvent`; `thermal.py` (temperatures), `maneuver.py` (flight plans, Hohmann, gravity assist), `explain.py` (what an edit did, as plain data), `save.py` (world <-> plain data), `measure.py` (ruler and Kepler orbit timer). |
| `src/render/` | camera, sprites, HUD, font, starfield, widgets, `shade.py` (sphere/star shading), `effects.py` (`ImpactEvent` -> particles). Everything that touches pixels. |
| `src/app/` | display context, frame loop, input routing, the interface, one class per tool, `saves.py` (the three slot files). |

`App.step_once()` (`src/app/main.py`) is one frame: events (each through `ctx.map_event`)
→ reconcile window size → rebuild the pixel-sized stack if the buffer changed shape →
`ui.update` → refresh `world.contact_radius` → `sim.advance` → `sim.update_trails` →
`state.sample_timer` → `state.explain.settle` → camera follow → drain `sim.impacts` into
`state.effects` → draw world → effects → explain panel → tool overlay → HUD → `ui.build`
→ `ui.draw` → `ctx.present`. It is factored out of `run()` so `tests/test_smoke.py` can
drive the real application headless. The interface is built **at the end of the frame**,
from the state about to be shown, so the next frame's clicks hit what is on screen.

### Invariants

Load-bearing; violating one produces bugs that only show under merges, edits or fast time
scales. Each has a long form with its reasoning and measurements in
**`docs/ARCHITECTURE.md` — read that entry before changing code it names.**

Physics and state
- **Units are AU, solar masses, days.** `src/units.py` is the only place `G` or a conversion factor is defined.
- **Never hold an array index across a frame.** `World.remove` swap-removes; store `body.id`, resolve with `World.index_of(id)` (`None` when gone).
- **Every mutation of mass, radius or membership is followed by `Simulation.refresh()`**, or cached accelerations go stale.
- **Mass and density are set; radius is derived** (`src/edits.py`). Kind follows mass across `units.STAR_MASS_THRESHOLD`.
- **`World.contact_radius` is not `World.radius`.** Only `find_collisions`' did-they-touch test uses `contact_radius` (refreshed each frame from what is drawn); every outcome uses true `radius`. Any radius mutation resets `contact_radius` to match.
- **A black hole's radius is `units.schwarzschild_radius`.** Collapse is one-way (`edits._refresh_kind` returns early for a hole); `units.COLLAPSE_MASS_THRESHOLD` (150 M☉) tops the mass scrub.
- **A black hole is drawn, picked and contact-refreshed through `render/scale.visual_radii`**; physics never reads it.
- **A black hole always takes the `i` slot in `find_collisions`** and always accretes: no graze, fragments or ejecta.
- **Collision/disruption thresholds use `units.disruption_threshold`**, never binding energy alone (material-strength branch dominates below ~100 m).
- **A maneuver node fires per substep, never per frame** (`sim.py`). `Simulation._next_burn_time` caches the soonest `due_time`; `refresh()` recomputes it, so every plan edit calls through there.
- **A craft is a one-way kind** and offers **no Edit rows** (`edits.edit_fields` returns `()`); fuel and flight plan live in the Fly panel.
- **A craft launches heliocentric, never into a parking orbit** (`tools.FlyTool._launch`); a 90-minute orbit would drive the global block level to its ceiling.
- **A Hohmann transfer is planned from semi-major axes** (`tools._mean_radius`), then refined by `maneuver.refine_window`. It reaches the target's orbit, not a rendezvous.
- `World.MAX_BODIES` is 200 and must not be raised (O(n²) force kernel budget).
- **Angles are radians** everywhere except the source tables in `src/scenarios.py`.

Temperature
- **Temperature is state** (`World.temperature`). Every path that creates a body (`World.add`, every collision outcome, every tidal fragment) must set one.
- **`thermal.pinned_temperature` decides who is free.** Stars and holes are pinned and not editable; `edits.set_temperature` refuses them; `_refresh_kind` and `collapse` write the pinned value at the kind change.
- **`thermal.advance` runs once a frame, never per substep** (O(n²)).
- **A locked field is refused in `edits.py`, not the panel.** Each setter checks only its **own** field in `Body.locked_fields`. Locks don't stop `collapse`, merges, ablation or pinned writes. `thermal.advance` skips a `"temp"`-locked body, checked after the pinned branch. Locks aren't inherited; Restart (`F5`) drops them all.

Interface and input
- **Every action has a button, and every button carries its hotkey** (`tests/test_ui.py` checks against `hud.KEY_HELP`). **A hotkey does not inherit `enabled=False`**, so a key path must repeat the control's precondition itself.
- **Events go interface → tool → global keys** (`src/app/input.py`).
- **A dialog owns every event** while `state.dialog` is set; only Cmd+Q and Cmd+F escape.
- **Pixel sizes scale by `internal_height / scale.REF_HEIGHT` (360); interface sizes come from `widgets.Metrics`.** Never hard-code a pixel count.
- **Nothing caches a size across frames.** Pixel-sized objects are rebuilt by `App._build_render_stack` when the buffer changes.
- **Mouse coordinates are internal pixels everywhere** (`ctx.map_event`, `ctx.mouse_pos()`).
- **`widgets.overlay_x` is where an overlay starts** (shared by the Fly readout and the explain panel).
- **The craft's render floor is `scale.craft_floor_px`**, shared by drawing and picking, and never reaches `contact_radius`. Real-size mode drops it.

Explain layer
- **Compare orbital *elements*, never positions** (`src/explain.py`). Temp rows: the edited body reads instantaneous `world.temperature`; other bodies read `thermal.equilibrium_temperature`; stars read `thermal.pinned_temperature`.
- **Every write path calls `ExplainRecorder.note` before the edit and `.record` immediately after, in the same call stack.** `settle` only diffs; it never snapshots, because a frame's `sim.advance` would already have relaxed the temperature.
- **One report per burst**: the baseline is taken on the first edit, and the report follows `SETTLE_MS` of wall-clock quiet.

Saves and measurement
- **`save.from_data` validates before it mutates.** Kind and temperature load as saved (never re-derived or settled); `sim.refresh()` follows; `World.rekey` restores ids.
- **The camera is saved as `au_across`, never `log_zoom`.**
- **A save file is written whole or not at all** (`.tmp` then `os.replace`). An unreadable slot disables Load/Rename; Save stays enabled.
- **The orbit timer measures angle swept, once a frame, and restarts when a frame outruns a quarter turn at periapsis**: limit `MAX_SAMPLE_FRACTION * period * (1-e)**1.5 / sqrt(1+e)`. The lap end is interpolated; Kepler reads the lap's frozen mean `a`. The timer lives on `AppState`; Restart/Load drop it; it is not saved.

### Design decisions

Full text in `docs/ARCHITECTURE.md`; read it before changing any of these.

- Physics: Velocity Verlet with dt = `DT_BASE / 2**k` (symplectic; `k` moves one notch per frame); a wall-clock physics budget (`DEFAULT_BUDGET_S`) that degrades to slow motion; swept collision detection with `event_id` immunity for fragments; three collision outcomes (`classify()`); destruction without contact (tides from attractors, heat from stars; one violation per call); tidal fragments get `no_collide_until` immunity; algebraic orbit previews (`orbits.py`); trails sampled by angle swept.
- Thermal: every body heats every other (one flux sum, no greenhouse term, so Earth reads −19 °C); a one-metre skin relaxed analytically (**tests: at 3000 K tau is 0.008 days, so a one-day step lands exactly on equilibrium**); impacts put 30% of Q_R into the remnant.
- Rendering: per-pixel shaded spheres (`shade.py`, light tilted 40° toward the viewer); a 25% size ladder capped at `SHADE_MAX_PX`; `Body.look` is appearance, dropped by `set_material`; hot bodies are coloured from the Planck locus, quantised to 100 K; the hole and craft are supersampled, and a hole is never drawn with `BLEND_ADD`; rings are two surfaces; drawn radius is `max(zoom*r, floor(r))`; the accretion disk's inner edge is exaggerated; `Body.spin`, `Body.disk` and colours live off the arrays or on `Material`; impact effects run on wall-clock time and never read the physics velocity; the sky is pre-rendered; text is Atkinson Hyperlegible.
- Display: `s` is native or 1 (`G` toggles); a `pygame.Window(allow_high_dpi=True)`, never `display.set_mode`; **never call `pygame.Window.from_display_module()`** (it segfaults; `test_layering.py` greps for it); `config.MIN_INTERNAL` is the smallest window, checked by `tests/test_layout.py`.
- Interface: a `Control` is a description, not a widget; icons are drawn from lines, not glyphs; steppers repeat while held (`ui.REPEAT_DELAY_MS`).

### Adding a tool

Subclass `Tool` in `src/app/tools.py`, register it in `TOOLS` (keyed by pygame key
constant; `TOOL_BY_NAME` derives from it), add its row to `ui.TOOL_BUTTONS` (the hotbar
reads that), and add it to the tool lists in `tests/test_layout.py` and the parity test in
`tests/test_ui.py`.

- `on_event` returns True to consume an event; also `draw_overlay`, `on_exit`, `name`, `hint`.
- `controls(state, metrics)` returns rows of `Control`; `perform(control, state)` acts on
  them. A tool never draws its own buttons or hit-tests the mouse; `ui.py` does both.
- Mixins: **`DragPan`** (drag pans, click selects: call `_pan_event`, override
  `_on_click`) and **`TargetLock`** (`X`-then-click target: `target_id`, `picking_target`,
  `_target_index`, `_target_label`, `_pick_target`). Call each mixin's `__init__`
  explicitly from the tool's own.

### Adding a control

Add a `Control` in the relevant `_build_*` method of `src/app/ui.py` and a branch in
`UiLayer.perform`. Give it a `hotkey` only if the key exists, and add the key to
`hud.KEY_HELP`. `enabled=False` disables it everywhere, including `perform`.

A tool panel row is not free: `tests/test_layout.py` checks the Edit panel stops above
the transport bar at `config.min_internal(3)`, which leaves room for six rows and no more
(which is why "Unlock all" shares the Collapse row).

## Docs

`docs/ARCHITECTURE.md` holds the long form of every invariant and design decision above.
`docs/superpowers/specs/` holds the design (Addendum A covers the clickable interface), and
`docs/superpowers/plans/` the implementation plans with their global constraints. Module
docstrings carry the reasoning for most decisions and are worth reading before changing
the code under them.
