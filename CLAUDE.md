# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python selftest.py                      # whole suite (553 tests, ~50s)
python -m pytest tests/test_orbits.py    # one file
python -m pytest tests/test_sim.py::test_paused_simulation_does_not_move   # one test
python -m src.app.main                  # run the app
python -c "from src.app.main import probe; probe()"   # report what SDL actually gives us
./run.command                           # double-clickable launcher; creates .venv on first run
```

The venv at `.venv/` is created by `run.command`; use `./.venv/bin/python` if the system
Python lacks pygame/numpy.

Tests that touch pygame set `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")` at module
top, before importing pygame. Any new render/app test must do the same or it will try to
open a window.

## Architecture

Three layers, enforced by `tests/test_layering.py`:

| | |
|---|---|
| `src/*.py` | physics. Pure numpy. **Must never import pygame** — a subprocess test asserts this. Includes `collisions.py` (detection and outcome resolution) and `destruction.py` (tidal and thermal destruction near a star), both emitting the plain-data `ImpactEvent`. |
| `src/render/` | camera, sprites, HUD, pixel font, starfield, button widgets, `effects.py` (turns an `ImpactEvent` into particles). Everything that touches pixels. |
| `src/app/` | display context, frame loop, input routing, the interface, one class per tool. |

`App.step_once()` (`src/app/main.py`) is one frame: events (each one put through
`ctx.map_event` first) → reconcile the window size → rebuild the pixel-sized stack if the
buffer changed shape → `ui.update` (held steppers) → refresh `world.contact_radius` from
the camera → `sim.advance` → `sim.update_trails` → camera follow → drain `sim.impacts`
into `state.effects` → draw world → `effects.update` / `effects.draw` → draw the tool
overlay → draw HUD → `ui.build` → `ui.draw` → `ctx.present`. It is factored out of `run()`
so `tests/test_smoke.py` can drive the real application headless.

The interface is built **at the end of the frame**, from the state about to be shown, so
the next frame's clicks hit exactly what is on screen.

### Invariants

These are load-bearing; violating one produces bugs that only show up under merges, edits
or fast time scales.

- **Simulation units are AU, solar masses, days.** `src/units.py` is the single source of
  truth for `G` and every conversion factor. No other module may define one.
- **Never hold an array index across a frame.** `World.remove` is a swap-remove, so an
  index can silently start referring to a different body. Selection, camera follow, and
  the trajectory predictor store `body.id` and resolve through `World.index_of(id)`,
  which returns `None` when the body is gone.
- **Every mutation of mass, radius or membership must be followed by
  `Simulation.refresh()`** (which calls `integrator.refresh(world)`) so the cached
  accelerations are not stale.
- **Mass and density are set; radius is always derived** (`src/edits.py`). There is no
  code path allowed to produce an inconsistent mass/radius/density triple. Kind follows
  mass: crossing `units.STAR_MASS_THRESHOLD` ignites a body, dropping below extinguishes it.
- **`World.contact_radius` is not `World.radius`.** Every collision *outcome* the physics
  computes (energy, impact parameter, disruption threshold, fragment count, ...) uses the
  true `radius`. Only the yes/no question "did these two touch" in `find_collisions` uses
  `contact_radius`, which the app layer refreshes once a frame from what the camera actually
  draws (`render/scale.contact_radii_au`, `World.clamp_contact_radii`) — otherwise a shot
  aimed at what is on screen misses whatever the drawn-radius floor inflated. A mutation of
  `radius` (merge, graze, accretion, ablation) must reset `contact_radius` to match, so it is
  never left below true size in the substeps before the next frame's refresh.
- **Collision and disruption thresholds are `units.disruption_threshold`, never gravitational
  binding energy alone.** Binding energy alone puts the shatter point at roughly escape
  velocity, which is far too easy to reach; the real threshold has a material-strength branch
  (scaled by `Material.toughness`) that dominates for anything smaller than ~100 m.
- **Every action has a button, and every button carries its hotkey.** A control never
  reimplements a key: it names an action that is the same `AppState` or tool method the
  key already called. `tests/test_ui.py` reads the shortcut list out of `hud.KEY_HELP` and
  fails if any documented key has no button.
- **Events go interface → tool → global keys** (`src/app/input.py`). The throw tool reads
  any press as an aim, so tool-first ordering would fire a comet through the Delete button.
- `World.MAX_BODIES` is 200 and must not be raised; the O(n²) force kernel is budgeted for it.
- Angles are radians everywhere except the source data tables in `src/scenarios.py`.
- Pixel-size constants scale by `internal_height / scale.REF_HEIGHT` (360); interface sizes
  come from `widgets.Metrics`, which derives every one of them from the font. Neither may
  hard-code a pixel count, because the `G` grid switch changes both.
- **Nothing may cache a size across frames.** The window is resizable, so the buffer can
  change shape at any frame boundary. `ui.build` already runs from the current size every
  frame; anything else sized in pixels (font, camera, starfield, sprite cache) is rebuilt
  by `App._build_render_stack`, which `step_once` calls whenever the buffer changed.
- **Mouse coordinates are internal pixels everywhere.** `ctx.map_event` translates every
  event on the way in and `ctx.mouse_pos()` is how a wheel event finds the pointer, so no
  reader of `event.pos` — interface, tools, effects — knows the grid exists.

### Design decisions worth knowing before changing them

- **Velocity Verlet, not RK4** (`integrator.py`). Symplectic: bounded oscillatory energy
  error rather than the secular inward spiral RK4 gives over thousands of orbits. dt is
  restricted to `DT_BASE / 2**k` — a continuously varying dt destroys symplecticity — and
  `k` moves at most one notch per frame so it cannot chatter.
- **The physics wall-clock budget is not optional** (`sim.py`, `DEFAULT_BUDGET_S`). At ten
  years per second the integrator wants thousands of substeps per frame; without the cap
  the window stops responding. It degrades to slow motion and the HUD reports `last_ratio`.
- **Collision detection is swept, not point-in-time** (`collisions.py`). A fast projectile
  is outside the target both before and after a step. Fragments from one shattering (or a
  graze, or cratering ejecta) share an `event_id` and ignore each other until
  `no_collide_until`, otherwise they re-merge on the next substep.
- **A collision is one of three outcomes, not a merge/shatter binary**: `classify()` picks
  accretion, a hit-and-run graze, or disruption from the specific impact energy against
  `disruption_threshold` and the impact angle against the target's own critical impact
  parameter — a graze needs both a shallow angle and a projectile heavy enough to survive
  being deflected (`GRAZE_MIN_MASS_RATIO`). Fragment speeds and the largest-remnant mass
  (Leinhardt & Stewart) are bounded by the energy the impact actually delivered above the
  threshold, not handed out at a flat multiple of escape velocity.
- **Destruction does not require contact** (`destruction.py`). A body can be torn apart by a
  star's tide (`tidal_stress_ratio`, gravity-only tidal stress is a pure density ratio,
  independent of the body's own size — the classic Roche result) or boiled away by its heat
  (`equilibrium_temperature` against `Material.vapour_temperature_k`), run once a substep
  over stars only. Like `CollisionResolver.resolve()`, only the single worst violation is
  handled per call, since removing a body shifts every index after it.
- **Impact effects advance on wall-clock time, never simulated time** (`render/effects.py`).
  A physics quantity (`ImpactEvent.vel`, AU/day) must never be used as a particle's
  real-second world velocity — that was a real bug: a flash inheriting it directly shot off
  screen within one frame. Every particle's velocity is a purpose-built AU-per-real-second
  visual speed, with no contribution from the physics velocity at all, so an impact looks
  the same at any time-scale.
- **Drawn radius is a floor, not a multiplier** (`render/scale.py`):
  `max(zoom * r_true, floor(r_true))`, floor ∝ `r_true**0.35` and clamped to `FLOOR_MAX`.
  Size ordering survives while nothing inflates far enough to swallow the orbits around
  it. Zooming in is continuous — the true size eventually overtakes the floor, no mode
  switch. Real-size mode drops the floor to one pixel.
- **Orbit previews are algebraic, not integrated** (`orbits.py`). Exact two-body elements
  from state vectors, which is what makes the aiming ellipse free to redraw every frame;
  a test checks it against 4000 integration steps to 0.02 AU.
- **Trails sample by angle swept about the dominant attractor**, not by time or distance,
  so Mercury's and Neptune's are equally smooth (`body.py`).
- **`DEFAULT_S = 1`** (`app/config.py`). SDL reports logical points, never device pixels,
  so one internal pixel per point renders as an exact 2×2 block on Retina. The display is
  opened twice — a fullscreen window is shorter than the desktop by the menu-bar strip,
  and sizing the buffer to the desktop leaves SDL scaling by a non-integer factor that
  smears the pixel grid. See that module's docstring for the measured numbers.
- **The window is `RESIZABLE`, and the upscale is ours, not SDL's** (`app/config.py`).
  `pygame.SCALED` pins the window's *minimum* size to the buffer, so a desktop-sized
  buffer made the window un-shrinkable — and a second `set_mode` with `SCALED` raises
  "failed to create renderer" on the cocoa driver, which `_open` caught and turned into
  "keep the display we had", so the fullscreen toggle and the grid switch silently did
  nothing. A plain `RESIZABLE` window has neither problem. `DisplayContext.present` does
  the coarse-grid upscale itself, by exactly `s`, and `map_event` / `to_internal` do the
  mouse translation `SCALED` used to do for us.
- **Nothing may touch `pygame._sdl2.video.Window`.** It is the only way to move, minimise
  or set a minimum size on a window in pygame 2.6, and the object
  `Window.from_display_module()` returns destroys the real SDL window when it is
  garbage-collected — the program then segfaults some frames later, inside
  `pygame.event.get`, a long way from the call that did it.
- **The interface has a smallest buffer it fits in, and a coarser grid is a divisor**
  (`config.MIN_INTERNAL`, `min_internal`, `grid_fits`). So which pixel grids are on offer
  is a property of the *window*: `ctx.usable_grids` drops the coarse ones in a narrow
  window, `cycle_grid` steps through only those, and shrinking the window steps `s` back
  down rather than laying controls out off the edge. `tests/test_layout.py` builds every
  panel of every tool at the floor and fails if anything lands outside.
- **Body colours live on `Material`, not in `render/palette.py`**, because they are a
  property of the substance rather than of the theme.
- **A control is a description, not a widget object** (`render/widgets.py`). One `Control`
  record drives its drawing, its hit-testing and its tooltip, so a button cannot be drawn
  in one place and clickable in another.
- **Icons are drawn from lines, not typed as glyphs.** The font is whatever SDL supplies
  and may not have a pause bar or a chevron; a missing glyph would render as a box.
- **Steppers act on press and repeat while held** (`ui.REPEAT_DELAY_MS`). Mass scrubs by
  ratio at 1.06 a notch, so igniting Jupiter is ~80 notches — one per click is unusable.

### Adding a tool

Subclass `Tool` in `src/app/tools.py` and register it in `TOOLS`, keyed by pygame key
constant; `TOOL_BY_NAME` is derived from that dict, so the buttons follow automatically.

- `on_event` returns True to consume an event; `draw_overlay`, `on_exit`, `name`, `hint`.
- `controls(state, metrics)` returns rows (a list of lists of `Control`) for the panel
  under the tool palette, and `perform(control, state)` acts on them. A tool never draws
  its own buttons or hit-tests the mouse — `ui.py` does both, which is what stops a tool's
  drag handling from fighting its own buttons over the same press.

### Adding a control

Add a `Control` in the relevant `_build_*` method of `src/app/ui.py` and a branch in
`UiLayer.perform`. Give it a `hotkey` only if the key actually exists, and add the key to
`hud.KEY_HELP` — the parity test reads that table. `enabled=False` disables it everywhere,
including `perform`, which is the door the held-repeat and the tests come through.

## Docs

`docs/superpowers/specs/` holds the design (Addendum A covers the clickable interface),
`docs/superpowers/plans/` the implementation plan with its global constraints (Phase 2 is
the interface). Module docstrings carry the reasoning for most decisions and are worth
reading before changing the code under them.
