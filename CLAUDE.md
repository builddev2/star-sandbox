# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python selftest.py                      # whole suite (1253 tests, ~3:40)
python -m pytest tests/test_orbits.py    # one file
python -m pytest tests/test_sim.py::test_paused_simulation_does_not_move   # one test
python -m src.app.main                  # run the app
python -c "from src.app.main import probe; probe()"   # report what SDL actually gives us
./run.command                           # double-clickable launcher; creates .venv on first run
# saves live in saves/slot{1,2,3}.json beside run.command (gitignored)
```

The venv at `.venv/` is created by `run.command`; use `./.venv/bin/python` if the system
Python lacks pygame/numpy. The dependency is **pygame-ce** (`import pygame` is unchanged);
it cannot share a venv with the original `pygame`, so `run.command` uninstalls that first
whenever it finds it.

Tests that touch pygame set `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")` at module
top, before importing pygame. Any new render/app test must do the same or it will try to
open a window.

## Architecture

Three layers, enforced by `tests/test_layering.py`:

| | |
|---|---|
| `src/*.py` | physics. Pure numpy. **Must never import pygame** — a subprocess test asserts this. Includes `collisions.py` (detection and outcome resolution) and `destruction.py` (tidal and thermal destruction near a star or a black hole), both emitting the plain-data `ImpactEvent`, `thermal.py` (every body's temperature),
`maneuver.py` (a craft's flight plan: nodes, burns, closest approach, the Hohmann
transfer and its launch window, and the gravity assist), and `explain.py` (what an edit
did and why, as plain data), and `save.py` (a world and a session as plain data, and
back). |
| `src/render/` | camera, sprites, HUD, pixel font, starfield, button widgets, `effects.py` (turns an `ImpactEvent` into particles). Everything that touches pixels. |
| `src/app/` | display context, frame loop, input routing, the interface, one class per tool, `saves.py` (the three slot files). |

`App.step_once()` (`src/app/main.py`) is one frame: events (each one put through
`ctx.map_event` first) → reconcile the window size → rebuild the pixel-sized stack if the
buffer changed shape → `ui.update` (held steppers) → refresh `world.contact_radius` from
the camera → `sim.advance` → `sim.update_trails` → `state.explain.settle` → camera follow
→ drain `sim.impacts` into `state.effects` → draw world → `effects.update` / `effects.draw`
→ draw the explain panel → draw the tool overlay → draw HUD → `ui.build` → `ui.draw` →
`ctx.present`. It is factored out of `run()` so `tests/test_smoke.py` can drive the real
application headless.

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
- **Collision and disruption thresholds are `units.disruption_threshold`, never gravitational
  binding energy alone.** Binding energy alone puts the shatter point at roughly escape
  velocity, which is far too easy to reach; the real threshold has a material-strength branch
  (scaled by `Material.toughness`) that dominates for anything smaller than ~100 m.
- **Every action has a button, and every button carries its hotkey.** A control never
  reimplements a key: it names an action that is the same `AppState` or tool method the
  key already called. `tests/test_ui.py` reads the shortcut list out of `hud.KEY_HELP` and
  fails if any documented key has no button. **A hotkey does not inherit a control's
  `enabled` gate.** Keyboard events bypass `UiLayer.perform`'s `clickable` check that
  enforces `enabled=False`, so a key path must duplicate any precondition the control
  carries (e.g., checking `body.kind == KIND_STAR` inside `EditTool.on_event` for Collapse
  `C`) — otherwise a greyed-out button's hotkey still works.
- **Events go interface → tool → global keys** (`src/app/input.py`). The throw tool reads
  any press as an aim, so tool-first ordering would fire a comet through the Delete button.
- **A maneuver node fires per substep, never per frame** (`sim.py`). At one year a second
  a frame is six days of simulated time and at ten years a second it is sixty, so a node
  checked at frame boundaries fires up to sixty days late -- millions of kilometres of
  miss at transfer speeds. `Simulation._next_burn_time` caches the soonest `due_time` in
  the world so the substep loop is one float comparison, and only scans bodies when one is
  actually due; a per-substep Python loop over 200 bodies would eat the wall-clock budget
  many times over. **`refresh()` recomputes it**, which is why every path that edits a
  plan calls through there.
- **A craft is a one-way kind, like a black hole** (`edits._refresh_kind`). No amount of
  mass scrubbing ignites a spacecraft. It also offers **no Edit rows at all**
  (`edits.edit_fields` returns `()`): `SPACECRAFT` is 2.7 g/cm3 but a 37 m hull at any
  real craft mass derives 0.003..0.2, so the density readout would be nonsense, `set_mass`
  would hold that derived density and silently inflate the hull, and `set_density` would
  relabel the craft "Rock". A craft's editable properties are fuel and flight plan, and
  both live in the Fly panel.
- **A craft is launched heliocentric, never into a parking orbit** (`tools.FlyTool._launch`).
  `Integrator.dynamical_time` takes the tightest pair in the *entire* system: a craft 300 km
  above Earth has a ninety-minute period and would drive the block level to its ceiling and
  step the whole solar system in seconds. The six existing moons already cost a factor of 64.
- **The craft's render floor is `scale.craft_floor_px`, shared by drawing and picking, and
  it never reaches `contact_radius`.** Making the picture bigger must not make the craft
  easier to *hit*; making it easier to *click* is the whole point, and only the draw side
  used to scale with the window. Real-size mode drops it, or the shuttle would be the one
  object that refuses to collapse to a dot.
- **A Hohmann transfer is planned from semi-major axes, not present distances**
  (`tools._mean_radius`). Mars swings 1.38..1.67 AU; aiming at where it stands today sends
  the craft to a radius it has already left -- measured, a 1.9 AU miss. The window is then
  refined against exact propagation (`maneuver.refine_window`), because a true anomaly does
  not advance at its mean motion. What remains is n-body drift over the transfer, which no
  closed form fixes: the craft reaches the target's *orbit* with textbook delta-v, not a
  rendezvous.
- `World.MAX_BODIES` is 200 and must not be raised; the O(n²) force kernel is budgeted for it.
- **Temperature is state, not a derived quantity.** It lives in `World.temperature`
  because it changes every step and must survive being edited; `Body.spin` and
  `Body.disk` are on the Body because they are static appearance. Every path that
  creates a body — `World.add`, every collision outcome, every tidal fragment —
  must give it one, or it starts at absolute zero, which is colder than the
  universe allows and produces a warm-up nobody asked for.
- **`thermal.pinned_temperature` decides who is free.** A star's temperature is
  `units.star_temperature(mass)` and a black hole's is its Hawking temperature;
  neither relaxes, neither is editable, and `edits.set_temperature` refuses both
  even though the panel already omits the row — the panel is not the only caller.
  `edits._refresh_kind` and `edits.collapse` write the pinned value the moment a
  body changes kind, so the array is never a frame behind what the body is.
- **A locked field is refused in `edits.py`, not in the panel.** `Body.locked_fields`
  is a frozenset of the same names `edits.edit_fields` returns, and every setter
  (`set_mass`, `set_density`, `set_temperature`, `set_spin`, `set_disk`,
  `set_material`/`cycle_material`) returns without writing when its **own** field is
  locked. The greyed-out stepper is the second line of defence, not the first: the
  arrow keys and the horizontal drag never pass through a button. A lock guards only
  its own field -- locking density does not stop a material swap from moving density
  -- and it reaches no further than the edit path: `collapse`, a merge, ablation and
  the pinned temperature `_refresh_kind` writes all still touch a locked field, because
  none of them is an edit. **`thermal.advance` is the one exception** -- a locked
  temperature would otherwise relax straight back to equilibrium on the very next
  frame, so `advance` skips a body with `"temp"` locked, checked after the pinned-kind
  branch so a star or hole ignores a stale lock left over from being a planet. Locks
  live on the Body beside `spin` and `disk` because the force kernel never reads them;
  nothing inherits them, and a kind change leaves them alone, so a lock on a field the
  current kind does not offer is inert rather than lost. **A Restart (`F5`) drops every
  lock**, because the scenario `world.clear()`s and rebuilds every `Body` from scratch,
  and a lock is a property of a body that no longer exists.
- **`thermal.advance` runs once a frame, never per substep.** The same rule
  `update_trails` follows, and for the same reason: the pass is O(n²), and at ten
  years a second the integrator takes thousands of substeps per frame. Ablation
  inside the substep loop therefore reads a temperature that is up to one frame
  old, which is harmless because the analytic step has already snapped a body to
  equilibrium at any time scale where a frame is long.
- Angles are radians everywhere except the source data tables in `src/scenarios.py`.
- Pixel-size constants scale by `internal_height / scale.REF_HEIGHT` (360); interface sizes
  come from `widgets.Metrics`, which derives every one of them from the font. Neither may
  hard-code a pixel count, because the `G` density toggle changes both.
- **Nothing may cache a size across frames.** The window is resizable, so the buffer can
  change shape at any frame boundary. `ui.build` already runs from the current size every
  frame; anything else sized in pixels (font, camera, starfield, sprite cache) is rebuilt
  by `App._build_render_stack`, which `step_once` calls whenever the buffer changed.
- **Mouse coordinates are internal pixels everywhere.** `ctx.map_event` translates every
  event on the way in and `ctx.mouse_pos()` is how a wheel event finds the pointer, so no
  reader of `event.pos` — interface, tools, effects — knows the grid exists.
- **The explain layer compares orbital *elements*, never positions**
  (`src/explain.py`). The world keeps running between an edit's baseline and
  its report -- at one month a second a 350 ms burst is ten days, at ten years
  a second it is 1278 -- and elements are invariant under motion along the
  orbit, so a position comparison would report a planet's own progress round
  its orbit as though the edit had caused it. **Temperature is the one row
  this argument does not cover, so it is handled differently per body.** The
  edited body's own Temp row reads the instantaneous `world.temperature` --
  the number he actually set -- because nothing heats itself: a temperature
  edit leaves the edited body's own equilibrium bit-identical, so building
  that row on `thermal.equilibrium_temperature` would diff to nothing and the
  panel would stay silent on the edit it has the most to say about. Every
  other body's Temp row uses `thermal.equilibrium_temperature`, because their
  instantaneous values relax on their own clock and would otherwise report
  the thermal simulation's own progress as though the edit had caused it, and
  a star's comes from `thermal.pinned_temperature`, since a radiator's heat is
  set by fusion rather than by what falls on it. **What keeps the edited
  body's own instantaneous reading honest is an `_after` snapshot taken
  synchronously at the write**, not merely on the next frame: `App.step_once`
  runs `sim.advance` -- which calls `thermal.advance` once, with the *whole*
  frame's worth of simulated time -- before it calls `explain.settle`, and one
  frame is not a short time at speed (60+ simulated days at ten years a
  second), while temperature's relaxation time constant is under a thousandth
  of a day for anything glowing. A snapshot deferred to "the next frame" would
  therefore already have relaxed a hot temperature back to equilibrium before
  ever being captured -- confirmed by measurement, not assumed. So every write
  path (`EditTool._scrub`, `EditTool._collapse`) calls `ExplainRecorder.note`
  before the edit and `ExplainRecorder.record` immediately after it, in the
  same call stack, before anything -- including that frame's own `sim.advance`
  -- can move the world; `settle`, called once a frame regardless of whether
  anything was written, only checks the quiet timer and diffs the baseline
  against the already-frozen `_after`, never taking a snapshot of its own.
  **One report per burst, not per scrub notch**: the baseline is taken on the
  first edit and the report is computed after `SETTLE_MS` of quiet, in
  wall-clock milliseconds -- doubling the Sun is a dozen-plus `set_mass` calls
  through a held stepper.
- **`widgets.overlay_x` is where an overlay starts**, so the Fly readout and
  the explain panel cannot disagree about where the body-list drawer ends.
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

### Design decisions worth knowing before changing them

- **A body is heated by every other body, not by stars** (`thermal.py`). One sum:
  `T_eq^4 = (1-A)/4 * sum_j T_j^4 (R_j/d_ij)^2 + T_CMB^4`. The Sun heating Mercury
  and a molten Earth heating the Moon are the same term, which is why "a hot planet
  warms its neighbours" needed no code of its own, and why a black hole needs no
  exception — 6e-8 K over a horizon a few km wide evaluates to nothing. Fluxes add;
  temperatures do not. There is deliberately **no greenhouse term**: it would be a
  tuned per-material constant rather than derived physics, so Earth reads −19 °C,
  which is true. Reproduces the real solar system to within a few kelvin.
- **Temperature relaxes on a one-metre skin, integrated analytically**
  (`units.THERMAL_SKIN_DEPTH_M`). Heating a whole planet gives a time constant near
  a hundred thousand years — real, and invisible; real surfaces equilibrate in days
  because only a thin skin follows the sunlight. `tau = rho h c_p / (4 sigma T^3)`
  linearised at the *hotter* of the body and its target, so a body dropped to 3 K
  beside the Sun warms up instead of staying frozen. The step is
  `T <- T_eq + (T-T_eq) exp(-dt/tau)`, unconditionally stable; an explicit step
  would need `dt < tau`, and tau is under an hour for anything glowing. Beware when
  writing tests: at 3000 K tau is 0.008 days, so a one-day step lands on equilibrium
  to the last bit of a float64 and there is no monotonic tail left to assert on.
- **An impact heats what survives it** (`thermal.IMPACT_HEAT_FRACTION`). 30% of the
  Q_R `collisions.py` already computes, over the material's specific heat. A
  Moon-forming impact puts ~1900 K into the remnant, which is what really happened.
- **A hot body's colour comes from the Planck locus, not the palette**
  (`render/palette.blackbody_rgb`). Temperature is quantised to 100 K before it
  reaches the sprite cache key, for the same reason the lighting angle is bucketed
  to 16 points: a body cooling continuously would miss the cache every frame. The
  blend runs before the star branch too, so a red dwarf is red and a blue giant is
  blue off the same helper.
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
- **Destruction does not require contact** (`destruction.py`). A body can be torn apart by
  a tide — raised by a star or a black hole — or boiled away by a star's heat
  (`equilibrium_temperature` against `Material.vapour_temperature_k`), without ever
  touching what does it. `_bodies_and_attractors` splits the two roles: **attractors**
  (stars and black holes) raise tides (`tidal_stress_ratio`, gravity-only tidal stress is
  a pure density ratio, independent of the body's own size — the classic Roche result);
  **stars** alone radiate. A black hole is never a victim, and a star may be a victim only
  of a black hole. Like `CollisionResolver.resolve()`, only the single worst violation is
  handled per call, since removing a body shifts every index after it.
- **Tidal fragments need immunity from immediate re-shredding** (`destruction.py`).
  `tidal_disrupt` places debris about 1.15 body-radii from where the parent died, which
  is deep inside the tidal radius that killed it. The destruction loop skips a body
  while `world.time < body.no_collide_until`; the immunity `tidal_disrupt` already
  grants is about six times longer than the transit through the tidal zone, so fragments
  clear it before being re-checked. (Debris inherits an orbit whose periapsis is still
  inside the tidal radius, so it returns and is shredded again on later passages — the
  world does still fill over several orbits. That is real physics, not a bug, and
  terminating it would need a dust floor in `tidal_disrupt` that nobody has specified.)
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
- **`s` is internal pixels per point, and only `native` or 1 is ever offered** (`app/config.py`).
  The renderer reports the display's density (`coordinates_from_window`); the software
  surface is `window * s`, uploaded into a streaming `Texture` every frame. `G` toggles
  native and 1 -- half the pixels is the escape hatch when the physics budget is saturated
  at fullscreen. Text is sized in points; `font_px` multiplies by `s`. See the module
  docstring for the measured numbers.
- **The window is a `pygame.Window(allow_high_dpi=True)`, never `display.set_mode`.**
  Only a `Renderer` reaches device pixels; a window surface is in points on every
  platform, which is why the original pygame (no high-DPI flag at all) was replaced by
  pygame-ce. A resize arrives as `WINDOWSIZECHANGED` (`event.x`, `event.y`), not
  `VIDEORESIZE`; `step_once` also polls `ctx.size_changed()` once a frame for the
  resizes no event announces.
- **Nothing may call `pygame.Window.from_display_module()`.** The object it returns
  destroys the real SDL window when it is garbage-collected, and the program segfaults
  some frames later inside `pygame.event.get`. `DisplayContext` creates the one `Window`
  for the process, keeps it for its whole life and destroys it in `close()`;
  `tests/test_layering.py` greps `src/` for the forbidden call.
- **The interface has a smallest window it fits in, in points** (`config.MIN_INTERNAL`,
  `min_internal(s)` scales it by the density). `tests/test_layout.py` builds every panel
  of every tool at the floor at both densities and fails if anything lands outside.
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
- Two mixins carry the behaviour more than one tool needs, so it exists once.
  **`DragPan`** is the left button that pans on a drag and selects on a click: call
  `_pan_event` from `on_event` and override `_on_click`. **`TargetLock`** is the
  `X`-then-click target: it owns `target_id` and `picking_target`, and gives
  `_target_index`, `_target_label` (set `no_target_label`) and `_pick_target`. Each
  mixin's `__init__` is called explicitly from the tool's own, so the order the
  attributes appear in is the order they are read in.

### Adding a control

Add a `Control` in the relevant `_build_*` method of `src/app/ui.py` and a branch in
`UiLayer.perform`. Give it a `hotkey` only if the key actually exists, and add the key to
`hud.KEY_HELP` — the parity test reads that table. `enabled=False` disables it everywhere,
including `perform`, which is the door the held-repeat and the tests come through.

A tool panel row is not free: `tests/test_layout.py` checks that the Edit panel stops
above the transport bar at `config.min_internal(3)`, where there is room for six rows
and no more. That is why "Unlock all" shares the Collapse row rather than taking one of
its own.

## Docs

`docs/superpowers/specs/` holds the design (Addendum A covers the clickable interface),
`docs/superpowers/plans/` the implementation plan with its global constraints (Phase 2 is
the interface). Module docstrings carry the reasoning for most decisions and are worth
reading before changing the code under them.
