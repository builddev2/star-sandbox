# Solar System Sandbox — Design

Date: 2026-08-18

## Purpose

A 2D pixel-art solar system sandbox for an 8-year-old, played alongside a parent.
The child watches the real solar system run, throws new bodies into it, edits the
planets that are already there, and smashes things together.

The premise: nothing is hand-animated. Every orbit shown is the consequence of
integrating real gravity from real J2000 orbital elements. Neptune's year comes out
at 165 years because the physics says so, and `selftest.py` asserts it.

## Current state

The physics core already exists (873 lines, all pure numpy, no pygame):

| Module | Responsibility |
|---|---|
| `units.py` | AU / solar-mass / day unit system, G, sphere geometry, escape velocity, binding energy, Hill radius |
| `world.py` | Structure-of-arrays state; swap-remove with stable body ids |
| `integrator.py` | Velocity Verlet, power-of-two block timesteps, vectorised O(n^2) force kernel |
| `orbits.py` | Kepler solver, elements <-> state vectors, dominant-attractor detection |
| `scenarios.py` | Real J2000 elements: Sun, 8 planets, 6 major moons |
| `materials.py` | Seven materials with real densities, tones, surface styles, blurbs |
| `body.py` | Per-body identity, angle-sampled orbit trails |

Nothing renders, nothing accepts input, there is no collision handling and no way
to run it. This spec covers the layer that makes it a program.

## Scope

### In

- Watch the real solar system evolve, with time controls
- Pan, zoom, follow a body, select a body, read its real numbers
- Throw new bodies in with a live exact-orbit aiming preview
- Edit existing bodies' mass, density and material; zero their velocity
- Collisions: merge, or shatter when the impact exceeds binding energy
- Pixel-art rendering at Retina resolution, fullscreen

### Out (v1)

Save/load. Tides, rotation, axial tilt, temperature or climate. Sound. Orbital
inclination — the simulation stays strictly 2D. A populated asteroid belt.
Relativistic corrections. Any form of sharing or multiplayer.

## Architecture

Three layers. The bottom one never imports pygame, and `selftest.py` asserts that
by checking `sys.modules` — this is what keeps the orbital mechanics testable
headlessly and is the single most important structural rule in the project.

```
src/                     physics — pure numpy, headless
  units.py  body.py  materials.py  world.py            exist
  integrator.py  orbits.py  scenarios.py               exist
  collisions.py          NEW  detection, merge, shatter
  sim.py                 NEW  Simulation: owns World + Integrator + collisions
src/render/              everything that touches pixels
  camera.py              world <-> screen, log zoom, follow, pan
  scale.py               display-radius law, true-scale toggle
  palette.py             theme colours (materials.py defers to this by design)
  sprites.py             procedural pixel discs, LRU cache
  pixelfont.py           embedded 5x7 bitmap font
  starfield.py           parallax star background
  draw.py                world rendering: starfield, trails, bodies, overlays
  hud.py                 top bar, inspector panel, hint line, help overlay
src/app/
  main.py                entry point, frame loop
  state.py               selection id, follow id, active tool, time scale, toggles
  input.py               event routing, keybinding table
  tools.py               Tool base, SelectTool, LaunchTool, EditTool
selftest.py              headless physics + collision assertions
run.command              double-clickable macOS launcher
```

### The swap-remove discipline

`World.remove` swap-removes, so array indices are not stable across a frame.
Every reference held across frames — selection, camera follow, the trajectory
predictor, collision bookkeeping — stores a body **id** and resolves it through
`World.index_of` each time it is used. Violating this is the single likeliest
source of bugs in this codebase, because a mid-frame merge silently reshuffles
rows. `selftest.py` covers it.

## Display

### Resolution

**Resolved by measurement, 2026-08-18.**  SDL reports only logical points --
`get_desktop_sizes()` gives 1728x1117 on a 16" MacBook Pro whose panel is
3456x2234, `get_window_size()` likewise, and `WINDOW_ALLOW_HIGHDPI` does not
exist in pygame 2.6.  Individual panel pixels are therefore unaddressable.

That turns out to be the right answer rather than a limitation: at one internal
pixel per logical point, macOS renders each of ours as an exact 2x2 block of
panel pixels.  A clean integer upscale, at the finest grid available.  So
**S = 1 is the default**, giving a 1728x1084 buffer, and `G` cycles S through
1, 2, 3.

Two consequences the measurement forced:

- **Text is sized in logical points**, not device pixels, because points are the
  unit the platform is calibrated in.  14 points at S=1.
- **A fullscreen window is not the desktop size.**  The menu-bar/notch strip
  costs 33 points of height, so a buffer sized to the desktop gets scaled by
  0.97 -- non-integer, smearing the very pixel grid it exists to preserve.  The
  window is therefore opened once, its true size read back, and the buffer
  re-derived from that.  Verified: S=1, 2 and 3 all upscale by exactly 1.0, 2.0
  and 3.0.

The display opens with pygame's `SCALED` flag so SDL performs the upscale on the
GPU.  Software-blitting would cost several milliseconds a frame; this costs
nothing.  Fullscreen by default, `Cmd+F` toggles.

### The size law

At true scale a solar-system view makes Earth about 0.001 pixels wide, so some
cheat is mandatory. This one is continuous and preserves size ordering:

```
u        = internal_height / 360            resolution unit
true_px  = zoom * r_true
floor_px = 2.0 * u * (r_true / r_earth) ** 0.35    clamped to [1.5*u, 14*u]
draw_px  = max(true_px, floor_px)
```

Zoomed out, `floor_px` governs: the Sun draws about 10.6u, Jupiter 4.7u, Earth
2.0u, the Moon 1.5u. Ordering is legible, but nothing inflates so far that a
planet's disc swallows the orbits around it — the failure mode of naive log
exaggeration, which would draw Earth's disc a hundred times wider than the
Moon's entire orbit.

Zoom in and `true_px` overtakes `floor_px`, so real scale takes over
**continuously**. There is no mode switch and nothing pops.

`R` ("real size") sets the floor to exactly 1 physical pixel. The entire solar
system collapses to single dots at once. That is the lesson, and it is one line.

### Text

Text size is decoupled from the pixel grid so a finer grid never costs
legibility:

```
text_scale = max(1, round(24 / (7 * S)))
```

Glyphs land at 21-28 physical pixels tall for every value of S. The font is an
embedded 5x7 bitmap covering ASCII 32-126, kept in `pixelfont.py` as compact
in-code data — no asset files, no licensing, perfectly crisp at integer scales.

### Sprites

Bodies are procedurally generated pixel discs, never authored assets — the child
creates bodies at arbitrary sizes and materials, so drawing must be generative.
Each is a filled disc in the material's mid tone, shaded with its dark tone away
from the nearest star and highlighted with its light tone toward it, plus
detail from `Material.style`:

- `cratered` — scattered darker dots
- `banded` — horizontal bands alternating tones
- `cracked` — light fracture lines
- `turbulent` — noise swirls

Stars additionally get a bright core and an additive radial glow halo. Below
3 pixels all detail is skipped and the body is a single coloured dot.

Cached in an LRU of 512 entries keyed by `(material_key, style, seed,
int(draw_px), lit_angle_bucket)`, with the lighting angle bucketed to 16
directions to keep the hit rate high while the camera moves.

Saturn is drawn with rings. This requires one new `rings` slot on `Body`.

### Trails

`body.Trail` already samples by angle swept about the dominant attractor, which
is what makes Mercury's and Neptune's trails equally smooth. Rendering splits
each trail into 8 chunks drawn with `pygame.draw.lines`, each chunk a step
dimmer toward the oldest. Per-segment fading would mean ~500 draw calls per
body and cost roughly 10ms a frame; 8 chunks costs 8 calls and is
indistinguishable at this resolution.

### Starfield

300 stars from a fixed seed, drawn as single pixels with varied brightness,
parallaxed at 0.02 so panning gives a subtle depth cue without implying the
stars are nearby.

## Simulation loop

```
poll events -> input router -> active tool, or global keys

advance:
    target = time_scale * frame_dt              days per real second
    budget = 12 ms wall clock
    advanced = 0
    while advanced < target and elapsed < budget:
        integrator.choose_k(world)
        dt = min(integrator.dt, target - advanced)
        integrator.step(world, dt)
        collisions.resolve(world, integrator, dt)
        advanced += dt
    hud_ratio = advanced / target

update trails
render -> starfield, trails, bodies, tool overlay, selection, HUD
present
```

The wall-clock budget is not optional. At 10 yr/s the integrator needs thousands
of substeps per frame; without a cap the window stops responding and reads as a
crash. With it, the simulation degrades to slow motion and the top bar reports
the shortfall as an actual-vs-requested ratio.

Time scale ladder, stepped by `[` and `]`:
1 hr/s, 6 hr/s, 1 day/s, 3 day/s, 1 week/s, 1 month/s, 3 month/s, 1 yr/s,
3 yr/s, 10 yr/s.

## Collisions

### Detection

Endpoint distance checks miss fast bodies that jump clean past a planet within
one substep — precisely what a hard-thrown rock does. Detection is instead a
closest-approach test along each pair's relative motion during the step: for
relative position `r0` and relative velocity `v`, find `t*` in `[0, dt]`
minimising `|r0 + v*t|`, and register a collision when that minimum falls below
`r_i + r_j`. Exact for linear motion, one vectorised O(n^2) pass, cannot be
fooled by speed.

Pairs sharing a non-zero `event_id` are skipped while `world.time <
no_collide_until`.

Colliding pairs are resolved in order of `t*`, earliest first.

### Merge or shatter

```
mu_red     = m1*m2 / (m1+m2)
impact_ke  = 0.5 * mu_red * v_rel**2
binding    = units.binding_energy(m_big, r_big)

impact_ke <  binding  ->  merge
impact_ke >= binding  ->  shatter
```

**Merge.** Mass and momentum conserved exactly. Density becomes the mass-weighted
mean of the two; radius follows from `units.radius_from_mass_density`, so the
result is physically sized. The survivor keeps the more massive body's name,
material, seed and trail, and becomes a star if the combined mass exceeds
`units.STAR_MASS_THRESHOLD`. Implemented as a mutation of the larger body's row
plus a `remove` of the smaller.

**Shatter.** Total mass splits into 6-12 fragments — one large remnant carrying
30-50% of the mass depending on how far the impact exceeded binding, the rest
distributed by a seeded power-law-ish split. Fragments are placed on a circle of
roughly the larger body's radius about the impact point and kicked radially
outward at a spread around `units.escape_velocity(M_total, r_big)`.

Fragment velocities are then corrected so that the summed momentum equals the
pre-collision momentum **exactly**. All fragments share a fresh `event_id` and a
`no_collide_until` of `world.time + 5 * r_big / kick_speed`, long enough to
separate and short enough that re-accretion under gravity begins promptly.
Fragments take `KIND_DEBRIS` and the parent's material.

**Honest caveat:** shattering is not energy-conservative. Fragment kinetic energy
is invented to look plausible, not derived from a fracture model — real impact
physics is an active research field. Momentum conservation is exact and is what
the tests assert.

**Capacity.** `world.MAX_BODIES` is 200. Fragment count shrinks toward 6 as the
world fills, and the collision falls back to a merge when fewer than 6 slots
remain.

After any collision, `integrator.refresh(world)` recomputes cached accelerations.

## Interaction

Each tool is a class owning its own event handling and its own preview drawing,
so adding tools later is a new file rather than surgery on a large event handler.

### SelectTool (`1`, default)

Click a body to select. `F` follows it. `Tab` cycles. Wheel zooms at the cursor.
Dragging empty space pans. `Esc` deselects.

### LaunchTool (`2`)

Presets in the bottom bar, chosen by number key or click:

| Preset | Material | Radius | Mass |
|---|---|---|---|
| Asteroid | rock | 50 km | from density |
| Rocky planet | rock | 6371 km | from density |
| Ice planet | ice | 2500 km | from density |
| Gas giant | gas | 69911 km | from density |
| Star | hydrogen | 0.3 solar radii | 0.3 solar masses |

Click-and-hold places the body, drag sets velocity, release launches. During the
drag:

- The **exact** predicted orbit is drawn, computed algebraically via
  `orbits.elements_from_state` against the dominant attractor and sampled at 128
  true anomalies — not integrated. It re-solves every frame, so the ellipse
  stretches and swings live under the cursor, opening into a hyperbola when the
  throw exceeds escape velocity. This is where orbital intuition actually comes
  from, and it is nearly free because `orbits.py` already does the hard part.
- Speed reads out in km/s beside the drag arrow.
- Holding `Shift` snaps to exactly circular velocity via
  `orbits.circular_velocity`.

### EditTool (`3`)

Select any body, including ones from the original scenario, and edit it.

**Invariant: mass and density are edited, radius is always derived** through
`units.radius_from_mass_density`. There is no way to produce an inconsistent
mass/radius/density triple, so the physics stays honest regardless of what the
child does.

- **Mass** — logarithmic drag-scrub, clamped to `[1e-16, 50]` solar masses.
- **Density** — linear drag-scrub, clamped to `materials.MIN_DENSITY_GCM3` ..
  `MAX_DENSITY_GCM3` (0.05 .. 25 g/cm^3). The composition label follows via
  `materials.nearest_by_density`.
- **Material** — cycles the seven presets, snapping density to that substance's
  real value.
- **`0`** zeroes the body's velocity. One line, and the best trick available:
  stop the Earth and watch it fall into the Sun.

Radius, surface gravity and escape velocity update live in the panel during a
scrub, and the drawn body resizes immediately.

Crossing `units.STAR_MASS_THRESHOLD` (0.08 solar masses) flips `kind` to star:
the body starts glowing and starts lighting the others, so shadows across every
moon visibly swing round. The panel shows current distance from that threshold,
making it a target to aim for.

**Every edit calls `integrator.refresh(world)`** — mass changes the force field
for every body at once, and cached accelerations would otherwise be a step stale.

## HUD

- **Top bar** — elapsed time (`Year 12.4 · Day 4521`), time scale, actual/requested
  ratio when the frame budget saturates, body count, FPS.
- **Right panel**, when a body is selected — name, kind, material; mass in Earth,
  Jupiter or solar masses as magnitude dictates; radius in km and Earth radii;
  density in g/cm^3; surface gravity in g; escape velocity in km/s. Then the
  orbit block: attractor name, semi-major axis, eccentricity, period,
  periapsis/apoapsis, current speed. Then the material's one-line blurb.
- **Bottom** — active tool and its controls.
- **Help overlay** on `?` — the full keybinding table.

### Keys

`Space` pause · `[` `]` time scale · `1` `2` `3` tools · `F` follow ·
`Tab` cycle selection · `Esc` deselect · `T` trails · `R` real size ·
`L` labels · `G` pixel grid · `0` zero velocity · `Del` delete selected ·
`F5` reset scenario · `?` help · `Cmd+F` fullscreen · `Cmd+Q` quit.

## Testing

Test-driven: `selftest.py` is written before the code it covers. It runs headless
and imports no pygame.

1. **Import discipline** — after importing every `src/*.py` physics module,
   `pygame` is absent from `sys.modules`.
2. **Kepler round-trip** — elements -> state -> elements is the identity to 1e-10.
3. **Real periods** — Earth's year is 365.25 days and Neptune's is 165 years,
   each within 0.5%, measured by integration rather than by formula.
4. **Energy conservation** — a two-body system over 100 orbits holds
   `|dE/E| < 1e-6`. This is the test that catches the inward spiral that
   velocity Verlet was chosen to prevent.
5. **Merge conservation** — mass and momentum conserved exactly.
6. **Shatter conservation** — mass and momentum conserved exactly; fragment count
   within 6..12 and within remaining capacity.
7. **Fragment cohesion** — fragments do not re-merge inside their
   `no_collide_until` window.
8. **Swap-remove discipline** — body ids resolve to correct rows after
   interleaved adds and removes.
9. **Anti-tunnelling** — a body moving fast enough to pass a planet entirely
   within one substep still registers the collision.
10. **Block timestep** — `choose_k` never moves more than one notch per call.

Plus a smoke test booting the real application under `SDL_VIDEODRIVER=dummy` and
rendering 100 frames, so rendering crashes surface without needing a display.

## Delivery

`run.command`, chmod +x, double-clickable in Finder: activates the venv and runs
`python -m src.app.main`. A README covers first-time setup
(`pip install -r requirements.txt`) and the keybindings.
