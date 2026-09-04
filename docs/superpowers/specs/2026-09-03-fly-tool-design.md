# The Fly tool: maneuver nodes, delta-v, and a space shuttle

## What this is for

The sandbox can throw a rock and watch it go. It cannot *steer* anything once it is
moving, which means the one subject the audience has actually asked about -- space travel
-- is the one thing the software cannot show.

This adds a spacecraft you can fly: a craft placed in orbit, a **maneuver node** on its
predicted path, and burns measured in metres per second out of a finite tank. The node is
the whole idea. Pick a point on your own orbit, say how hard to burn there, and watch the
new orbit appear *before* committing to it. That single mechanism is what turns a gravity
simulator into a spaceflight simulator, and it is what Kerbal Space Program teaches
orbital mechanics with.

The audience is an eight-year-old who is ahead of the curriculum in astrophysics and
mathematics. So the numbers are real, the transfer solver shows its answer rather than
hiding it, and the gravity assist is drawn as the velocity triangle it actually is.

## What already exists, and what does not

`src/orbits.py` is most of the way there and this design leans on it hard:

| Already written | Used for |
|---|---|
| `elements_from_state` | orbit from a state vector, exact |
| `conic_points` | drawing an arbitrary conic, bound or hyperbolic |
| `propagate` | exact Kepler propagation, bound **and** hyperbolic |
| `_orbit_flight_time` | days to sweep between two true anomalies |
| `dominant_attractor` | which body a craft is orbiting, by sphere of influence |
| `circular_velocity` | placing a craft on a circular orbit |

What is missing is a body that stays steerable, a burn attached to a *future* point rather
than to the moment of release, and the arithmetic that turns those into a transfer.

## Architecture

Three pieces, one per layer, matching the split `tests/test_layering.py` enforces.

### `src/maneuver.py` -- new physics module

Pure numpy plus `orbits` and `units`. **Never imports pygame**, and must be added to
`PHYSICS_MODULES` in `tests/test_layering.py` or the enforcement silently stops covering
it. All the interesting mathematics lives here so that it is testable headless.

```python
@dataclass(frozen=True)
class Maneuver:
    due_time: float      # absolute world days, not an offset
    prograde: float      # AU/day, signed: negative is retrograde
    radial: float        # AU/day, signed: positive is outward
```

`due_time` is absolute rather than a countdown so that it does not have to be rewritten
every frame as the clock advances; the countdown the panel shows is
`due_time - world.time`, computed where it is displayed.

Functions:

- `node_state(world, index, maneuver)` -> `(attractor, rel_pos, rel_vel)` at the node,
  by propagating the craft's current two-body state forward to `due_time`.
- `burn_velocity(rel_pos, rel_vel, maneuver)` -> the relative velocity after the impulse.
  Prograde unit vector is the normalised velocity; radial is the normalised position.
- `plan_conics(world, index, plan, samples)` -> a conic for each node in the plan, each
  propagated from the previous node's post-burn state. Drives the overlay.
- `closest_approach(mu, craft_state, target_state, horizon_days, samples)` ->
  `{time, distance, craft_rel, target_rel}`. Both bodies are propagated on their own
  two-body orbits about the shared attractor and sampled in **time**, which is the one
  parameterisation that lets two different orbits be compared at the same instant.
- `hohmann(mu, r1, r2, theta_craft, theta_target, n_craft, n_target)` -> the two burns,
  the transfer time, and **the wait until the phase angle is right**. The wait is the
  part that teaches launch windows and it is the part a naive implementation omits.
- `flyby(mu_planet, v_inf_in, periapsis)` -> turn angle `delta = 2 asin(1/e)` with
  `e = 1 + periapsis * |v_inf|^2 / mu_planet`, and the rotated outgoing `v_inf`.

`hohmann` returns Delta-v figures that must be checkable against the textbook: Earth to
Mars comes out at about 2,945 m/s for the departure burn, 2,650 m/s for the
circularisation, and roughly 259 days in transit. These numbers are the acceptance test.

### `src/body.py`, `src/world.py`, `src/materials.py`, `src/edits.py` -- the craft

A new kind, `KIND_CRAFT = "craft"`. Two new `Body` slots beside `spin` and `disk`, for the
same reason those live there -- the O(n^2) force kernel never reads them:

- `plan`: a tuple of `Maneuver`, empty by default, kept sorted by `due_time`.
- `fuel_dv`: remaining budget in AU/day.

**A craft is a one-way kind.** `edits._refresh_kind` returns early for it exactly as it
does for a black hole, so no amount of mass scrubbing can ignite a spacecraft into a star.
`edits.edit_fields` returns `("mass", "density", "temp", "material")` as for a planet;
nothing about the craft needs a special editor.

A `SPACECRAFT` material, defined outside the `MATERIALS` tuple exactly as `SINGULARITY`
is, so that `cycle_material` and `nearest_by_density` can never land on it. Aluminium
density 2.7 g/cm3, high toughness, albedo 0.30, vapour temperature 2,000 K. That last one
is not decoration: a craft flown too close to the Sun really should boil away, and
`destruction.py` will do it for free with no special case.

Tidal disruption needs no exception either. `units.strength_acceleration` scales as
`r^-1.38`, so a ten-metre hull is enormously strong against a tide, and it survives
passages that shred a moon. That is correct physics, not a carve-out.

Collisions need no exception. A craft hitting a planet is a negligible-mass projectile:
`classify()` returns accretion, the craft is absorbed, and `ImpactEvent` gives the effects
layer a crash for free.

### `src/render/scale.py` and `src/render/sprites.py` -- drawing a shuttle

**`visual_radii` gets a craft branch.** A ten-metre hull is 7e-11 AU. This is the black
hole's problem in miniature and it breaks the same two things: the craft would draw as a
single pixel, and `contact_radii_au` would cap its pick radius at something far below one
pixel, making it unselectable. `units.craft_apparent_radius()` returns a fixed stand-in
radius so drawing, picking and the contact refresh all read one helper and cannot
disagree. **Physics never reads it** -- the same rule the black hole already follows.

**The craft is drawn as a space shuttle orbiter, pointing where it is going.**

Not a dot, and not a generic triangle. The silhouette is the orbiter: a blunt nose, a
delta wing swept back to a wide trailing edge, a vertical stabiliser, and the black
thermal-tile underside against the white upper surface. It is drawn from polygons, in the
same spirit as `render/widgets.py` drawing its icons from lines rather than typing them as
glyphs -- there is no sprite sheet, and there is no font that has a shuttle in it.

- `sprites._make_craft(px, heading_bucket)` draws the orbiter nose-along-+x into a surface
  and rotates it by the bucket angle.
- `SpriteCache.get_craft(px, heading)` buckets the heading into `LIT_BUCKETS` (16)
  compass points and keys on `("craft", px, heading_bucket)`. Heading is bucketed for
  precisely the reason the lighting angle already is: a craft turning continuously along
  its orbit would miss the cache on every frame, and regenerating a sprite per frame is
  what the cache exists to prevent.
- Below `CRAFT_DETAIL_PX` (7 px) there is no room for a wing; draw a bright two-pixel dot
  instead. Above it, the shuttle.
- Blitted **normally, never with `BLEND_ADD`** -- a shuttle is an opaque object, and the
  reasoning is the black hole's, not the star's.

`render/draw.py` gets a craft branch alongside the existing black-hole branch, taking the
heading from `world.vel[i]`.

### `src/sim.py` -- executing a burn

`Simulation` gains `_next_burn_time`, the earliest `due_time` across every craft in the
world, recomputed on `refresh()` and after any burn.

Inside the substep loop, **after** `integrator.step` and before collision resolution:

```python
if world.time >= self._next_burn_time:
    self._execute_due_burns(world)
```

That is one float comparison per substep, not a scan over bodies. This matters: at ten
years per second the integrator takes thousands of substeps per frame, and a per-substep
Python loop over 200 bodies would eat the wall-clock budget many times over -- the same
reasoning that keeps `thermal.advance` and `update_trails` at once per frame.

But the check itself must be **per substep, not per frame**. At 1 year/s a frame is six
days of simulated time; a node checked only at frame boundaries would fire up to six days
late, which at Mars-transfer speeds is a miss of several million kilometres.

Executing a burn: resolve the craft's attractor, build the impulse in the world frame from
the prograde and radial unit vectors at that instant, add it to `world.vel`, deduct
`hypot(prograde, radial)` from `fuel_dv`, pop the node, and call `integrator.refresh`.
A burn that costs more than the tank holds is **scaled down to what is left** rather than
refused -- running dry mid-burn is a real thing that happens to real missions, and it is a
better lesson than a button that does nothing.

### `src/app/tools.py` -- the FlyTool

Registered in `TOOLS` under `pygame.K_4`; `TOOL_BY_NAME` follows automatically.

**On the key `4`:** while the Throw tool is active, `4` picks its Gas giant preset rather
than switching tools. This shadowing already exists for `2` and `3` and the codebase
accepts it -- `LaunchTool.on_event` deliberately leaves `K_1` alone as the escape hatch.
Fly is no different, and the palette button always works. Not a new problem, and not one
this change fixes.

## The panel: six rows, and that is a hard limit

`tests/test_layout.py` checks that a tool panel stops above the transport bar at
`config.min_internal(3)`, where there is room for six rows and no more. That constraint
shapes the whole interface, and it is why the readouts are in the overlay rather than in
the panel.

| Row | Controls |
|---|---|
| 1 | `Launch craft` &middot; `Refuel` |
| 2 | `Target: Mars` (hotkey `X`) |
| 3 | `-` `When: T+42 d` `+` |
| 4 | `-` `Prograde 1,240 m/s` `+` |
| 5 | `-` `Radial 0 m/s` `+` |
| 6 | `Hohmann to Mars` (`M`) &middot; `Clear node` (`N`) |

Every control exists in every state, greying out rather than disappearing, so the panel
does not reflow under the cursor. `enabled=False` when there is no craft selected.

**A hotkey does not inherit a control's `enabled` gate.** Keyboard events bypass
`UiLayer.perform`'s `clickable` check, so every key path must re-check its own
precondition inside `FlyTool.on_event` -- the rule `EditTool` already follows for `C`.

Scrubbing follows `EditTool`'s conventions exactly: Up/Down picks a row, Left/Right
changes it, steppers act on press and repeat while held. Delta-v scrubs linearly at 10 m/s
a notch, three notches to a click. The node time scrubs by **ratio** at 1.06 a notch, the
way mass does, because it has to span minutes to years.

## The overlay

- The craft's current orbit, in the existing dim trail colour.
- The post-burn conic for the first node, in `palette.ACCENT`.
- The conic after a second node, dimmer, propagated from the first node's post-burn state.
- A draggable node handle on the current orbit.
- Beside the node: the new periapsis and apoapsis, the countdown, and -- when a target is
  locked -- the closest approach the plan achieves and when.
- Delta-v remaining, as a bar as well as a number. `widgets.draw_bar` already exists.

**Encounter analysis is cached behind a dirty flag.** `closest_approach` samples a few
hundred propagations; recomputing it every frame for every candidate body would be
noticeable. It is recomputed when the plan, the selection or the target changes, and at
most once every 200 ms otherwise.

## The gravity assist

When the planned trajectory passes inside a body's Hill radius, the overlay draws the
velocity triangle at closest approach:

- **v_craft** -- the craft's velocity about the primary.
- **v_planet** -- the planet's velocity about the primary.
- **v_inf** -- their difference, the speed the planet actually sees.

Then the turn: `e = 1 + r_p |v_inf|^2 / mu_planet`, `delta = 2 asin(1/e)`, and `v_inf`
rotated by `delta` gives the outgoing leg. The two heliocentric speeds are labelled, so
what is on screen reads as *"arrived at 12.4 km/s, left at 17.1 km/s -- the planet's own
motion did the rest."*

This is the honest version. `|v_inf|` is unchanged by the flyby and the drawing shows
that; what changed is its direction, and therefore the sum. Nothing here is a fudge, and
all of it is algebra already available from `elements_from_state`.

## Where a craft is launched, and why not a parking orbit

`Launch craft` places the craft on a **circular orbit about the selected body's own
dominant attractor** -- the Sun, for a planet -- at that body's current orbital radius,
offset slightly ahead of it.

The obvious alternative, a low parking orbit around the selected planet, is rejected on
physics grounds. `Integrator.dynamical_time` takes the tightest pair in the *entire*
system and steps everything at that cadence. A craft 300 km above Earth has an orbital
period of about 90 minutes, which would drive the block level to its `MAX_K` ceiling and
step the entire solar system in units of seconds. The measured cost of the six existing
moons is already a factor of 64 in step size; a parking orbit would be far worse, and the
whole simulation would grind to a stop the moment a craft was created.

Starting heliocentric and co-orbital costs nothing in timestep, and it is the correct
frame for interplanetary transfers anyway -- which is what the Hohmann button computes.

## Testing

Test-driven throughout, and split along the layering boundary.

**Headless physics** (`tests/test_maneuver.py`), no pygame:

- A prograde burn at periapsis raises apoapsis and leaves periapsis alone.
- A retrograde burn from a circular orbit lowers periapsis.
- `hohmann` Earth to Mars: departure ~2,945 m/s, arrival ~2,650 m/s, transit ~259 days,
  each to a few per cent. This is the acceptance test for the whole solver.
- The phase-angle wait is right: after waiting the computed time and executing both burns,
  the craft arrives within a small fraction of an AU of the target.
- `closest_approach` on two known circular orbits returns the analytic separation.
- `flyby` turn angle: a slow pass close in turns hard, a fast pass far out barely turns,
  and `|v_inf|` is conserved to floating point.
- A burn costing more than the tank is scaled to the fuel available.
- Round-trip: `node_state` then `burn_velocity` then `elements_from_state` reproduces the
  elements the panel displays.

**Simulation** (`tests/test_sim.py` additions):

- A node fires within one substep of its due time, at every time scale including 10 yr/s.
- A node does **not** fire while the simulation is paused.
- Fuel is deducted by exactly the magnitude of the impulse applied.
- Removing a craft with a pending node leaves `_next_burn_time` consistent.

**Tools and interface** (`tests/test_tools_fly.py`, mirroring `test_tools_launch.py`):

- Every control is present and correctly disabled with no craft selected.
- Each hotkey re-checks its own precondition and is a no-op without a craft.
- Launching from Earth produces a craft bound to the Sun, not to Earth.
- The panel fits six rows at `config.min_internal(3)` (added to `tests/test_layout.py`).
- `FlyTool` is added to the tool list in `test_ui.py`'s shortcut-parity test, and `4`,
  `M`, `N` are added to `hud.KEY_HELP`.

**Rendering** (`tests/test_sprites.py` additions):

- A craft sprite is generated, is not blank, and is cached across repeated heading values
  within one bucket.
- `visual_radii` returns the stand-in radius for a craft and the true radius for
  everything else.

**Layering**: `src.maneuver` added to `PHYSICS_MODULES`.

## Deliberately not in this change

- **A second craft type, staging, or a rocket equation.** Delta-v is a budget, not a mass
  ratio. Tsiolkovsky is the right next step and it is not this step.
- **Inclination.** The simulation is planar. A plane change is the one manoeuvre this
  model cannot express, and that is a known and accepted ceiling.
- **Landing.** A craft that meets a planet is absorbed by the existing collision path.
- **Saving the craft or its plan.** The sandbox has no persistence at all yet; that is its
  own piece of work and this change does not pretend otherwise.
