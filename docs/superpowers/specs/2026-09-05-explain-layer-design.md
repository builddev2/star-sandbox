# The explain layer: saying what an edit did, and why

## The problem

The sandbox is honest and silent.  It shows real J2000 elements, real units and a
real disruption threshold, and it never once says what any of it means.  Double
the Sun's mass and Earth's year drops from 365 days to 138; the inspector shows
the new number, destroys the old one, and offers no reason.

The audience is an eight-year-old who is ahead of his age in astrophysics and
maths, and whose interest is the resource actually at risk.  What he does most is
edit things to see what happens.  Right now "what happens" is a number he cannot
compare against anything and a cause he has to already understand.

The existing design language for teaching is already in the codebase, in the Fly
tool's flyby triangle: draw the thing, put the two real speeds beside it, and the
comment says the numbers are "what make it a lesson rather than a decoration."
This spec applies that rule to editing.

## What this is not

Decomposed out, each to its own spec later (see Roadmap):

- direct-aim versus Hohmann transfer in the Fly tool,
- missions and objectives,
- a discovery log of event-triggered facts.

And explicitly out of scope here: history, a back button, per-body memory,
persistence across restart, and any explanation of throws, collisions, ignition,
tidal disruption or escape.  One report at a time, replaced by the next.

## The shape of it

A floating panel over the world, appearing when an edit settles and staying until
the next one or until it is closed.  It never pauses, never blocks a click, and
never moves.

```
  WHAT CHANGED                    [x]
  ----------------------------------
  You doubled the Sun's mass.

  Earth
    year      365 d  ->   138 d
    stretch    0.02  ->   0.49
  Mars
    year      687 d  ->   241 d
  Jupiter
    year     11.9 y  ->   4.4 y

  A heavier Sun pulls harder, so
  every planet falls inward and
  speeds up.  Round orbits became
  long ovals.

  [ Show the maths ]
```

The ripple is the whole point.  "You made the Sun heavier" is not a lesson,
because he is the one who did it.  "...and Earth's year is now 138 days" is.

## Architecture

Three pieces, on the three existing layers, following the `ImpactEvent` ->
`render/effects.py` contract exactly: physics computes a plain-data record, the
render layer decides what it looks like.

### `src/explain.py` -- pure, no pygame

`tests/test_layering.py` runs a subprocess that fails the build if anything under
`src/*.py` imports pygame.  This module stays inside that rule, which means it
returns numbers and format tags, never formatted strings.

```python
snapshot(world) -> Snapshot
```

Per body id: name, kind, mass, radius, density, instantaneous temperature,
equilibrium temperature, and the osculating elements about its dominant
attractor (`a`, `e`, `period`, `periapsis`, `apoapsis`, `bound`, attractor name).
Keyed by `body.id`, never by index -- `World.remove` is a swap-remove, and a
snapshot outlives at least one frame.

```python
diff(before, after, edited_id, field) -> Report | None
```

```
Report  = (action: str, changes: [Change], why: [str], maths: [Formula])
Change  = (name: str, is_edited: bool, rows: [Row])
Row     = (label: str, before: float, after: float, format: str)
Formula = (general: str, substituted: str)
```

`format` is one of `"mass"`, `"length"`, `"period"`, `"temperature"`,
`"density"`, `"plain"`.  The render layer maps each to the `hud.format_*`
function that already exists for it, so there is exactly one place that knows
what a mass looks like as text.

`None` when nothing moved past the noise floor.

**Comparison is by orbital element, not by position.**  The world keeps running
between the baseline and the report -- at one month per second, a 350 ms burst is
ten days of simulated time -- and elements are invariant under motion along the
orbit.  Comparing positions or speeds would report the planet's own progress
round its orbit as though the edit had caused it.

**Temperature rows use `thermal.equilibrium_temperature`, not
`world.temperature`.**  For the same reason: temperature relaxes on its own clock
(a skin-depth time constant under an hour for anything warm), so the
instantaneous value drifts on its own during the burst.  The equilibrium value is
a function of geometry and radiators, so it moves only when the edit moves it.
This also makes the temperature ripple work for free: `thermal.py` heats every
body by every other body, so making Earth glow really does raise the Moon's
equilibrium temperature, and the panel will say so without a line of code
specific to that case.

**Ranking.**  The edited body leads, always.  Every other body is scored by its
largest relative row change; the top three are kept; anything whose every row
moved by less than 1% is dropped.  So editing Pluto reports on Pluto and nothing
else, which is correct.

**Row labels are the inspector's own words** -- `One lap`, `Stretch`, `Closest`,
`Mass`, `Temp` -- not friendlier synonyms.  This is a deliberate change from the
approved mockup, which said "year": two names for one quantity is the same defect
`EditTool`'s closing comment names when it refuses to print a number the panel
already shows, and he should learn one word per thing rather than two.

**Which rows a body offers** follows the kind split `hud._rows` already makes.
The edited body shows the fields that actually moved (mass, radius, density,
temperature, horizon).  A rippled body shows its orbit rows -- year, stretch,
closest -- and its equilibrium temperature if that moved.

### The `why` rule table

Keyed on `(field, direction, role)`.  `role` is `"attractor"` when at least one
rippled body's dominant attractor is the edited body, and `"self"` otherwise --
so a mixed ripple counts as `"attractor"`, which is the case that carries the
lesson.  A table, not a chain of `if`s, so adding a case is a row.

| field | direction | role | line |
|---|---|---|---|
| mass | up | attractor | "A heavier {X} pulls harder, so everything around it falls inward and speeds up.  Round orbits become long ovals." |
| mass | down | attractor | "A lighter {X} pulls more weakly, so everything around it swings outward and slows down." |
| mass | up | self | "{X} got heavier.  Its density did not change, so it grew." |
| mass | down | self | "{X} got lighter, so it shrank." |
| density | up | self | "The same amount of stuff, packed tighter, so {X} shrank." |
| density | down | self | "The same amount of stuff, spread thinner, so {X} swelled." |
| material | -- | self | "{X} is now made of {new}.  {new} is {denser/lighter} than {old}, so it {shrank/swelled}." |
| temp | up | self | consequence lines, below |
| temp | down | self | consequence lines, below |
| spin | -- | self | "A faster-spinning hole drags space round with it.  Matter can orbit closer in before it must fall, and more of what falls comes back out as light." |
| disk | -- | self | "How much matter is falling in.  None of it changes the hole's pull -- only how brightly it feeds." |
| collapse | -- | self | "{X} collapsed into a black hole.  Its mass did not change, so nothing orbiting it moved at all -- but all of that mass now sits inside a horizon {r} across." |

The collapse line is the most valuable one in the table, because the report will
show an empty ripple and the sentence explains why an empty ripple is the right
answer.

Temperature consequences, appended in this order, only when the edit crossed the
threshold:

- above 373 K on a body whose material contains water: "Above 373 K water boils.  Any ocean {X} had is steam now."
- below 273 K likewise: "Below 273 K water freezes solid."
- above `material.vapour_temperature_k`: "Past {T} K, {material} itself boils away."
- always, on any temperature edit: "You set it, but nothing pins it there.  Sunlight decides where it settles, and it is already sliding back toward {T_eq} K."

That last line is the lesson `edits.set_temperature`'s docstring already names as
"the whole lesson" and which nothing on screen currently tells him.

### The maths

`Formula(general, substituted)`, hidden behind a button.  Rendered as two lines:
the relationship, then the same thing with his numbers in it.

- orbital period: `T = 2*pi * sqrt(a^3 / G M)` / `T = 2*pi * sqrt(0.659^3 / (G x 2.00)) = 138 days`
- circular speed: `v = sqrt(G M / r)`
- density: `rho = m / (4/3 pi r^3)`
- horizon: `r = 2 G M / c^2`
- equilibrium temperature: `T^4 = (1-A)/4 * sum (T_j^4 (R_j/d)^2)`

Which formulas appear is chosen by the same `(field, role)` key as the `why`
line.  Verified: at `a = 0.659 AU` and `M = 2.0` suns, that expression really
does evaluate to 138.2 days in the simulation's own units.

### `src/app/state.py` -- coalescing the burst

Mass scrubs by ratio at 1.06 a notch, three notches a press, repeating while
held.  Doubling the Sun is a dozen-plus separate `set_mass` calls, and a report
per call would be a strobe.

`ExplainRecorder`, owned by `AppState`:

- `note(world, body_id, field, now_ms)` -- called *before* each write.  Opens a
  burst with a baseline snapshot if none is open; if one is open for a different
  body or field, closes that one first and opens a new one.  Otherwise just
  updates `last_ms`.
- `settle(world, now_ms)` -- called once a frame from `App.step_once`.  A burst
  idle for `SETTLE_MS` (350) closes: `diff` runs, the result becomes
  `state.report` (or is discarded if `None`), and the burst is cleared.
- `dismiss()` -- clears `state.report`.

Three lifecycle cases, each of which is a crash or a lie if unhandled:

- **Restart (`F5`)** `world.clear()`s and rebuilds every `Body`, so an open burst
  and a standing report both refer to bodies that no longer exist.  `sim.reset()`
  drops both, the same way it drops selection.
- **The edited body is deleted before the burst settles.**  Snapshots are keyed
  by `body.id`, so `diff` finds nothing for it; it returns `None` and the burst
  is discarded rather than reporting on a body that is gone.
- **Explanations turned off (`Y`).**  `note` returns immediately, so no snapshot
  is ever taken and the feature costs nothing at all when it is off -- it does
  not record silently and hide the result.

Wall-clock milliseconds from `pygame.time.get_ticks()` -- the clock
`state.selected_at` already uses -- never simulated time, for the same reason
`render/effects.py` gives: this is a property of how fast a child's hand moves,
and it must behave identically at one hour per second and at ten years per
second.

`EditTool` has exactly two write choke points -- `_scrub`, which buttons, arrow
keys and the horizontal drag all funnel through, and `_collapse` -- so this is
two call sites.

### `src/render/explain.py` -- the panel

`ExplainPanel(font)` with `rect(report, metrics, overlay_x)` and
`draw(surface, report, metrics, show_maths)`.  Formats every `Row` through the
`hud.format_*` function its `format` tag names.

Anchored at `metrics.content_top`; left edge from a new shared helper:

```python
# render/widgets.py
def overlay_x(metrics, show_body_list): ...
```

`FlyTool.readout_x` already implements exactly this "start clear of the body-list
drawer" rule and will be changed to call the helper, so the two cannot disagree
about where the drawer ends -- the same argument `scale.contact_radii_au` makes
about drawing and aiming reading one function.

Width is `hud.PANEL_WIDTH_CHARS` (34), matching the inspector, so the interface
has one text-panel width rather than two.  Height is clipped to
`metrics.content_bottom`, trimming rows from the end the way `hud.visible_rows`
already trims.

**Stacking with the Fly readout.**  Both are overlays in the same corner and both
can be up at once (edit a planet, then switch to Fly).  `AppState` keeps the last
computed explain rect; `FlyTool._draw_readout` starts at its bottom plus a gap
when one is present, and at its current `y = 34` when it is not.

### `src/app/ui.py` -- the buttons

`_build_explain(state, m)`, called from `build` alongside the others:

- `explain.close`, icon `close`, on the panel's top right.
- `explain.maths`, label "Show the maths" / "Hide the maths", on the panel's
  last row, only when the report carries formulas.

And one view toggle, `view.explain`, hotkey **Y**, in the existing view cluster,
turning the whole feature off for someone who has stopped wanting it.  A
`("Y", "explanations on and off")` row goes into `hud.KEY_HELP`; `tests/test_ui.py`
reads that table and fails if a documented key has no button, which is the
intended pressure.

Branches for all three in `UiLayer.perform`.  `Y` is also handled in the global
key path, and -- per the invariant about hotkeys not inheriting a control's
`enabled` gate -- carries no precondition the button does not.

The panel takes no place in the column layout, so it cannot push the Edit panel
or the zoom cluster off screen at `config.min_internal(3)`.  That is why it
floats.

## Frame order

One line added to `App.step_once`, after `sim.advance` and before the world is
drawn:

```
... -> sim.advance -> sim.update_trails -> state.explain.settle(world, now)
    -> camera follow -> drain impacts -> draw world -> ...
```

The panel itself is drawn with the other overlays, before the HUD, so the HUD's
hint bar and the transport row stay on top of it.

## Testing

`tests/test_explain.py`, pure, no pygame, so it runs in the physics tier:

- **the worked example**: a fresh solar system, `set_mass(Sun, 2x)`, and the
  report's Earth rows read 365.3 -> 138.2 days and 0.017 -> 0.492.  Pinned to
  those numbers, since they came out of the simulation itself.
- **the edited body leads**, and the ripple is ordered Earth, Mars, Jupiter --
  by how much each moved, not by index.
- **the noise floor**: an edit to a small outer body produces a report naming
  only that body.
- **invariance**: run `world` forward a hundred days between `snapshot` and
  `diff` and assert every reported figure agrees with the un-advanced report to
  within 1% -- far tighter than the 62% change being reported, and loose enough
  for the truth that osculating elements *do* drift a little under n-body
  perturbation.  Not equality: Jupiter really does pull Earth's elements around
  over a hundred days, and a test asserting otherwise would be asserting a
  falsehood and failing at random.  This is the claim the whole design rests on,
  so it is tested rather than assumed.
- **temperature uses equilibrium**: heat Earth, advance far enough that the
  instantaneous temperature has visibly relaxed, and assert the reported ripple
  moved by less than 1% -- while asserting that a report built from
  `world.temperature` instead would have moved by more than 10%.  The second
  half is what makes the test fail if someone swaps the source back.
- **collapse**: the report has an empty ripple and the "nothing orbiting it
  moved" line.
- **rule coverage**: every `(field, direction, role)` key an edit can produce
  yields a `why` line -- no combination falls through to silence.

Burst behaviour (`tests/test_explain_burst.py`, needs the app layer, so
`SDL_VIDEODRIVER=dummy` at module top):

- a held stepper across many frames produces exactly one report;
- changing field mid-hold produces two;
- a burst that never settles produces none.

Layout and interface:

- `tests/test_layout.py` -- the panel at `config.min_internal(3)` stays inside
  `content_top..content_bottom`, clear of the body-list drawer, and clear of the
  Fly readout when both are up.
- `tests/test_ui.py` -- already enforces `KEY_HELP` parity; `Y` must have its
  button.
- `tests/test_layering.py` -- already enforces that `src/explain.py` imports no
  pygame.

## Roadmap

Later sub-projects, each its own spec, in the order recommended:

1. **Fly: why not straight there.**  Let him aim directly at Mars and burn, watch
   it miss, and compare the delta-v that would need against the transfer's.
2. **Discovery log.**  Event-triggered real facts on ignition, collapse, tidal
   disruption and disruptive impacts, collected into something he fills up.
3. **Missions.**  Objectives with success detection: reach Mars under a fuel
   budget, build a world that could hold liquid water, break Jupiter apart.
