# The Measure tool: a ruler, and an orbit timer that checks Kepler

## Why

The sandbox already *uses* real numbers everywhere; this lets him take them out
and check them.  Two things a textbook quotes and he can now measure himself:

- **How far, and how long light takes.** "The Sun is 8.3 light-minutes away" is
  a fact he can read off the ruler between Earth and the Sun.
- **Kepler's third law.** Time a planet all the way round, set the measured
  period beside what `P = 2 pi sqrt(a^3 / GM)` predicts, and turn it round:
  `a^3 / P^2` in AU and years *is* the mass of what it orbits, in suns.  Timing
  the Moon weighs the Earth.

The difference between measured and predicted is not an error to hide: it is
every other body pulling on the orbit.  The Moon, pulled by the Sun, comes out
about 1% slow against its own two-body prediction.

## What it is

A fifth tool, **Measure** (key `5`), with a ruler icon on the hotbar.

**Ruler.** From the selected body to a target locked with `X`-then-click
(`TargetLock`, the same mixin Throw and Fly use).  A line is drawn between the
two; the overlay readout gives:

- centre-to-centre distance (`hud.format_length`);
- the surface-to-surface gap, only once the two are within 0.02 AU, where
  `format_length` switches to kilometres and the gap starts to differ from
  the centre distance in a digit anyone reads;
- light travel time over the centre distance (`hud.format_light_time`, new);
- closing or opening speed along the line between them (`hud.format_speed`).

**Orbit timer.** `P` (or the "Time its orbit" button) starts timing the selected
body round `orbits.dominant_attractor`.  The overlay shows how far round it is,
then, after each full lap, the measured period, Kepler's prediction, the percentage
between them, and the mass `a^3 / P^2` implies (`hud.format_mass`).  A stop
button (no key) clears it.

## Decisions

- **Measured by angle swept, sampled once a frame**, like trails.  A lap is
  complete when the accumulated angle about the attractor reaches `2 pi`; the
  crossing time is interpolated linearly between the two samples that straddle
  it.  Prototyped: the error is 0.035% at 7-day frames, against 1.6% without
  the interpolation.
- **"Too fast" is refused, not guessed.** The trusted gap is scaled by
  periapsis rate, not a flat quarter of the mean period -- a body runs fastest
  there, so a flat limit would let an eccentric orbit's gap straddle periapsis
  and count backwards.  Past it, the direction of travel between samples is
  ambiguous.  The timer says so, restarts, and resumes when the child slows
  time down.  At the default month-a-second, Io (1.77 d) is too fast and the
  Moon is not.
- **Retrograde counts.** The swept angle is signed and laps count its magnitude.
- **The timer lives on `AppState`, not the tool.** A tool is rebuilt whenever it
  is picked; timing Neptune should survive a look at the Edit panel.  `step_once`
  samples it after `update_trails`.  Restart and Load clear it, like the explain
  report.  It is **not saved**, because it is a stopwatch, not part of the world.
- **The timer holds ids, never rows** (swap-remove).  A body or attractor that
  disappears reads "lost"; a body whose dominant attractor changes (a craft after
  a slingshot) reads "moved on" and stops.
- **Kepler's prediction uses each lap's time-weighted mean `a`**, frozen the
  instant the lap completes, with `mu = G (M + m)` at that crossing -- the
  osculating `a` wanders across a perturbed lap (the Moon's swings enough to
  walk the prediction from 27.22 to 28.09 days within one lunar month), and a
  number read off mid-wander is not one the timer can stand behind.
- **Physics in `src/measure.py`** (pure numpy, no pygame), formatters in
  `render/hud.py`, the tool in `app/tools.py`.
- **The readout is an overlay, not panel rows.** The panel carries two rows (the
  target lock, and time/stop).  The readout is drawn at `widgets.overlay_x`,
  below the explain panel when one is up, the same as Fly's.

## Out of scope

Saving a running timer; a ruler between two arbitrary points in space; timing
more than one body at once; synodic periods.
