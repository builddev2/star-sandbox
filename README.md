# Solar System Sandbox

A 2D pixel-art solar system you can break.

The planets are not animated. Each one starts from its real J2000 orbital
elements and moves because gravity is being integrated, which is why Neptune's
year comes out at 165 years, why Mercury's orbit is visibly off-centre from the
Sun, and why throwing something heavy at Jupiter changes everything else's orbit
too.

## Running it

Double-click `run.command`. The first launch sets up a virtual environment;
after that it opens straight away.

From a terminal instead:

```bash
pip install -r requirements.txt
python -m src.app.main
```

## Controls

Everything is a button. Nobody has to be told a shortcut before they can play:
the tools are down the left, the bodies are listed by name beside them, time
runs along the bottom, and what you can do to whatever you have selected is
under its readout on the right. Hover anything to find out what it does.

Every button also has its key printed on it, which is how you stop needing the
buttons.

| | |
|---|---|
| `Space` | pause and resume |
| `[` `]` | slower / faster (1 hour per second up to 10 years per second) |
| Wheel, `+` `-` | zoom |
| `H` | back to the whole solar system |
| Drag | pan |
| Click | select a body; double-click flies to it |
| `Tab` | cycle through bodies |
| `B` | the list of bodies |
| `F` | follow the selected body |
| `1` `2` `3` | Look / Throw / Edit |
| `0` | stop the selected body dead |
| `T` `R` `L` | trails / real sizes / labels |
| `G` | pixel grid size |
| `W` | window size -- two thirds, half, a third, full width |
| `Del` | delete the selected body |
| `F5` | start over |
| `?` | controls |
| `Cmd+F` | fullscreen |
| `Cmd+Q` | quit |

The quit button, top right, asks once before it goes: nothing is saved, and it
sits next to buttons a small hand is reaching for anyway.

## Things worth trying

- Pick Earth out of the list, hit **Stop**, and watch it fall into the Sun.
- Turn on **real sizes**. Everything in the solar system becomes a single dot at
  once. That is what the real scale looks like; the rest of the time the sizes
  are exaggerated so you can see anything at all.
- **Throw**, pick a gas giant, and drag out in space. The green ellipse is the
  exact orbit it will follow, worked out algebraically as you aim, not guessed.
  Pull further and it turns red: that one is leaving forever. The colour changes
  at exactly escape velocity.
- Switch on **Circular orbit** (or hold `Shift`) while aiming, and whatever you
  throw drops into a perfect round orbit.
- **Edit** Jupiter, and hold the mass **+** down. Somewhere around 84 times
  heavier it ignites into a second star -- and every shadow in the solar system
  swings round, because bodies are lit by whichever star is nearest.
- Throw something fast enough at a planet and it will not merge. If the impact
  carries more energy than the planet's own gravity is holding it together with,
  it shatters into fragments that fly apart and then slowly pull themselves back
  into a new body.
- Zoom into Jupiter to see Io, Europa, Ganymede and Callisto in the right order
  at the right distances.

## Tests

```bash
python selftest.py
```

The physics layer never imports pygame, which is what lets the orbital mechanics
be tested without a display. One of the tests enforces exactly that.

Some of what the suite checks: Earth's year is 365 days and Neptune's is 165;
Jupiter, Saturn and Neptune's periods all land within 0.5% of their real values;
the predicted orbit drawn while aiming matches 4000 steps of actual integration
to within 0.02 AU; collisions conserve mass and momentum exactly; and the
integrator's energy error grows sublinearly rather than spiralling planets
inward.

The interface is covered the same way: every button is pressed through the real
frame loop, no two controls are allowed to overlap or hang off the screen, and
clicking a button is checked not to also throw a comet through it.

## How it is put together

| | |
|---|---|
| `src/` | physics: units, gravity, orbits, collisions, edits. Pure numpy. |
| `src/render/` | camera, sprites, HUD, buttons. Everything that touches pixels. |
| `src/app/` | frame loop, input, the interface, one class per tool. |

Buttons and keys are never two implementations of the same feature: a control
describes itself and names an action, and that action is the method the key
already called. `tests/test_ui.py` asserts every shortcut has a button.

Design notes and the implementation plan are in `docs/superpowers/`.

## Known gaps

- Orbit trails are stored in world coordinates, so when you follow a planet its
  moons have no visible trails -- their absolute paths are dominated by the
  planet's own motion around the Sun. Trails drawn relative to the followed body
  would fix this.
- The simulation is strictly 2D: no orbital inclination, so every body shares
  one plane.
- Nothing is saved. Closing the window starts over.
