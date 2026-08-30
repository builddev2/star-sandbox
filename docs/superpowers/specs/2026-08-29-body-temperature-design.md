# Body temperature

Every body in the sandbox carries a temperature. It is shown, it is editable, and it
is real: a body warms because light falls on it, and cools because it radiates. The
Sun heating Mercury and a molten Earth heating the Moon are the same equation.

## Why this shape

The simulation already computes a temperature -- `destruction.equilibrium_temperature`
decides whether a comet boils -- but it is instantaneous, stateless and star-only. Three
things follow from wanting to *edit* a temperature:

1. **It has to be state.** You cannot set a derived quantity and have it stay set. So
   temperature joins mass and radius in the World's arrays.
2. **It has to relax.** State that never moves is not physics. A body you heat must
   cool back down, on a timescale you can watch.
3. **Every body has to radiate.** Once a planet has a temperature of its own, there is
   no reason for it not to warm its moons -- and doing it that way means "proximity to
   a hotter planet" needs no code of its own. It is the same sum.

## The physics

### Equilibrium

A body absorbs sunlight over its cross-section and radiates over its whole surface.
Balancing the two for body `i` against every radiator `j`:

```
T_eq(i)^4  =  (1 - A_i)/4 * SUM_j  T_j^4 * (R_j / d_ij)^2   +   T_CMB^4
```

`A_i` is the body's albedo, `R_j` the radiator's **true** radius, `d_ij` the separation.
Fluxes add; temperatures do not -- two identical suns make a body 2^(1/4) = 1.19 times
hotter, not twice.

For a single star this reduces to the textbook form
`T = T_star * sqrt(R_star / 2d) * (1-A)^(1/4)`, which reproduces the real solar
system to within a few kelvin:

| body | model (real albedo) | measured T_eq |
|---|---|---|
| Mercury | 440 K | 440 K |
| Venus | 227 K | 232 K |
| Earth | 254 K | 254 K |
| Mars | 210 K | 210 K |
| Saturn | 81 K | 81 K |
| Neptune | 47 K | 46 K |

This is the bare radiative-equilibrium temperature. **There is no greenhouse term**, by
decision: a greenhouse factor is a tuned per-material constant, not derived physics, and
one invented number would undermine the honesty of every other number in the readout.
Earth reads -19 C, which is true, and is the number that makes the greenhouse effect
worth explaining.

### Radiators

Everything radiates; nothing is special-cased out.

- **Stars** are pinned to `units.star_temperature(mass)` -- their temperature is set by
  fusion, not by their surroundings, so they are never relaxed and never edited. A star
  does not warm in its own light.
- **Planets, moons, debris** relax toward `T_eq` and are editable.
- **Black holes** radiate at their Hawking temperature. For a stellar-mass hole that is
  6e-8 K over a horizon a few kilometres wide -- utterly negligible, which is exactly why
  no special case is needed: the general term evaluates to nothing. A *feeding* hole's
  accretion disk is genuinely luminous, and is deliberately **out of scope**: `Body.disk`
  is a 0..1 dial for how bright to draw, not an accretion rate in solar masses per year,
  and inventing one to get a luminosity would be inventing physics.
- **The cosmic microwave background** is a floor at 2.725 K, added with no albedo
  applied. Albedo is a visible-light property; in the microwave everything is black. A
  body has to reach 10,400 AU before sunlight drops below this floor.

### Relaxation

Whole-body heat capacity gives Earth a thermal time constant near 100,000 years -- real,
and completely invisible. The reason real surfaces equilibrate in days is that only a
thin skin takes part. So the model heats a **1 metre thermal skin**:

```
tau  =  rho * h * c_p / (4 * sigma * T_ref^3)      T_ref = max(T, T_eq)
```

`rho` is the body's *current* density, so squeezing a planet denser really does make it
hold its heat longer. Linearising at `max(T, T_eq)` rather than at `T` is what makes a
body dropped to 3 K near the Sun warm up quickly instead of staying frozen: the rate is
set by whichever end of the exchange is hotter.

The skin depth is the one free parameter, and it sets the whole feel of the feature:

| material | 50 K | 255 K | 700 K | 1500 K |
|---|---|---|---|---|
| rock | 1078 d | 8.1 d | 0.39 d | 0.04 d |
| iron | 1446 d | 10.9 d | 0.53 d | 0.05 d |
| ice | 759 d | 5.7 d | 0.28 d | 0.03 d |
| gas | 5429 d | 40.9 d | 2.0 d | 0.20 d |

Hot things cool fast and cold things barely drift, because radiated power goes as T^4.
That is Stefan-Boltzmann, not a tuning choice.

The step is integrated **analytically**, never by explicit Euler:

```
T_new  =  T_eq + (T - T_eq) * exp(-dt / tau)
```

This is unconditionally stable. At `dt >> tau` it collapses to `T = T_eq`, which is the
physically correct limit and is what lets the sandbox run at ten years per second without
a temperature exploding. An explicit step would need `dt < tau`, and `tau` is under an
hour for anything glowing.

### When it steps

Once per `Simulation.advance`, over the days actually simulated -- **not** per substep.
This is the same call `update_trails` already makes, and for the same reason: at ten years
per second the integrator takes thousands of substeps per frame, and an O(n^2) thermal
pass at 200 bodies would eat the 12 ms physics budget many times over. Temperature is
slow-varying and feeds back into nothing the integrator reads, so a frame is the right
granularity.

The consequence is that ablation inside the substep loop reads the temperature as it was
at the top of the frame. At the time scales where a frame is long, the exponential form
has already snapped the body to equilibrium, so a comet arriving at perihelion is hot on
the frame it arrives.

### Impact heating

`collisions.py` already computes `specific_impact_energy` -- the Q_R of the literature --
and its own comment already observes that "most of a real impact's energy goes to heat,
not motion". Nothing consumed that heat until now.

```
dT  =  IMPACT_HEAT_FRACTION * Q_R(J/kg) / c_p
```

with `IMPACT_HEAT_FRACTION = 0.3`, inside the 0.2-0.5 range giant-impact studies find is
retained rather than radiated or carried off by ejecta. A Moon-forming impact on Earth
delivers about 5e6 J/kg, so `dT` is roughly 1900 K: the remnant is molten, glows, and
cools over months. That is what really happened.

Applied to merge remnants, graze survivors and collision fragments. Bodies that merge
combine their existing heat by mass weight first, then take the impact term:

```
T = (m1*T1 + m2*T2) / (m1+m2) + dT
```

Fragments of a shattering and of a tidal disruption inherit their parent's temperature;
tidal disruption adds no impact heat, because no impact happened.

### What is deliberately not modelled

Radiogenic and primordial internal heat (which is why Jupiter reads 102 K here against a
measured 110 K), greenhouse warming, latent heat of vaporisation during ablation, day/night
and seasonal cycles, and accretion-disk luminosity. Each is real; each needs a parameter the
sandbox does not have, and inventing one would cost more honesty than it buys.

## Architecture

### `src/thermal.py` (new, pure numpy)

The whole model, and the only module that knows the equations.

- `equilibrium_temperatures(world) -> np.ndarray` -- the target for every body, vectorised.
- `equilibrium_temperature(world, i) -> float` -- one body, for tests and the readout.
- `relaxation_days(world, i, t_eq) -> float` -- tau, in days.
- `advance(world, days)` -- relax every body; pin stars and black holes.
- `settle(world)` -- put every body **at** its equilibrium. Called when a world is built
  or reset so a scenario starts in thermal balance rather than at absolute zero.
- `impact_heating(material, specific_energy_sim) -> float` -- dT in kelvin.
- `pinned_temperature(world, i) -> float | None` -- the fusion or Hawking temperature for
  a star or a hole, `None` for anything that relaxes. One function so nothing can disagree
  about which bodies are pinned.

Added to `tests/test_layering.py`'s `PHYSICS_MODULES`, along with `src.destruction`,
which was missing from that list.

### Changes to existing modules

| module | change |
|---|---|
| `src/units.py` | `STEFAN_BOLTZMANN`, `CMB_TEMPERATURE_K`, `THERMAL_SKIN_DEPTH_M`, `MIN_TEMPERATURE_K`, `MAX_TEMPERATURE_K`, `to_celsius`, SI density conversion |
| `src/materials.py` | `albedo` and `heat_capacity_j_kgk` on `Material` |
| `src/world.py` | `temperature` array; carried by `add`, `remove`, `clear`; `temperatures` view |
| `src/destruction.py` | ablation reads `world.temperature[i]`; `equilibrium_temperature` moves to `thermal.py` |
| `src/collisions.py` | merge, graze, shatter and tidal fragments set remnant temperature |
| `src/edits.py` | `set_temperature`, `scrub_temperature`, `temp` in the planet edit fields |
| `src/sim.py` | `thermal.advance` after the substep loop; `thermal.settle` on init and reset |
| `src/app/tools.py` | Temp row in the Edit panel; fix the `EDIT_FIELDS` bug |
| `src/render/hud.py` | `Temp   254 K  (-19 C)` inspector row |
| `src/render/palette.py` | `blackbody_rgb(kelvin)` on the Planck locus |
| `src/render/sprites.py` | hot bodies tint toward their blackbody colour |

### Materials

| material | albedo | c_p (J/kg/K) |
|---|---|---|
| iron | 0.20 | 450 |
| rock | 0.15 | 800 |
| rock & ice | 0.35 | 1400 |
| water | 0.06 | 4180 |
| ice | 0.60 | 2000 |
| gas giant | 0.50 | 10000 |
| hydrogen | 0.40 | 14300 |
| singularity | 0.0 | inf |

Albedos are the real bolometric values for those substances; heat capacities are real
specific heats. A singularity absorbs everything and cannot be heated, which the pinned
temperature already guarantees -- the values are there so no code path reads a missing
field.

## Interface

### Readout

One row in the inspector for an ordinary body, between Radius and Density:

```
Temp       254 K  (-19 C)
```

Kelvin first because that is the unit the physics is in, Celsius in brackets because that
is the unit the audience has. A star shows its surface temperature in the same row. A
black hole already has its `Own heat` row and gains nothing.

### Editing

The Edit tool's rows for an ordinary body become `Mass / Density / Temp / Material`. Temp
scrubs by ratio like mass -- it spans 2.725 K to 100,000 K, four and a half orders of
magnitude, which a linear scrub cannot cover -- clamped to that range at both ends.

A star gets `Mass / Density / Material` and no Temp row: its temperature is fusion's
business and follows its mass. A black hole is unchanged.

Editing a temperature does not pin it. Heat Earth to 3000 K and it glows, boils off mass
while it is above its vaporisation point, and cools back to 254 K over the following
months. That relaxation *is* the lesson.

Measured against `tests/test_layout.py`: the Edit panel has 2.4 spare rows of headroom at
the tightest grid and smallest window, so the fourth row fits without moving the floor.

No new hotkey -- the existing Up/Down/Left/Right row navigation covers it -- so
`hud.KEY_HELP` and the button/hotkey parity test are untouched.

### Colour

`palette.blackbody_rgb(kelvin)` returns the colour of a blackbody at that temperature,
following the Planck locus. `sprites._make_body` blends the material's tones toward it
with a weight that is zero below 800 K and full by 3000 K, so ordinary worlds look exactly
as they do now and a heated one visibly glows: dull red, then orange, then white.

## Testing

New `tests/test_thermal.py`:

- The single-star form reproduces `T_star * sqrt(R_star/2d)` for a zero-albedo body, and
  Earth's distance gives 278.3 K.
- Temperature falls as `1/sqrt(d)`.
- Two identical stars give `2^(1/4)` times one star, not twice -- fluxes add, temperatures
  do not.
- A hot neighbour raises a body's equilibrium: an Earth-sized body at 288 K contributes
  about 26 K at the Moon's distance, and about 164 K when molten at 1800 K.
- Albedo cools: a high-albedo body is `(1-A)^(1/4)` cooler at the same distance.
- Far from everything, a body sits at the CMB floor and not below it.
- A star's temperature is its mass temperature and is unmoved by a hot neighbour.
- A black hole's temperature is its Hawking temperature and never relaxes.
- Relaxation is monotone toward equilibrium, converges, and is stable for a step a
  thousand times tau.
- `settle` leaves every body exactly at equilibrium, and a settled world is a fixed point
  of `advance`.
- Impact heating scales linearly with specific energy and inversely with heat capacity.
- A merge's remnant temperature is the mass-weighted mean plus the impact term.

Changes to existing tests:

- `tests/test_destruction.py` -- the two ablation tests drive `destruction.apply` directly,
  so they must now settle or advance the temperature themselves. This is a real behaviour
  change and the right one: a comet heats up over its approach rather than being
  instantaneously at equilibrium.
- `tests/test_layering.py` -- `src.thermal` and `src.destruction` join `PHYSICS_MODULES`.
- `tests/test_hud.py` -- the new inspector row.
- `tests/test_tools_edit.py` -- four edit fields for a planet, three for a star.

## Risks

- **Layout.** A fourth Edit row was measured to fit with room to spare, and
  `tests/test_layout.py` is the enforcement if that ever stops being true.
- **Budget.** The thermal pass is O(n^2) at n <= 200 and runs once a frame; it costs
  roughly 0.3 ms against a 12 ms budget. Running it per substep would not fit, which is
  why it does not.
- **Behaviour change to ablation.** Comets now have thermal lag. Intended.
