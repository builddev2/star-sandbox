# Black Holes — Design

Date: 2026-08-27

## Purpose

Add a black hole to the set of things the child can create, alongside planets, stars and
debris. It is the one object in the sandbox whose behaviour cannot be guessed from the
others: it is heavier than a star and smaller than a city, it eats whatever touches it,
it shreds anything that comes close without touching it at all, and it is the only body
you cannot see directly.

The premise of the project holds. Nothing here is hand-animated and nothing is invented.
Every number below comes from a real formula, and the ones that look absurd — a density
of 1.8e19 kg/m^3, an evaporation time of 1e67 years — are absurd because reality is.

## What a black hole actually is, and what that costs

### There are exactly three properties, and one of them is always zero

The no-hair theorem says a black hole is completely described by mass, angular momentum
and charge. Astrophysical black holes have no net charge — any imbalance is neutralised
by the surrounding plasma almost immediately — so there are **two** editable properties,
not three:

| Property | Symbol | Range here | Scrub notch | Why |
|---|---|---|---|---|
| Mass | `M` | 3 .. 150 M_sun | x1.06 (ratio, as now) | The observed stellar-mass range, LIGO merger remnants included |
| Spin | `a*` | 0 .. 0.998 | +/- 0.02 (linear) | Dimensionless. 0.998 is the Thorne limit: radiation from the disk torques a hole back down before it can reach 1 |
| Disk | — | 0 .. 1 | +/- 0.05 (linear) | Feeding rate, driving disk brightness and extent. 0 is a starved hole: shadow and photon ring only |

The **lower** bound of that range is `units.TOV_LIMIT_MASS = 3.0`, the
Tolman-Oppenheimer-Volkoff limit — the heaviest a neutron star can be before it collapses
— below which there is an observed gap where nature makes no black holes at all.

It is deliberately **not** enforced as a clamp. Collapse conserves mass exactly, so
collapsing the Sun produces a 1 M_sun hole that nature could not make. The alternatives
are worse: refusing the collapse removes the single most valuable thing this feature
offers, and silently tripling the Sun's mass to permit it invents matter. The limit is
reported in the inspector instead, where it teaches something rather than blocking
something.

**Accretion is not bound by `MAX_MASS`.** The 150 M_sun ceiling is a limit on what the
child can *dial*; `collisions.merge` writes `world.mass[i]` directly and so a hole that
eats a star, or merges with another hole, grows past it. That is correct — it is how
every black hole above the stellar range was actually made — and it is the only route to
one.

Everything else — radius, density, surface gravity, escape velocity, temperature,
lifetime, the inner edge of the accretion disk — is derived. That is a stronger version
of the invariant the codebase already keeps for ordinary bodies ("mass and density are
set; radius is always derived"), not a violation of it. Only the law changes:

    ordinary body:  radius = (mass / (4/3 pi density))^(1/3)
    black hole:     radius = 2GM/c^2

A third quantity, the **accretion disk**, is exposed as an editable value even though it
is not a property of the hole. It is the brightness of the matter falling in, it is the
only part of the object that emits anything, and it is what the child is actually looking
at. A starved hole and a feeding hole look completely different and are equally real.

### The measured numbers

Computed, not quoted, and checked against textbook values:

    c                       173.1446326742 AU/day
    r_s per solar mass      1.974126e-08 AU  =  2.95325 km      (textbook 2.953 km)
    shadow radius           sqrt(27)/2 r_s   =  2.598076 r_s
    photon sphere           1.5 r_s
    density, 1 M_sun        1.843e19 kg/m^3  =  80x nuclear saturation density
    Hawking temperature     6.169e-8 K x (M_sun / M)
    evaporation time        2.1e67 yr x (M / M_sun)^3

Kerr ISCO from Bardeen, Press & Teukolsky (1972), with the accretion efficiency that
follows from the binding energy at that orbit:

| `a*` | ISCO | ISCO in `r_s` | Efficiency |
|---|---|---|---|
| 0.000 | 6.000 GM/c^2 | 3.000 | 5.72 % |
| 0.500 | 4.233 GM/c^2 | 2.117 | 8.21 % |
| 0.900 | 2.321 GM/c^2 | 1.160 | 15.58 % |
| 0.998 | 1.237 GM/c^2 | 0.618 | 32.10 % |

Spin is not decoration. It moves the inner edge of the disk in by a factor of five and
takes the efficiency from 5.7 % to 32 % — which is why accreting black holes outshine
the galaxies they sit in, and why the disk visibly tightens and brightens when the child
drags the spin slider.

### Three things fall out for free

**Escape velocity at the horizon is exactly `c`.** `units.escape_velocity(M, r_s)`
computes `sqrt(2GM/r_s)`, and substituting `r_s = 2GM/c^2` gives `sqrt(c^2)`. The
existing function returns the right answer with no special case, and the inspector row
reading `Escape  299792 km/s` is a consistency check on the whole unit system.

**Tidal disruption already works.** `destruction.tidal_stress_ratio` is generic in mass
and distance; it is only gated on `KIND_STAR`. The tidal radius `r_t = R (M_bh/m)^(1/3)`
puts Earth's destruction **32,213 horizon radii** out from a 10 M_sun hole — 0.00636 AU,
about 950,000 km. Spaghettification is therefore a large, slow, visible event happening
far from a dot, which is the best possible outcome: the child sees the cause and the
effect as separate things.

**Density is not editable, and cannot be.** `materials.MAX_DENSITY_GCM3` is 25 g/cm^3;
a solar-mass hole is 1.8e16 g/cm^3, fifteen orders of magnitude past the cap. Density
becomes a read-only curiosity in the inspector. It scales as `M^-2`, which is the
counterintuitive fact worth showing: bigger black holes are *less* dense.

## Where the collapse threshold goes

Collapse is triggered by mass, which was initially a concern — a 3 M_sun star is not a
black hole, and real collapse needs a supernova. But 150 M_sun is close to where nature
stops making stars: above the pair-instability window the core collapses directly to a
black hole with no explosion at all. So the top of the mass range and the collapse
threshold are the same number, and that is physically honest rather than a compromise:

    edits.MAX_MASS                  50.0 -> 150.0
    units.COLLAPSE_MASS_THRESHOLD   150.0

Scrubbing mass upward now walks a ladder where every rung is a real number:

    rock -> brown dwarf (0.013) -> star (0.08) -> black hole (150)

You cannot make a 200 M_sun star, because nature cannot either. It collapses instead.

**Collapse is one-way.** `_refresh_kind` returns early for a black hole: reducing its
mass afterwards leaves a smaller black hole, never a star. Nothing in the simulation may
un-collapse one.

## What a black hole is not allowed to do

Three rules, each of which is a branch beside existing logic rather than a change to it:

**It is never grazed and never disrupted.** `classify()` returns `ACCRETE`
unconditionally when either body is a black hole. No hit-and-run, no fragments, no
cratering ejecta — nothing comes back out of a horizon, and the effects layer should not
imply otherwise.

**It always wins, regardless of mass.** This is the one change to shared collision code.
`find_collisions` puts the more massive body in the `i` slot, and resolution treats `i`
as the survivor. A 3 M_sun hole striking a 10 M_sun star would therefore be **absorbed by
the star**. When exactly one body of a pair is a black hole, it takes the `i` slot
whatever the masses say. When both are, mass decides as usual, and the result is a
black-hole merger.

**It raises tides but emits no heat.** `_bodies_and_stars` becomes
`_bodies_and_attractors`: black holes join the tidal loop, but the thermal-ablation loop
stays stars-only. Hawking radiation from a stellar-mass hole is 6 nanokelvin — colder
than the cosmic microwave background, which means such a hole absorbs more than it emits
and grows rather than evaporating.

**A star may be shredded by a black hole, but not by another star.** `_bodies_and_stars`
currently excludes stars from being victims at all, which is right for a pair of stars —
they would have to be almost touching — but wrong for a black hole, where a tidal
disruption event is the single most spectacular thing that happens. Stars join the victim
list, guarded so that only a black hole can raise a disruptive tide on one. Without this
the black-hole scenario below cannot do the one thing it exists to show.

*Deliberate omission:* a real accretion disk reaches millions of kelvin and would boil a
comet long before the tide reached it. Here the disk is a drawn object, not a simulated
one, so it heats nothing. Modelling disk luminosity would mean giving the render layer's
brightness value a physical meaning that feeds back into `destruction.py`, and that
crosses the layer boundary the project is built on. If it is wanted later it belongs as a
real emission temperature on the body, derived from spin and feeding rate, not as a
reading of the sprite.

## Impact speed must be clamped to `c`

`find_collisions` computes the mutual escape velocity from `sep = radius[big] +
radius[small]`, and a black hole's true radius is essentially zero. For a 10 M_sun hole
swallowing Earth, `sep` collapses to Earth's own radius and `v_esc` reaches 11.8 AU/day —
20,400 km/s, about 7 % of `c`. For a heavier hole and a smaller projectile it exceeds `c`
outright, and `impact_energy` then scales as the square of a nonsense number.

The outcome is unaffected, since `classify()` short-circuits to `ACCRETE`. But
`ImpactEvent.energy` feeds the effects layer's thresholds, so a runaway value produces a
runaway visual. `v_impact` is clamped to `units.C`. Cheap, correct, and it belongs in the
physics rather than being guarded against downstream.

## The visibility problem, and the aim problem behind it

`scale._floor` gives a body a minimum drawn radius of `2.0 x (r/R_earth)^0.35` pixels,
clamped to 6. For a 10 M_sun black hole:

    r_s / R_earth = 0.00464     ->  floor 0.31 px  ->  clamped up to FLOOR_MIN = 1 px

**One pixel.** The Sun draws at 6. The most dangerous object in the sandbox would be the
least visible thing on screen, smaller than a fragment of ejecta. Pushing the notional
size out to 200 `r_s` to stand for the disk still only reaches 2 px; the 0.35 curve is
far too flat to rescue an object seven orders of magnitude below Earth.

The second half of the problem is worse and is not about drawing at all.
`contact_radii_au` clips the collision radius to `MAX_AIM_FACTOR x true_radius`, which
for the same hole is `1.6e-6` AU — a target roughly **0.0001 pixels across** at system
zoom. A black hole would be, in practice, impossible to hit. The whole premise of the
contact-radius machinery is that "a shot aimed at what is on screen" connects; a black
hole breaks it by a factor of a million.

Both halves have one cause and take one fix.

### `scale.visual_radii(world)`

A single helper returning a per-body array equal to `world.radius` for everything except
black holes, where it returns

    units.apparent_radius(mass) = SUN_RADIUS x M^0.8

the main-sequence mass-radius relation — physically motivated as *the disk is about the
size of the star that made it*. Three readers call it, so they cannot disagree:

1. `draw.py`, for the drawn-radius floor
2. `state.pick_body_at`, so clicking works where the disc appears
3. `main.step_once`, for the contact-radius refresh, so aim works where the disc appears

`display_radii_px` and `contact_radii_au` each gain an optional `floor_radii` /
`aim_radii` argument defaulting to the true radii, so every existing caller is unchanged.
In `contact_radii_au` the argument replaces the true radius **in the upper clip only**:

    clip(drawn_au,  radii,  aim_radii * MAX_AIM_FACTOR)

The lower bound stays the true radius, so a contact radius can still never fall below the
real one — the invariant the whole contact-radius mechanism rests on.

Measured result — every hole in the 3..150 range saturates `FLOOR_MAX`:

| `M` | apparent radius | raw floor | drawn |
|---|---|---|---|
| 3 | 0.01120 AU | 14.1 px | 6 px |
| 10 | 0.02934 AU | 19.7 px | 6 px |
| 150 | 0.25608 AU | 42.0 px | 6 px |

All three draw identically at wide zoom, exactly as stars already do, and zooming in
separates them.

### What stays honest

**Physics never sees the apparent radius.** Every outcome — impact energy, tidal stress,
disruption threshold, the merged radius — uses the true `r_s`. Only the yes/no contact
test and the drawing use the visual one, which is precisely the existing split between
`radius` and `contact_radius`.

**`World.clamp_contact_radii` still protects neighbours.** A 10 M_sun hole's contact
radius reaches 0.094 AU at system zoom, which is large; the existing geometry-aware clamp
cuts it to half the distance to whatever is nearest, exactly as it already does for the
Sun.

**Real-size mode collapses it to a dot.** `display_radii_px(real_size=True)` bypasses the
floor entirely, so pressing `R` shrinks a black hole from 6 px to 1 px while the planets
stay roughly where they were. That contrast is the lesson, and it costs nothing because
the branch already exists.

## The visuals

One new function in `render/sprites.py`, `_make_black_hole(px, spin, disk, seed)`, where
`px` is the drawn **shadow** radius and every other layer is a multiple of it. Painted
back to front:

| Layer | Extent | How |
|---|---|---|
| Jets | to `4 px x spin` | Two tapering vertical beams, pale blue-white. Only when `spin > 0` and `disk > 0`, and only above ~6 px |
| Lensed far side | `~1.6 px` | An arc above and below the shadow — the back half of the disk bent into view. The signature shape |
| Accretion disk | `2.0 px` .. `1.05 px` inner (by spin), out to `2.6 px` | Concentric near-edge-on ellipses, vertical squash ~0.22, heat-ramped white -> blue -> orange -> deep red outward |
| Doppler beaming | — | Each ellipse drawn twice under a clip rect, ~2.5x brighter on the approaching side. Which side approaches is fixed per body by the low bit of `body.seed`, so a given hole always spins the same way |
| Shadow | `px` | Opaque black fill, over the disk |
| Photon ring | `px` | A 1 px near-white circle at the shadow's edge |

**The inner edge moves with spin, exaggerated the way drawn radii already are.** The true
`isco_radius / shadow_radius` ratio runs from 1.155 at `a* = 0` down to 0.238 at 0.998 —
but a black hole draws at 6 px, so that entire range spans less than one pixel, and above
`a* = 0.87` the disk's inner edge passes *inside* the shadow and is occluded entirely.
The drawn inner edge is therefore mapped linearly onto `2.00 px .. 1.05 px`, preserving
the ordering and the direction: more spin, tighter disk. This is the same trade
`scale._floor` already makes, and the true ratios belong in the constant's comment.

The gap it opens between the photon ring and the disk's inner edge is real — at low spin
there is a visible dark band, and at high spin the disk closes right up against the ring.
Pairing that with the innermost ring brightening (efficiency runs 5.7 % to 32 % over the
same range) gives two honest cues for what spin means, without a word of explanation.

**Blitted normally, not additively.** Stars use `BLEND_ADD`, which cannot draw black — it
only ever lightens. A black hole's defining feature is an opaque hole in the sky, so its
sprite is a plain alpha blit and the outer falloff is carried in alpha rather than baked
into colour. This is the opposite of the star path and the reason for it should stay in
the docstring, since the star code carries a comment warning against exactly the
arrangement used here.

The sprite cache key extends with bucketed spin and disk, in the same spirit as the
existing 16-bucket lighting angle: a continuous value would miss the cache every frame
the child drags a stepper.

**A new icon `blackhole`** in `widgets.py` — a filled disc inside a ring, drawn from
primitives, because the font may not have a glyph for anything and a missing one renders
as a box.

**A new effect outcome `swallow`** in `effects.py`: a ring collapsing *inward*, which is
the reverse of every existing effect and needs no new particle kind, only a negative
radius ramp. `_build_event` emits it when the target is a black hole. `ImpactEvent`
already carries free-form outcome strings (`tidal`, `vaporise`), so no schema change.

## The interface

**Throw** gains a sixth preset on key `6`. `LaunchPreset` gains a `kind` field, and
`preset_body` derives radius from `r_s` when the kind is a black hole rather than from
the material's density.

**Edit** shows fields per kind. A black hole shows **Mass / Spin / Disk** in place of
Mass / Density / Material, so the panel never offers a control that means nothing. It
also gains a **Collapse** button (hotkey `C`), enabled only when the selection is a star.
Turning the Sun into a black hole and watching the planets carry on orbiting a dark point
is the single most valuable thing this feature makes possible, and it should be one
click.

`C` goes into `hud.KEY_HELP`, which `tests/test_ui.py` reads to assert that every
documented key has a button.

**Scenario** selection is new machinery: `state.scenario`, `cycle_scenario()`, and a
control beside Restart, since `sim.reset()` currently hard-codes `solar_system`. The new
`scenarios.black_hole()` places a 10 M_sun hole with one body already on a plunging
orbit, so a tidal disruption fires within seconds of loading rather than requiring the
child to arrange one.

**Inspector**: for a black hole the density / gravity / ignition block is replaced by
horizon radius, shadow radius, ISCO, spin, Hawking temperature, evaporation time and
accretion efficiency. `edits.ignition_ratio` and the "Nx heavier to ignite" row must be
suppressed, or a black hole reports how close it is to becoming a star.

## Layering

Unchanged, and `tests/test_layering.py` continues to enforce it.

- `units.py` gains every new constant and formula and remains the only place `c` is
  written down. Nothing else may define it.
- `body.py`, `edits.py`, `collisions.py`, `destruction.py`, `scenarios.py` stay pure
  numpy. `spin` and `disk` live on `Body` beside `rings` and `seed`, not in World's
  arrays, because the force kernel never reads them.
- `render/scale.py` may import from `world`; the physics still never learns what a camera
  is, and the render layer still never learns what a collision is.
- Nothing caches a size across frames. `visual_radii` is recomputed per frame from n <= 200
  bodies, which is negligible beside the O(n^2) kernel.

## Testing

New `tests/test_black_hole.py`:

- `r_s` equals 2.95325 km per solar mass
- escape velocity at the horizon equals `c` to floating-point tolerance
- density scales as `M^-2`
- ISCO matches the Bardeen table above at `a* = 0, 0.5, 0.9, 0.998`, and efficiency with it
- a black hole takes the `i` slot against a heavier star, and survives
- kind is sticky: mass reduced below the threshold leaves a black hole
- a star scrubbed past 150 M_sun collapses
- tidal disruption fires from a black hole with no star present
- `contact_radius` equals the true `r_s` immediately after a swallow, never below it
- `v_impact` never exceeds `c`

New sprite tests: an opaque black centre, a bright rim at the shadow edge, spin moving the
inner disk edge inward, `disk = 0` drawing no disk.

Extended: `test_scale.py` for `visual_radii` and both optional arguments; `test_layout.py`
and `test_ui.py` for the new controls at the minimum buffer size; `test_hud.py` for the
black-hole inspector rows.

All render tests set `SDL_VIDEODRIVER=dummy` before importing pygame.

## Sources

- Bardeen, Press & Teukolsky (1972), *Rotating Black Holes*, for the Kerr ISCO
- Thorne (1974), for the `a* = 0.998` equilibrium spin limit
- [NASA GSFC black hole accretion disk visualization](https://gms.gsfc.nasa.gov/old/13326)
  for the lensed-far-side and Doppler-beaming appearance
- [Black hole shadows, photon rings and lensing rings](https://arxiv.org/abs/1906.00873)
  for the `sqrt(27)/2 r_s` shadow radius
- [Maximum mass of a black hole which can tidally disrupt a star](https://academic.oup.com/mnras/article/527/3/6233/7453654)
  for the tidal radius and its relation to the ISCO
