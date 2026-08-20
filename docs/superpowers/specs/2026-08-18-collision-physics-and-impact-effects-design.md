# Collision Physics and Impact Effects — Design

Date: 2026-08-18

## The bug that started this

An asteroid was thrown at the Sun and nothing happened. Two independent causes, both
measured headlessly rather than guessed:

**1. The shot genuinely missed.** Collision detection tests the *true* radius; the eye
sees the *drawn* radius, which `render/scale.py` floors at 6 px so that the Sun is
visible at all at system zoom. That floor makes the drawn solar disc 20x the true
radius at 64 px/AU and 129x at 10 px/AU. Firing asteroids at impact parameters of
0 / 20 / 50 / 90 % of the drawn disc, the first two were absorbed and the last two flew
clean through the visible Sun with no event at all, closest approach 4.5 and 8.1 solar
radii. The player aims at what is drawn; the physics answers about something 20 times
smaller.

**2. A real hit is invisible anyway.** A successful impact calls `merge`, the projectile
vanishes between one frame and the next, and that is the whole of it. `Simulation.events`
is computed every frame and read by nothing. There is no flash, no ejecta, no shock, no
sound, and no effects system of any kind in the codebase.

## What else is wrong with the physics

Found while reading, and worth fixing in the same pass because the effects system needs
richer outcomes to have anything to show:

- **The shatter threshold is the wrong physics.** `is_catastrophic` compares impact
  energy against gravitational binding energy alone. Solving that for velocity gives a
  threshold of about 1.1x the mutual escape velocity — 30 m/s for a 50 km rock. Real
  bodies below a few hundred metres are held together by material strength, not by their
  own gravity, and the measured catastrophic-disruption threshold for a 50 km basalt
  body is nearer 900 m/s. Small bodies shatter far too easily, and nothing in between
  merge and total destruction can ever happen.
- **Fragments always leave at exactly escape velocity**, whether the impact barely
  disrupted the target or overshot by a factor of a thousand.
- **Fragment kinetic energy is never bounded by the impact energy**, so a shattering can
  create energy from nothing.
- **Every sub-catastrophic hit is a perfect merge**, including a 95 %-offset graze at
  100 km/s. There is no hit-and-run, no cratering, no partial accretion.
- **Impact speed is measured at the end of the step**, not at contact, so gravitational
  focusing during the final approach is missed. A body dropped from rest onto another
  registers an impact speed near zero instead of the escape velocity it actually arrives
  at.
- **Nothing is destroyed by tides or by heat.** A sungrazer that passes three solar radii
  should be pulled apart and boiled away. Here it sails through untouched, which is
  exactly the near-miss the player saw.

## Approach

Three pieces, in dependency order.

### A. Contact radii: collide with what is drawn

`World` gains a `contact_radius` array, defaulting to the true radius. `find_collisions`
tests against it instead of `radius`. The app layer refreshes it once per frame from the
camera:

    contact = clip(display_radius_px(r, zoom, height) / zoom, r, r * MAX_AIM_FACTOR)

The physics layer never learns what a camera is — it is handed an array and told these
are the radii that count for contact. The render layer never learns about collisions.

`MAX_AIM_FACTOR = 8`. The cap is load-bearing: uncapped, the Sun's drawn disc is 0.6 AU
across at 10 px/AU and would swallow Mercury the moment the player zoomed out. At 8x the
Sun's contact radius is 0.037 AU, a factor of eight inside Mercury's perihelion, and
Earth's is a factor of seven inside the Moon's orbit. A test asserts this for every body
in the default scenario at every zoom the camera allows.

Everything the *physics* decides — outcome, fragment count, energies — keeps using the
true radius. Only the question "did these two touch" uses the contact radius, because
that is the only question the player asks with the mouse.

### B. Realistic collision outcomes

**Impact speed at contact.** Gravitational focusing during the final approach is exact
for two bodies and costs one square root:

    v_impact^2 = v_rel^2 + v_esc^2,   v_esc^2 = 2 G (m1 + m2) / (R1 + R2)

**Disruption threshold.** Replace binding energy with the Benz & Asphaug (1999)
catastrophic-disruption specific energy, which has a strength branch that dominates
below about 200 m and a gravity branch that dominates above it:

    Q*_D(R, rho) = 608 * R^-0.38 + 0.01574 * rho_gcm3 * R^1.36   [J/kg, R in metres]

Both constants are the published basalt fit converted from erg/g and centimetres. The
strength branch is scaled by a per-material `toughness` (iron 3.0, rock 1.0, ice 0.4,
gas and hydrogen 0.05) — a gas giant has no material strength worth speaking of, and its
gravity branch dwarfs it anyway. The gravity branch is not scaled; it is physics, not
chemistry.

**Outcome regimes**, chosen from the specific impact energy `Q_R` (impact kinetic energy
in the centre-of-mass frame per unit total mass), the threshold `Q*_D`, and the impact
angle:

| Condition | Outcome |
|---|---|
| `b > b_crit` and projectile not negligible | **hit-and-run**: a graze. Both survive, deflected, with restitution and a little mass transfer |
| `Q_R < Q*_D` | **accretion**: merge, with a crater's worth of ejecta |
| `Q_R >= Q*_D` | **disruption**: break up, largest remnant from the Leinhardt & Stewart law |

`b = sin(theta)` is the normalised impact parameter at contact, recovered from the swept
geometry; `b_crit = R_target / (R_target + R_projectile)`. This formula does the right
thing for free at both extremes: two equal bodies graze easily, while for an asteroid
onto the Sun `b_crit` is 0.99999 and a hit-and-run is correctly impossible.

**Largest remnant** follows Leinhardt & Stewart (2012), replacing the invented
`0.50 - 0.08*log2(excess)`:

    M_lr / M_tot = 0.5 * (2 - Q_R / Q*_D)                  for Q_R/Q*_D <= 1.8
    M_lr / M_tot = 0.1 * (Q_R / (1.8 Q*_D)) ** -1.5        beyond (supercatastrophic)

**Fragment speeds are bounded by energy.** The kinetic energy available to disperse
fragments is the impact energy above what disruption itself consumed, times an
efficiency (most of an impact goes to heat and comminution, not to motion):

    KE_disp = f_ke * (Q_R - Q*_D) * M_tot,   f_ke = 0.2

Fragment speeds are drawn from a distribution and then scaled so their centre-of-mass
kinetic energy equals `KE_disp` exactly, with a floor at mutual escape velocity so the
pieces do not simply fall back together. A marginal disruption now drifts apart and an
overkill impact sprays.

**Cratering ejecta.** A sub-catastrophic impact excavates a mass that scales with
`Q_R / Q*_D`. Below a threshold, or when there is no room in the 200-body budget, that
ejecta is *visual only* — the effects layer draws the spray and the physics merges
cleanly. Above it, real fragment bodies are spawned. This is the split that keeps the
O(n^2) kernel affordable: small impacts are a picture, big ones are physics.

### C. Destruction without contact

**Tidal disruption.** A body is pulled apart when the tidal stress across it exceeds
what holds it together — its own gravity plus its material strength:

    2 G M_primary R_body / d^3  >  g_self + strength_accel

Expressed this way rather than as a bare Roche radius, a rubble-pile comet disrupts near
a star while a small monolithic rock survives, which is what actually happens.

**Thermal destruction.** Equilibrium temperature near a star, `T = T_star * sqrt(R_star
/ 2d)`, drives sublimation once it passes the material's vaporisation point. Mass is
ablated over time and the body is destroyed when it is gone. This is what makes a
sungrazer a sungrazer, and it gives the near-miss that started this bug report the
dramatic end it deserves.

Both run once per substep over stars only, so they cost `O(n * n_stars)`.

### D. The effects layer

New module `src/render/effects.py`. The physics emits plain data records; the render
layer owns every pixel.

`Simulation` gains `impacts`, a list of `ImpactEvent` records drained each frame:
position, velocity, energy, outcome, the radii involved, material tones, and whether the
target is a star. The record is a dataclass of floats and tuples — no pygame, no
surfaces — so `tests/test_layering.py` stays satisfied.

`EffectField.spawn(event, camera)` turns one record into particles; `update(dt)` and
`draw(surface, camera)` run from the frame loop. Five elements, layered:

1. **Flash** — an additive disc that expands fast and fades, colour ramping white →
   yellow → orange → red as it cools. Peak radius and lifetime come from the logarithm
   of the impact energy, so a pebble and a planet both read correctly.
2. **Shock ring** — one or two expanding thin rings, offset in time.
3. **Ejecta sparks** — ballistic particles, born white-hot and cooling to the material's
   own tones, thrown into the hemisphere the impact came from.
4. **Debris plume** — slower, dimmer, larger particles for energetic impacts only.
5. **Stellar flare** — for impacts on a star, an arc of prominence plus a brief
   brightening of the star itself.

Three rules make it behave:

- **Wall-clock driven.** Effects advance on real seconds, never simulated days, so an
  impact looks the same at 1 day/s and at 10 years/s.
- **Anchored in world space, sized in screen space.** Each effect keeps a world position
  and drifts with the impact's velocity, so panning and following work; its size is
  computed against the drawn radii of the bodies involved with a pixel minimum, so it is
  visible at any zoom.
- **Budgeted.** A hard cap on live particles, oldest evicted first. A hundred-body
  pile-up must not be able to drop the frame rate.

## Testing

Physics, all headless and pure numpy:

- Regression for the reported bug: an asteroid aimed at the Sun's drawn disc is
  destroyed rather than passing through.
- `Q*_D` reproduces the published basalt curve at 1 m, 100 m and 100 km, and the
  strength/gravity crossover lands near 200 m.
- Outcome regimes: equal bodies grazing hit-and-run; a slow touch merges; a fast
  head-on shatters; an asteroid onto the Sun accretes and never grazes.
- Energy: fragment kinetic energy never exceeds the impact energy.
- Momentum and mass conservation across every outcome (already covered for merge and
  shatter; extend to hit-and-run and cratering).
- Contact radius never reaches a neighbouring body's orbit, at any zoom.
- Sungrazer: a body passing within a few solar radii is destroyed by heat.
- Tidal: a rubble pile disrupts near the Sun, a small rock does not.

Effects, headless with the dummy SDL driver:

- An impact record produces particles; they expire; the cap holds under flooding.
- The effect field advances on wall-clock time and is unaffected by the time scale.
- A smoke test drives `App.step_once` through a real collision without raising.

## Out of scope

Sound (the project has no audio layer), collisional heating changing a body's material,
and any per-frame cost that scales worse than the existing O(n^2) kernel.
