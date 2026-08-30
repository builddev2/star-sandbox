"""How hot everything is.

One rule, and no special cases: **every body radiates as a blackbody at its own
temperature and its own true radius, and every body is warmed by the sum of what
falls on it.**  The Sun heating Mercury and a molten Earth heating the Moon are
the same term in the same sum, which is why "a hot planet warms its neighbours"
needed no code of its own.

Balancing absorption over a cross-section against emission over a whole surface:

    T_eq(i)^4  =  (1 - A_i)/4 * SUM_j T_j^4 (R_j / d_ij)^2  +  T_CMB^4

For a single star that reduces to the textbook `T = T_star sqrt(R_star/2d)
(1-A)^(1/4)`, which reproduces the real solar system to within a few kelvin --
Mercury 440 K, Earth 254 K, Neptune 47 K.  There is deliberately **no
greenhouse term**: it would be a tuned per-material constant rather than derived
physics, and one invented number would undermine every honest number beside it.
So Earth reads -19 C here, which is true, and is exactly the number that makes
the greenhouse effect worth explaining.

The microwave background is added as a floor with no albedo applied -- albedo is
a visible-light property, and in the microwave everything is black.  Nothing in
the sandbox gets colder than 2.725 K, and sunlight does not fall below it until
10,400 AU.

**Stars and black holes are pinned, never relaxed.**  A star's temperature is set
by fusion and follows its mass; a hole's is its Hawking temperature.  Neither
warms in its own light, and neither is editable.  A stellar-mass hole radiates at
six hundredths of a microkelvin across a horizon a few kilometres wide, so it
falls out of the general sum as nothing -- which is why it needs no exception.  A
*feeding* hole's accretion disk is genuinely luminous and is deliberately left
out: ``Body.disk`` is a 0..1 dial for how brightly to draw one, not an accretion
rate, and inventing a rate to get a luminosity would be inventing physics.

Imports numpy and nothing from pygame -- ``tests/test_layering.py`` asserts it.
"""

import math

import numpy as np

from . import units
from .body import KIND_BLACK_HOLE, KIND_STAR


def pinned_temperature(world, i):
    """The temperature this body's own physics fixes, or None if it relaxes.

    One function, so nothing can disagree about which bodies are held and which
    are free: ``advance`` writes these back every step, ``settle`` seeds from
    them, and ``edits`` refuses to scrub them.
    """
    kind = world.bodies[i].kind
    if kind == KIND_STAR:
        return units.star_temperature(float(world.mass[i]))
    if kind == KIND_BLACK_HOLE:
        return units.hawking_temperature_k(float(world.mass[i]))
    return None


def radiator_temperatures(world):
    """The temperature every body *shines* at.

    The stored array for anything that relaxes, the pinned value for anything
    that does not -- so an equilibrium is correct even on a world nobody has
    settled yet, which is what a freshly built scenario is.
    """
    n = world.n
    temps = world.temperature[:n].astype(np.float64, copy=True)
    for i in range(n):
        pinned = pinned_temperature(world, i)
        if pinned is not None:
            temps[i] = pinned
    return temps


def equilibrium_temperatures(world):
    """The temperature every body is heading toward, kelvin.

    O(n^2) and vectorised.  At the 200-body ceiling this is a 40,000-element
    matrix and costs a fraction of a millisecond -- which is affordable once a
    frame and nowhere near affordable once a substep.  See ``sim.advance``.
    """
    n = world.n
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    pos = world.pos[:n]
    radius = world.radius[:n]
    temps = radiator_temperatures(world)
    albedo = np.array([world.bodies[i].material.albedo for i in range(n)],
                      dtype=np.float64)

    delta = pos[:, None, :] - pos[None, :, :]
    dist2 = np.einsum("ijk,ijk->ij", delta, delta)
    # A body does not warm itself, and two bodies at the same point would
    # divide by zero rather than reach an infinite temperature.
    np.fill_diagonal(dist2, np.inf)
    dist2 = np.where(dist2 > 0.0, dist2, np.inf)

    flux = (temps ** 4 * radius ** 2)[None, :] / dist2
    absorbed = (1.0 - albedo) * 0.25 * flux.sum(axis=1)
    return np.clip(absorbed + units.CMB_TEMPERATURE_K ** 4, 0.0, None) ** 0.25


def equilibrium_temperature(world, i):
    """Where body ``i`` is heading, kelvin.  The whole-world sum, indexed --
    never a second implementation that could drift from it."""
    return float(equilibrium_temperatures(world)[i])


#: How many times ``settle`` sweeps the world.  One pass is exact for bodies
#: warmed only by stars, since a star's temperature is pinned rather than
#: solved for; the extra passes are what let planets warming *each other*
#: converge.  Those terms are tiny, so this converges immediately -- the loop
#: is here so the answer does not depend on the order bodies happen to sit in.
SETTLE_PASSES = 4

#: Largest exponent handed to exp().  Anything past this is zero to a double
#: anyway; clamping keeps a tau of microseconds from underflowing noisily.
_MAX_DECAY_EXPONENT = 700.0


def relaxation_days(world, i, t_eq):
    """The thermal time constant of body ``i``, in days.

        tau = rho * h * c_p / (4 sigma T_ref^3)

    ``rho`` is the body's *current* density, so squeezing a planet denser in
    the Edit tool really does make it hold its heat longer, and ``h`` is
    ``units.THERMAL_SKIN_DEPTH_M`` -- see that constant for why a skin rather
    than the whole body.

    ``T_ref`` is the *hotter* of the body and its target, not the body alone.
    The exchange runs at the rate the hot end sets: linearising at the body's
    own temperature would leave something dropped to 3 K beside the Sun frozen
    for eighteen thousand years, which is an artefact of the linearisation and
    not a fact about the world.
    """
    material = world.bodies[i].material
    c_p = material.heat_capacity_j_kgk
    mass = float(world.mass[i])
    radius = float(world.radius[i])
    if not math.isfinite(c_p) or c_p <= 0.0 or mass <= 0.0 or radius <= 0.0:
        return math.inf

    rho = units.to_kgm3(units.density_from_mass_radius(mass, radius))
    t_ref = max(float(world.temperature[i]), float(t_eq), units.CMB_TEMPERATURE_K)
    denominator = 4.0 * units.STEFAN_BOLTZMANN * t_ref ** 3
    if denominator <= 0.0:
        return math.inf
    seconds = rho * units.THERMAL_SKIN_DEPTH_M * c_p / denominator
    return seconds / units.SEC_PER_DAY


def _clamp(kelvin):
    return float(min(max(kelvin, units.MIN_TEMPERATURE_K), units.MAX_TEMPERATURE_K))


def advance(world, days):
    """Relax every body toward its equilibrium over ``days``.

    Integrated **analytically**, never by an explicit step::

        T <- T_eq + (T - T_eq) exp(-dt / tau)

    which is unconditionally stable and collapses to ``T = T_eq`` when the step
    dwarfs the time constant -- the physically right limit, and the reason the
    sandbox can run at ten years per second without a temperature diverging.
    An explicit step would need ``dt < tau``, and tau is under an hour for
    anything glowing.

    Stars and black holes are written back to their pinned values rather than
    relaxed: neither warms in its own light.
    """
    n = world.n
    if n == 0 or days <= 0.0:
        return
    target = equilibrium_temperatures(world)
    for i in range(n):
        pinned = pinned_temperature(world, i)
        if pinned is not None:
            world.temperature[i] = pinned
            continue
        t_eq = float(target[i])
        tau = relaxation_days(world, i, t_eq)
        if not math.isfinite(tau) or tau <= 0.0:
            world.temperature[i] = _clamp(t_eq)
            continue
        decay = math.exp(-min(days / tau, _MAX_DECAY_EXPONENT))
        current = float(world.temperature[i])
        world.temperature[i] = _clamp(t_eq + (current - t_eq) * decay)


def settle(world):
    """Put every body *at* its equilibrium, with no transient.

    Called when a world is built or reset, so a scenario opens in thermal
    balance.  Without it every planet would start at the microwave background
    and spend its first simulated months warming up to where it has in fact
    been sitting for four billion years.
    """
    for _ in range(SETTLE_PASSES):
        target = equilibrium_temperatures(world)
        for i in range(world.n):
            pinned = pinned_temperature(world, i)
            world.temperature[i] = (pinned if pinned is not None
                                    else _clamp(float(target[i])))
