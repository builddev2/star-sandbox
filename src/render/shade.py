"""Per-pixel shaded spheres and stars, in numpy.

A body is a lit sphere seen from above the plane of the system.  The light
comes from the nearest star, in the plane -- which, taken literally, is a
permanent half-moon: the terminator is a straight line through the centre of
every disc.  So the light is **tilted ``LIGHT_TILT`` toward the viewer**, which
lights about seventy percent of the disc and gives the terminator a curve.  It
is a cheat of exactly the kind the drawn-radius floor is: the direction stays
honest, the picture reads.

**The size ladder.**  A zoom notch is 18%, and a sprite is keyed by its pixel
radius, so a zoom from 100 px to 700 px would regenerate a body a dozen times.
Above ``LADDER_START`` the shader runs at the next rung of a 25% ladder and
the result is scaled to the requested size (``shade_radius``); a notch then
lands on the rung it already has.  ``SHADE_MAX_PX`` caps the rung: a 384 px
sphere is 590,000 pixels and takes ~180 ms once, and above that ``smoothscale``
cannot be told from the real thing.

Everything here is vectorised over the whole sprite; nothing loops over
pixels in Python.  Surfaces come from ``pygame.image.frombuffer(...).copy()``
-- never ``convert_alpha`` -- because once the window is a ``pygame.Window``
there is no display surface to convert to.
"""

import math

import numpy as np
import pygame

#: Degrees the light is lifted out of the plane toward the viewer.
LIGHT_TILT = 40.0
#: Night-side floor, as a fraction of full light.  Never black.
AMBIENT = 0.08
#: Night-side floor for a dot too small to show a night side at all: below a
#: few pixels there is no room for a terminator, so a half-lit disc just reads
#: as a partly missing dot.  Ramped down to AMBIENT between SMALL_AMBIENT_LO
#: and SMALL_AMBIENT_HI pixels so a real planet still keeps its night.
AMBIENT_SMALL = 0.45
SMALL_AMBIENT_LO = 4.0
SMALL_AMBIENT_HI = 16.0
#: Terminator softness: n.L runs from this to the next value while the light
#: fades in.
TERMINATOR_LO = -0.08
TERMINATOR_HI = 0.25
#: Limb darkening exponent on the view-facing component of the normal.
LIMB_POWER = 0.6
#: Largest radius the shader runs at; bigger sprites are shaded here and scaled.
SHADE_MAX_PX = 384
#: Above this radius a sprite is shaded at the next rung of a geometric ladder.
LADDER_START = 48
LADDER_RATIO = 1.25
#: Atmosphere rim strength on the disc, and the halo's width as a fraction of
#: the radius.
ATMOSPHERE_STRENGTH = 0.55
ATMOSPHERE_HALO = 0.09


# --- noise --------------------------------------------------------------------

def _lattice(n, seed):
    """A seeded n x n lattice of random values in 0..1."""
    return np.random.default_rng(int(seed) & 0xFFFFFFFF).random((n, n))


def value_noise(xs, ys, seed, cells):
    """Smooth value noise sampled at the coordinate grids ``xs``, ``ys``
    (0..1).  ``cells`` lattice points per unit; wraps, so a band profile
    tiles."""
    lattice = _lattice(cells, seed)
    fx = (xs * cells) % cells
    fy = (ys * cells) % cells
    x0 = fx.astype(int) % cells
    y0 = fy.astype(int) % cells
    x1 = (x0 + 1) % cells
    y1 = (y0 + 1) % cells
    tx = fx - np.floor(fx)
    ty = fy - np.floor(fy)
    tx = tx * tx * (3 - 2 * tx)
    ty = ty * ty * (3 - 2 * ty)
    a = lattice[y0, x0]
    b = lattice[y0, x1]
    c = lattice[y1, x0]
    d = lattice[y1, x1]
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty


def fbm(xs, ys, seed, octaves=4, cells=4, gain=0.5):
    """Fractal Brownian motion: octaves of value noise, each finer and fainter.
    Values in 0..1."""
    total = np.zeros_like(xs, dtype=np.float64)
    amplitude = 1.0
    norm = 0.0
    for k in range(octaves):
        total += amplitude * value_noise(xs, ys, seed + 101 * k, cells << k)
        norm += amplitude
        amplitude *= gain
    return total / norm


def _smoothstep(lo, hi, x):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _smoothstep_scalar(lo, hi, x):
    """``_smoothstep`` for a single python float, not an array."""
    t = min(1.0, max(0.0, (x - lo) / (hi - lo)))
    return t * t * (3 - 2 * t)


def _rgb(tone):
    return np.asarray(tone, dtype=np.float64)


def _lerp3(a, b, t):
    """Blend two RGB tones by a per-pixel weight ``t`` of shape (h, w)."""
    a, b = _rgb(a), _rgb(b)
    return a[None, None, :] + (b - a)[None, None, :] * t[..., None]


def _toward(colour, tone, t):
    """Move a per-pixel colour array toward one tone by weight ``t``."""
    return colour + (_rgb(tone)[None, None, :] - colour) * t[..., None]


# --- the ladder ---------------------------------------------------------------

def shade_radius(px):
    """The radius the shader actually runs at for a request of ``px``."""
    px = max(1, int(px))
    if px <= LADDER_START:
        return px
    rung = LADDER_START
    while rung < px:
        rung = int(math.ceil(rung * LADDER_RATIO))
    return min(rung, SHADE_MAX_PX)


def sphere_size(px, atmosphere):
    """Side of the square surface ``sphere`` returns: the disc plus, with an
    atmosphere, room for the halo."""
    px = max(1, int(px))
    pad = int(math.ceil(ATMOSPHERE_HALO * px)) + 1 if atmosphere else 0
    return 2 * (px + pad) + 1


# --- the sphere ---------------------------------------------------------------

def _geometry(px, pad):
    """Per-pixel distance, unit-sphere coordinates and coverage alpha for a
    disc of radius ``px`` centred in a square with ``pad`` extra pixels.

    ``u, v`` (and so ``z``) are held at the direction's last *fully covered*
    pixel from ``px - 0.5`` outward, rather than following the true (clamped)
    radius out to the disc edge and beyond: the true radius drives ``z`` to 0
    at ``r == px``, so the antialiased rim pixel's limb term would go to 0
    and every coverage pixel would render black under its partial alpha,
    with an atmosphere's Fresnel rim making that a near-opaque dark ring
    between the disc and its halo.  Taking the last full pixel's shading
    instead means the rim, and the halo beyond it, are lit like the disc
    edge they sit against.
    """
    half = px + pad
    coords = np.arange(2 * half + 1, dtype=np.float64) - half
    xs, ys = np.meshgrid(coords, coords)
    r = np.hypot(xs, ys)
    alpha = np.clip(px + 0.5 - r, 0.0, 1.0)
    inner = np.minimum(r, px - 0.5)
    safe_r = np.where(r > 0, r, 1.0)
    u = np.where(r > 0, xs / safe_r * inner / px, 0.0)
    v = np.where(r > 0, ys / safe_r * inner / px, 0.0)
    z = np.sqrt(np.clip(1.0 - u * u - v * v, 0.0, 1.0))
    return r, u, v, z, alpha


def _lighting(u, v, z, lit_angle, ambient=AMBIENT):
    """Diffuse light per pixel, 0..1, with the terminator softened and the
    night side held at ``ambient``.  ``v`` grows downward on screen while the
    world's y grows upward, hence the minus.

    ``ambient`` is a per-call parameter, not always the module constant: a
    dot a few pixels across has no room to show a night side at all, so
    ``sphere`` raises it toward ``AMBIENT_SMALL`` for a small sprite -- see
    that constant's docstring."""
    tilt = math.radians(LIGHT_TILT)
    lx = math.cos(lit_angle) * math.cos(tilt)
    ly = -math.sin(lit_angle) * math.cos(tilt)
    lz = math.sin(tilt)
    ndotl = u * lx + v * ly + z * lz
    diffuse = _smoothstep(TERMINATOR_LO, TERMINATOR_HI, ndotl) * np.clip(ndotl, 0, 1)
    limb = np.power(z, LIMB_POWER)
    return (ambient + (1.0 - ambient) * diffuse) * limb


def _albedo(u, v, z, style, tones, seed):
    """Surface colour per pixel before lighting: (h, w, 3) floats, 0..255."""
    dark, mid, light = tones
    # Texture coordinates: a longitude from u with the z term stretching the
    # texture toward the limb like a real globe, and a latitude from v.
    lon = (np.arctan2(u, np.maximum(z, 1e-6)) / math.pi + 1.0) * 0.5
    lat = (v + 1.0) * 0.5

    if style == "cratered":
        base = fbm(lon, lat, seed, octaves=4, cells=6)
        colour = _lerp3(dark, mid, _smoothstep(0.3, 0.7, base))
        rng = np.random.default_rng(seed + 7)
        for _ in range(int(rng.integers(6, 13))):
            cx, cy = rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)
            radius = rng.uniform(0.04, 0.14)
            d = np.hypot(u - cx, v - cy) / radius
            floor = _smoothstep(1.0, 0.6, d)
            rim = _smoothstep(0.75, 1.0, d) * _smoothstep(1.25, 1.0, d)
            colour = colour * (1 - 0.3 * floor)[..., None]
            colour = _toward(colour, light, 0.22 * rim)
        return colour

    if style == "banded":
        warp = (fbm(lon, lat, seed + 3, octaves=3, cells=3) - 0.5) * 0.12
        profile = fbm(np.zeros_like(lat), lat * 2.0 + warp, seed, octaves=4, cells=6)
        turb = fbm(lon * 2.0, lat * 2.0 + warp, seed + 11, octaves=3, cells=8)
        t = np.clip(profile * 0.8 + turb * 0.2, 0, 1)
        # Contrast-stretch about the midpoint: the raw fbm average hugs 0.5
        # too tightly for the bands to read at a glance.
        t = np.clip((t - 0.5) * 2.2 + 0.5, 0, 1)
        colour = _lerp3(dark, mid, _smoothstep(0.25, 0.55, t))
        upper = _lerp3(mid, light, _smoothstep(0.55, 0.85, t))
        colour = colour + (upper - colour) * _smoothstep(0.5, 0.6, t)[..., None]
        # One storm oval, saturated toward the dark tone (rust on Jupiter)
        # rather than washed toward the light one.
        rng = np.random.default_rng(seed + 5)
        sx, sy = rng.uniform(-0.6, 0.6), rng.uniform(-0.5, 0.5)
        d = np.hypot((u - sx) / 0.22, (v - sy) / 0.12)
        storm_tone = _rgb(dark) * 0.75 + _rgb(light) * 0.25
        return _toward(colour, storm_tone, _smoothstep(1.0, 0.6, d))

    if style == "cracked":
        base = fbm(lon, lat, seed, octaves=3, cells=4)
        colour = _lerp3(dark, mid, 0.6 + 0.4 * base)
        ridge = 1.0 - np.abs(fbm(lon, lat, seed + 17, octaves=4, cells=8) * 2 - 1)
        return _toward(colour, light, _smoothstep(0.86, 0.96, ridge))

    if style == "turbulent":
        cells = fbm(lon, lat, seed, octaves=3, cells=10)
        colour = _lerp3(mid, light, _smoothstep(0.45, 0.7, cells))
        return _toward(colour, dark, _smoothstep(0.45, 0.3, cells))

    if style == "terrestrial":
        ocean, land, cloud = tones
        height = fbm(lon, lat, seed, octaves=5, cells=4)
        # 0.56..0.62 (the brief's number) still measured 55% ocean, a hair
        # under the target floor; nudged up another 0.02 per the brief's own
        # fallback ("if Earth still reads mostly green, raise the land
        # threshold by 0.02").
        landmask = _smoothstep(0.58, 0.64, height)
        mountain = _smoothstep(0.70, 0.80, height)
        colour = _lerp3(ocean, land, landmask)
        colour = _toward(colour, _rgb(land) * 0.55, mountain * landmask)
        shallow = _smoothstep(0.44, 0.5, height) * (1 - landmask)
        colour = _toward(colour, _rgb(ocean) * 1.4, 0.5 * shallow)
        caps = _smoothstep(0.8, 0.9, np.abs(v)) * (0.6 + 0.4 * height)
        colour = _toward(colour, cloud, caps)
        clouds = _smoothstep(0.55, 0.75, fbm(lon * 1.5, lat, seed + 23, octaves=4, cells=6))
        return _toward(colour, cloud, 0.55 * clouds)

    # Unknown style: a plain mid-toned ball.
    return np.broadcast_to(_rgb(mid), u.shape + (3,)).copy()


def to_surface(rgb, alpha):
    """(h, w, 3) colour floats and (h, w) alpha floats 0..1 to a SRCALPHA
    surface that owns its pixels."""
    h, w = alpha.shape
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return pygame.image.frombuffer(rgba.tobytes(), (w, h), "RGBA").copy()


def sphere(px, lit_angle, tones, style, seed, atmosphere=None):
    """A shaded planet or moon of radius ``px``, lit from ``lit_angle``
    (a world angle: 0 is +x, pi/2 is up on screen).  Shaded at
    ``shade_radius(px)`` and scaled to ``sphere_size(px, atmosphere)``."""
    px = max(1, int(px))
    shade_px = shade_radius(px)
    pad = int(math.ceil(ATMOSPHERE_HALO * shade_px)) + 1 if atmosphere else 0
    r, u, v, z, alpha = _geometry(shade_px, pad)
    # A 6-px dot has no room for a night side -- it must read as a coloured
    # dot, not a half-lit smudge -- so small sprites get a much higher floor,
    # ramped back down to AMBIENT by SMALL_AMBIENT_HI px where a real night
    # side has room to read.
    ambient = AMBIENT + (AMBIENT_SMALL - AMBIENT) * (
        1.0 - _smoothstep_scalar(SMALL_AMBIENT_LO, SMALL_AMBIENT_HI, shade_px))
    light = _lighting(u, v, z, lit_angle, ambient)
    rgb = _albedo(u, v, z, style, tones, seed) * light[..., None]

    if atmosphere is not None:
        atm = _rgb(atmosphere)
        # A Fresnel rim on the disc, brighter on the day side.
        fresnel = np.power(1.0 - z, 3.0)
        day = 0.15 + 0.85 * np.clip(light, 0, 1)
        rgb = rgb + atm[None, None, :] * (ATMOSPHERE_STRENGTH * fresnel * day * alpha)[..., None]
        # A soft halo just outside the rim, carried in alpha, brighter toward
        # the light.
        halo_w = max(1.0, ATMOSPHERE_HALO * shade_px)
        outside = np.clip((r - shade_px + 0.5) / halo_w, 0.0, 1.0)
        halo = (1.0 - outside) ** 2 * (r > shade_px - 0.5) * 0.9
        lx, ly = math.cos(lit_angle), -math.sin(lit_angle)
        side = 0.35 + 0.65 * np.clip((u * lx + v * ly) * 0.5 + 0.5, 0, 1)
        halo = halo * side
        # Blend toward the atmosphere colour by the halo's weight wherever
        # the disc's own coverage is below 1, so a pixel with no disc
        # coverage at all (alpha == 0) comes out exactly atm, not the
        # leftover (and possibly near-black) rgb computed above.
        denom = np.maximum(np.maximum(alpha, halo), 1e-6)
        w = (halo / denom) * (alpha < 1)
        rgb = rgb + (atm[None, None, :] - rgb) * w[..., None]
        alpha = np.maximum(alpha, halo)

    surface = to_surface(rgb, alpha)
    size = sphere_size(px, atmosphere)
    if surface.get_width() != size:
        surface = pygame.transform.smoothscale(surface, (size, size))
    return surface


# --- stars --------------------------------------------------------------------

#: Glow radius as a multiple of a star's disc radius.
GLOW_SCALE = 2.2
#: Limb darkening: the edge of the disc is this fraction of the centre.
STAR_LIMB = 0.4
GRANULE_CONTRAST = 0.06
#: Peak corona brightness, as a fraction of the star's colour; below the limb.
CORONA_PEAK = STAR_LIMB * 0.9


def star_size(px):
    """Side of the square surface ``star`` returns: disc plus corona."""
    return 2 * (int(max(1, int(px)) * GLOW_SCALE) + 1) + 1


def star(px, colour, seed=0):
    """A star of radius ``px`` in ``colour``, corona included.

    Blit with ``BLEND_ADD``, which **ignores alpha** and adds raw channel
    values.  So the falloff lives in the colour itself and the surface is
    opaque everywhere; encoding the falloff as alpha would produce a solid
    disc that swallows everything within a couple of AU.
    """
    px = max(1, int(px))
    shade_px = shade_radius(px)
    half = int(shade_px * GLOW_SCALE) + 1
    coords = np.arange(2 * half + 1, dtype=np.float64) - half
    xs, ys = np.meshgrid(coords, coords)
    r = np.hypot(xs, ys)
    col = _rgb(colour)

    inside = np.clip(shade_px + 0.5 - r, 0.0, 1.0)
    z = np.sqrt(np.clip(1.0 - (r / shade_px) ** 2, 0.0, 1.0))
    limb = STAR_LIMB + (1.0 - STAR_LIMB) * z
    u = np.clip(xs / shade_px, -1, 1)
    v = np.clip(ys / shade_px, -1, 1)
    lon = (np.arctan2(u, np.maximum(z, 1e-6)) / math.pi + 1.0) * 0.5
    lat = (v + 1.0) * 0.5
    granules = 1.0 - GRANULE_CONTRAST * (1.0 - fbm(lon, lat, seed, octaves=3, cells=12))
    disc = limb * granules * inside
    disc_rgb = col[None, None, :] * disc[..., None]
    # A white-hot core, so the centre saturates whatever the colour.
    core = np.clip((0.72 * shade_px - r) / (0.2 * shade_px), 0, 1) * inside
    disc_rgb = disc_rgb + (255.0 - disc_rgb) * (core * 0.8)[..., None]

    # The corona: an inverse-square falloff outside the disc, tapered to zero
    # over the outer half-radius so the glow fades rather than stopping at a
    # hard-edged circle, with a few soft streamers at seeded angles.
    ratio = shade_px / np.maximum(r, shade_px)
    taper = np.clip((GLOW_SCALE * shade_px - r) / (0.5 * shade_px), 0.0, 1.0)
    corona = ratio ** 2 * taper
    rng = np.random.default_rng(seed + 31)
    angle = np.arctan2(ys, xs)
    streamers = np.zeros_like(r)
    for _ in range(int(rng.integers(3, 6))):
        a0 = rng.uniform(-math.pi, math.pi)
        width = rng.uniform(0.25, 0.5)
        d = np.abs((angle - a0 + math.pi) % (2 * math.pi) - math.pi)
        streamers += np.exp(-(d / width) ** 2)
    corona = corona * (1.0 + 0.3 * streamers) * (1.0 - inside)
    # Scaled so the brightest corona pixel sits just under the limb: the
    # photosphere's edge stays the brightest thing in the picture, and the
    # falloff keeps its shape instead of flattening into a shelf.
    corona = corona * (CORONA_PEAK / max(float(corona.max()), 1e-9))

    rgb = disc_rgb + col[None, None, :] * corona[..., None]
    surface = to_surface(rgb, np.ones_like(r))
    size = star_size(px)
    if surface.get_width() != size:
        surface = pygame.transform.smoothscale(surface, (size, size))
    return surface
