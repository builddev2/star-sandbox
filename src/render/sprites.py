"""Procedurally drawn bodies.

Nothing here is authored art.  The child creates bodies at arbitrary radii out of
arbitrary materials, so there is no sprite sheet that could cover the space --
every disc is generated on demand and cached.  The pixels come from
``render/shade.py``; this module decides *what* to shade (tones, style,
atmosphere -- from the material, or from the body's look) and owns the cache.

The lighting direction is bucketed to 16 compass points.  A continuous angle
would miss the cache on every frame the camera moves, which would mean
regenerating every sprite, every frame.  Temperature, spin, disk and heading
are bucketed for the same reason.

Big sprites are shaded once per rung of ``shade``'s size ladder and scaled to
each size asked for (``SpriteCache._scaled``); the scaled sprites live in the
LRU and the shaded bases in a small side table.
"""

import math
from collections import OrderedDict

import numpy as np
import pygame

from .. import units
from . import palette, shade

#: Lighting directions the cache distinguishes.
LIT_BUCKETS = 16

#: Below this a body's own thermal emission is far too faint to see against the
#: light it reflects; by GLOW_FULL_K there is nothing else left to see.  Real
#: numbers: iron starts to visibly glow around 800 K and is white by 3000 K.
GLOW_MIN_K = 800.0
GLOW_FULL_K = 3000.0
#: Temperature is quantised to this many kelvin before it reaches the cache
#: key.  A body cooling continuously would otherwise miss the cache on every
#: frame, which is precisely what the cache exists to prevent.
GLOW_TEMP_STEP = 100.0

# --- black holes --------------------------------------------------------------
#
# Everything below is a multiple of the drawn *shadow* radius -- the dark disc
# an observer actually sees, which light bending magnifies to sqrt(27)/2 horizon
# radii.  The horizon itself is never drawn: it is 2.6x smaller and, at any zoom
# a child uses, well under a pixel.

#: How far the accretion disk reaches, in shadow radii.
BH_DISK_OUTER = 2.6
#: Vertical squash of the disk: near-edge-on, which is the view that makes a
#: black hole legible instead of a plain ring.
BH_DISK_SQUASH = 0.22
#: The disk's far side, lensed up over the top of the shadow and down under it.
#: Much rounder than the disk itself, which is what lifts it clear.
BH_LENS_ARC = 1.6
BH_LENS_SQUASH = 0.62
#: Jet length at full spin, in shadow radii.
BH_JET_LENGTH = 4.0
#: Brightness ratio between the approaching and receding halves of the disk --
#: relativistic beaming.  About right for gas orbiting at a few tenths of c.
BH_DOPPLER = 2.5
#: Concentric ellipses the disk is built from.
BH_DISK_RINGS = 7
#: Below this many pixels there is no room for a disk; draw shadow and rim only.
BH_DETAIL_PX = 5

#: Where the disk's inner edge is drawn, in shadow radii, at zero spin and at
#: the Thorne limit.
#:
#: The true ISCO/shadow ratio runs 1.155 down to 0.238 over that range -- but a
#: black hole draws at 6 px, so the whole span is under one pixel, and past
#: a* = 0.87 the true inner edge passes *inside* the shadow and is occluded
#: outright.  Widened to a legible range, exactly the trade ``scale._floor``
#: already makes for radii: the ordering and the direction stay honest, and the
#: dark band between the photon ring and the disk really does close as spin
#: rises.
BH_INNER_MAX = 2.00
BH_INNER_MIN = 1.05

#: Disk colours, inner to outer: white, blue-white, orange, deep red.
_DISK_RAMP = (
    (0xFF, 0xFF, 0xF4),
    (0xCF, 0xE4, 0xFF),
    (0xFF, 0xD2, 0x8A),
    (0xE8, 0x82, 0x28),
    (0x9A, 0x30, 0x10),
)
#: The photon ring: the sharpest edge in the picture.
_PHOTON_RING = (255, 250, 235)
#: Jets are synchrotron light -- blue-white, and dimmer than the disk.
_JET_COLOUR = (170, 205, 255)


def _inner_disk_px(px, spin):
    """Drawn inner edge of the accretion disk, in pixels.  See BH_INNER_MAX."""
    t = min(abs(float(spin)), units.MAX_SPIN) / units.MAX_SPIN
    return px * (BH_INNER_MAX + (BH_INNER_MIN - BH_INNER_MAX) * t)


def _ramp_colour(t):
    """A point along the disk's inner-to-outer colour ramp."""
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    span = len(_DISK_RAMP) - 1
    k = min(int(t * span), span - 1)
    local = t * span - k
    a, b = _DISK_RAMP[k], _DISK_RAMP[k + 1]
    return tuple(int(a[i] + (b[i] - a[i]) * local) for i in range(3))


def _draw_ellipse_ring(surface, colour, alpha, centre, rx, ry, width):
    """One concentric ellipse of the disk, centred on ``centre``."""
    rx, ry = max(1, int(rx)), max(1, int(ry))
    rect = pygame.Rect(centre[0] - rx, centre[1] - ry, rx * 2, ry * 2)
    pygame.draw.ellipse(surface, tuple(colour) + (alpha,), rect,
                        max(1, int(width)))


def _make_black_hole(px, spin, disk, seed):
    """A black hole, painted back to front.

    Blitted **normally, never with BLEND_ADD**.  A star's sprite is additive
    because a star only ever lightens what is behind it; a black hole's whole
    point is an opaque hole in the sky, and BLEND_ADD cannot draw black.  The
    outer falloff is therefore carried in alpha rather than baked into colour,
    which is the opposite of the arrangement the star path warns against.
    """
    px = max(1, int(px))
    spin = min(max(float(spin), 0.0), units.MAX_SPIN)
    disk = min(max(float(disk), 0.0), 1.0)

    detailed = px >= BH_DETAIL_PX
    outer = int(px * BH_DISK_OUTER) + 2
    jet = int(px * BH_JET_LENGTH * spin) if (detailed and disk > 0.0) else 0
    half = max(outer, jet)
    size = half * 2 + 1
    centre = (half, half)
    surface = pygame.Surface((size, size), pygame.SRCALPHA)

    if detailed and disk > 0.0:
        # 1. Jets, first, so everything else covers their roots.
        if jet > 0:
            width = max(1, int(px * 0.28))
            alpha = int(150 * disk)
            for sign in (-1, 1):
                tip = centre[1] + sign * jet
                pygame.draw.polygon(
                    surface, _JET_COLOUR + (alpha,),
                    [(centre[0] - width, centre[1]),
                     (centre[0] + width, centre[1]),
                     (centre[0] + max(1, width // 3), tip),
                     (centre[0] - max(1, width // 3), tip)])

        inner = _inner_disk_px(px, spin)
        reach = px * BH_DISK_OUTER * (0.6 + 0.4 * disk)
        approaching_left = bool(int(seed) & 1)

        # 2. The far side of the disk, bent up over the shadow and down under
        #    it.  Rounder than the disk, which is what lifts it clear.
        arc_rx = px * BH_LENS_ARC
        _draw_ellipse_ring(surface, _ramp_colour(0.35), int(190 * disk),
                           centre, arc_rx, arc_rx * BH_LENS_SQUASH,
                           max(1, px // 5))

        # 3. The disk itself, inner to outer, with 4. Doppler beaming applied
        #    by redrawing the approaching half brighter.
        for k in range(BH_DISK_RINGS):
            frac = k / max(1, BH_DISK_RINGS - 1)
            rx = inner + (reach - inner) * frac
            if rx <= 0:
                continue
            ry = rx * BH_DISK_SQUASH
            base = _ramp_colour(frac)
            # A fast-spinning hole radiates 32% of what it eats against a
            # still one's 5.7%, so the inner rings brighten with spin too.
            gain = 1.0 + 0.6 * (spin / units.MAX_SPIN) * (1.0 - frac)
            alpha = int(min(255, 230 * disk * (1.0 - 0.45 * frac)))
            dim = tuple(min(255, int(c * gain / BH_DOPPLER)) for c in base)
            bright = tuple(min(255, int(c * gain)) for c in base)
            width = max(1, int(px * 0.22))

            _draw_ellipse_ring(surface, dim, alpha, centre, rx, ry, width)
            clip = pygame.Rect(0 if approaching_left else centre[0], 0,
                               centre[0] + 1, size)
            previous = surface.get_clip()
            surface.set_clip(clip)
            _draw_ellipse_ring(surface, bright, alpha, centre, rx, ry, width)
            surface.set_clip(previous)

    # 5. The shadow, opaque, over everything drawn so far -- which is what
    #    makes the disk's far side emerge from behind it.
    pygame.draw.circle(surface, (0, 0, 0, 255), centre, px)

    # 6. The photon ring, right at the shadow's edge.
    pygame.draw.circle(surface, _PHOTON_RING + (255,), centre, px,
                       1 if detailed else 0)
    if not detailed:
        # Too small for a one-pixel outline to survive; draw the rim as a
        # filled disc and punch the shadow back out of it.
        pygame.draw.circle(surface, _PHOTON_RING + (255,), centre, px + 1)
        pygame.draw.circle(surface, (0, 0, 0, 255), centre, px)
    return surface


# --- spacecraft ---------------------------------------------------------------
#
# Not authored art and not a glyph: the same rule render/widgets.py follows for
# its icons.  There is no font with a shuttle in it, and a missing glyph would
# render as a box.

#: Below this many pixels there is no room for a wing; draw a bright dot.
#: Matched to ``scale.CRAFT_MIN_PX`` so the silhouette appears exactly when the
#: floor stops being a bare dot -- one threshold, not two that can disagree.
CRAFT_DETAIL_PX = 5
#: The render/pick floor that keeps a distant craft's silhouette readable
#: lives in ``render.scale.craft_floor_px`` -- shared with picking, which is
#: why it is not duplicated here.

#: White upper surface, black thermal tile below, grey trim.
_CRAFT_HULL = (0xF2, 0xF5, 0xFA)
_CRAFT_TILE = (0x22, 0x26, 0x30)
_CRAFT_TRIM = (0xB8, 0xBE, 0xC8)
_CRAFT_GLASS = (0x6C, 0xC4, 0xE8)


def _make_craft(px, heading):
    """A shuttle orbiter, nose along ``heading``.

    Drawn nose-along-+x from polygons and then rotated, which is both cheaper
    and steadier than re-deriving every vertex per angle: the outline is
    authored once in units of the body length and scaled.

    Blitted **normally, never with BLEND_ADD**.  Additive blending cannot draw
    the black underside, and the tile pattern is most of what makes the
    silhouette read as a shuttle rather than as a paper dart -- the same
    reasoning the black hole path follows, not the star's.
    """
    px = max(1, int(px))
    # The surface only has to hold the un-rotated outline below --
    # pygame.transform.rotate enlarges its own returned surface to fit
    # whatever it is given, so nothing here needs to allow for rotation.
    # The farthest vertex point() ever places (the aft wing corner, at
    # along=-0.95, across=1.05) sits 1.42 body-lengths from centre; the
    # margin below rounds that up and adds a pixel for int()'s truncation.
    margin = int(math.ceil(px * 1.45)) + 1
    size = margin * 2 + 1
    centre = margin
    surface = pygame.Surface((size, size), pygame.SRCALPHA)

    if px < CRAFT_DETAIL_PX:
        pygame.draw.circle(surface, _CRAFT_HULL, (centre, centre),
                           max(1, px // 2))
        return surface

    def point(along, across):
        """Body coordinates in fractions of the body length, to pixels."""
        return (int(centre + along * px), int(centre - across * px))

    # Delta wing first, so the fuselage sits on top of it.
    pygame.draw.polygon(surface, _CRAFT_TRIM, [
        point(0.30, 0.0), point(-0.70, 1.05), point(-0.95, 1.05),
        point(-0.95, -1.05), point(-0.70, -1.05),
    ])
    # Vertical stabiliser, seen from the side the flat model gives us.
    pygame.draw.polygon(surface, _CRAFT_TRIM, [
        point(-0.62, 0.0), point(-0.95, 0.62), point(-1.00, 0.60),
        point(-0.86, 0.0),
    ])
    # Fuselage: blunt nose, straight body.
    pygame.draw.polygon(surface, _CRAFT_HULL, [
        point(1.00, 0.10), point(0.86, 0.26), point(-0.86, 0.30),
        point(-0.95, 0.0), point(-0.86, -0.30), point(0.86, -0.26),
        point(1.00, -0.10),
    ])
    # Thermal tile along the underside, and the flight deck windows.
    pygame.draw.polygon(surface, _CRAFT_TILE, [
        point(1.00, -0.10), point(0.86, -0.26), point(-0.86, -0.30),
        point(-0.95, 0.0), point(-0.90, -0.02), point(0.90, -0.02),
    ])
    if px >= CRAFT_DETAIL_PX + 4:
        pygame.draw.line(surface, _CRAFT_GLASS,
                         point(0.80, 0.16), point(0.62, 0.20), 1)

    if heading:
        surface = pygame.transform.rotate(surface, math.degrees(heading))
    return surface


def glow_fraction(kelvin):
    """How far a body has gone from its own colours to its blackbody glow, 0..1."""
    if kelvin <= GLOW_MIN_K:
        return 0.0
    return min(1.0, (kelvin - GLOW_MIN_K) / (GLOW_FULL_K - GLOW_MIN_K))


def glow_key(kelvin):
    """The quantised temperature the cache is keyed on.  Zero means no glow, so
    every ordinary body shares one key however its temperature drifts."""
    if glow_fraction(kelvin) <= 0.0:
        return 0
    return int(round(float(kelvin) / GLOW_TEMP_STEP))


def _blend_toward(tone, target, weight):
    return tuple(int(round(tone[k] + (target[k] - tone[k]) * weight))
                 for k in range(3))


#: How a named body wants to be drawn: tones, style and atmosphere colour.
#: For "terrestrial" the tones mean (ocean, land, cloud).  A body whose look
#: is not here draws from its material.
LOOKS = {
    "mercury": (((70, 70, 74), (128, 128, 132), (196, 196, 200)), "cratered", None),
    "venus": (((150, 120, 60), (214, 190, 120), (246, 232, 190)), "banded", (232, 214, 160)),
    "terrestrial": (((16, 52, 120), (86, 118, 62), (240, 244, 250)), "terrestrial", (120, 170, 255)),
    "mars": (((92, 40, 22), (176, 92, 54), (232, 190, 160)), "cratered", (220, 170, 130)),
    "moon": (((60, 60, 64), (118, 118, 122), (186, 186, 190)), "cratered", None),
    "jupiter": (((120, 70, 40), (214, 178, 132), (244, 232, 210)), "banded", (200, 180, 150)),
    "saturn": (((150, 120, 70), (214, 190, 130), (240, 226, 190)), "banded", (210, 190, 150)),
    "uranus": (((90, 170, 180), (150, 220, 230), (210, 240, 245)), "banded", (150, 220, 230)),
    "neptune": (((22, 40, 120), (56, 96, 200), (140, 180, 240)), "banded", (90, 130, 230)),
}


def appearance(material, look):
    """What to shade: ``(tones, style, atmosphere)`` for a material and a look."""
    if look in LOOKS:
        return LOOKS[look]
    return material.tones, material.style, None


def _make_body(material, seed, px, lit_angle, is_star, glow_k=0.0, look=None):
    tones, style, atmosphere = appearance(material, look)
    dark, mid, light = tones
    if glow_k > 0.0:
        # Its own light, mixed into the light it reflects.  Applied before the
        # star branch as well as after it, so a red dwarf really is red and a
        # blue giant really is blue -- the same Planck locus, the same helper.
        target = palette.blackbody_rgb(glow_k)
        weight = glow_fraction(glow_k)
        dark = _blend_toward(dark, target, weight)
        mid = _blend_toward(mid, target, weight)
        light = _blend_toward(light, target, weight)
    if is_star:
        return shade.star(px, light, seed)
    return shade.sphere(px, lit_angle, (dark, mid, light), style, seed, atmosphere)


# --- rings ----------------------------------------------------------------------

#: Inner and outer edge of the ring system, the gap, and the vertical squash,
#: all in planet radii.  Saturn's real rings run 1.24..2.27 radii with the
#: Cassini division at 1.95; the squash is the same near-edge-on view the
#: black hole's disk uses.
RING_INNER = 1.4
RING_OUTER = 2.3
RING_GAP = 1.95
RING_SQUASH = 0.30


def ring_size(px):
    """Side of the square surfaces ``ring_surfaces`` returns."""
    return 2 * (int(math.ceil(max(1, int(px)) * RING_OUTER)) + 2) + 1


def _ring_surfaces_at(rung, tones, seed, lit_angle):
    """Both ring halves shaded at exactly ``rung`` pixels -- the ladder rung,
    built once and reused for every ``px`` that maps to it, the same way
    ``_make_body`` is built once per rung for the sphere ladder."""
    rung = max(1, int(rung))
    half = int(math.ceil(rung * RING_OUTER)) + 2
    coords = np.arange(2 * half + 1, dtype=np.float64) - half
    xs, ys = np.meshgrid(coords, coords)
    # Unsquash: a point on the drawn ellipse maps to a radius in ring units.
    rr = np.hypot(xs, ys / RING_SQUASH) / rung
    band = (rr >= RING_INNER) & (rr <= RING_OUTER)
    t = np.clip((rr - RING_INNER) / (RING_OUTER - RING_INNER), 0, 1)
    density = shade.fbm(t, np.zeros_like(t), seed + 41, octaves=3, cells=8)
    alpha = (0.35 + 0.5 * density) * band
    alpha = alpha * (1.0 - shade._smoothstep(0.06, 0.02, np.abs(rr - RING_GAP)))
    # Soft edges, one ring-unit-pixel wide.
    edge = np.minimum(np.clip((rr - RING_INNER) * rung, 0, 1),
                      np.clip((RING_OUTER - rr) * rung, 0, 1))
    alpha = alpha * edge
    dark, mid, light = tones
    rgb = shade._lerp3(mid, light, density)

    # The planet's shadow: a band of the planet's width along the anti-star
    # direction, on the far side only.
    lx, ly = -math.cos(lit_angle), math.sin(lit_angle)
    along = xs * lx + ys * ly
    across = np.abs(xs * -ly + ys * lx)
    shadow = (along > 0) * shade._smoothstep(rung * 1.05, rung * 0.9, across)
    back_rgb = rgb * (1.0 - 0.75 * shadow)[..., None]

    back_alpha = alpha * (ys < 0)
    front_alpha = alpha * (ys >= 0)
    back = shade.to_surface(back_rgb, back_alpha)
    front = shade.to_surface(rgb, front_alpha)
    return back, front


def ring_surfaces(px, tones, seed, lit_angle):
    """A flat ring system as two surfaces, ``(back, front)``: the half drawn
    behind the planet and the half drawn in front of it, so the disc can sit
    between them.  The planet's shadow is cut across the back half along the
    anti-star direction.  Shaded at the ladder rung like a sphere, then
    scaled to ``px`` -- see ``_ring_surfaces_at`` for the rung-level build
    that ``SpriteCache.get_rings`` reuses across every ``px`` on one rung."""
    px = max(1, int(px))
    rung = shade.shade_radius(px)
    back, front = _ring_surfaces_at(rung, tones, seed, lit_angle)
    size = ring_size(px)
    if back.get_width() != size:
        back = pygame.transform.smoothscale(back, (size, size))
        front = pygame.transform.smoothscale(front, (size, size))
    return back, front


class SpriteCache:
    """LRU of rendered bodies, keyed by everything that changes the pixels."""

    #: Shaded bases held in the side table (see ``_scaled``).
    BASE_LIMIT = 48

    def __init__(self, limit=512):
        self.limit = limit
        self._entries = OrderedDict()
        self._bases = {}

    def __len__(self):
        return len(self._entries)

    def clear(self):
        self._entries.clear()
        self._bases.clear()

    def _hit(self, key):
        """The sprite cached under ``key``, freshened, or None."""
        surface = self._entries.get(key)
        if surface is not None:
            self._entries.move_to_end(key)
        return surface

    def _store(self, key, surface):
        """Keep a new sprite, dropping the least recently used if we are full."""
        self._entries[key] = surface
        if len(self._entries) > self.limit:
            self._entries.popitem(last=False)
        return surface

    def _scaled(self, key, px, make, size):
        """A sprite at ``px`` from a base shaded at the ladder rung.

        ``make(radius)`` shades at that radius; ``size(radius)`` is the side of
        the surface it returns.  Below ``LADDER_START`` the base *is* the
        sprite.  Bases live in a small side table so a zoom notch re-uses the
        rung it already has instead of re-shading a screen-filling planet.

        ``make`` may return one surface or a tuple of surfaces sharing one
        rung (rings' back and front halves) -- either way the same shape
        comes back, each element scaled to ``size(px)``.
        """
        if px <= shade.LADDER_START:
            # Below the ladder the base *is* the sprite, and the LRU already
            # holds it: storing it here too would fill the side table with
            # tiny sprites and evict the big bases the table exists to keep.
            return make(px)
        rung = shade.shade_radius(px)
        base = self._bases.get(key + (rung,))
        if base is None:
            base = make(rung)
            if len(self._bases) >= self.BASE_LIMIT:
                # FIFO by design: the table only ever holds big (ladder-rung)
                # bases, of which there are few, so 48 slots is ample and
                # which one is dropped first barely matters.
                self._bases.pop(next(iter(self._bases)))
            self._bases[key + (rung,)] = base
        if rung == px:
            return base
        side = size(px)
        if isinstance(base, tuple):
            return tuple(pygame.transform.smoothscale(b, (side, side)) for b in base)
        return pygame.transform.smoothscale(base, (side, side))

    def get(self, material, seed, px, lit_angle, is_star,
            temperature=units.CMB_TEMPERATURE_K, look=None):
        px = max(1, int(px))
        bucket = int(round(lit_angle / (2 * math.pi) * LIT_BUCKETS)) % LIT_BUCKETS
        heat = glow_key(temperature)
        look = look if look in LOOKS else None
        key = (material.key, seed, px, bucket, bool(is_star), heat, look)
        hit = self._hit(key)
        if hit is not None:
            return hit
        angle = bucket * 2 * math.pi / LIT_BUCKETS
        glow = heat * GLOW_TEMP_STEP
        atmosphere = appearance(material, look)[2]
        if is_star:
            size = shade.star_size
        else:
            def size(radius):
                return shade.sphere_size(radius, atmosphere)
        surface = self._scaled(
            (material.key, seed, bucket, bool(is_star), heat, look), px,
            lambda radius: _make_body(material, seed, radius, angle, is_star, glow, look),
            size)
        return self._store(key, surface)

    def get_rings(self, px, tones, seed, lit_angle):
        """Both halves of a ring system, cached under one key.

        Routed through ``_scaled`` like every other big sprite: without it a
        ring system was rebuilt from scratch (fbm over a 1777x1777 grid at
        rung 384) on every distinct ``px``, instead of sharing the one base
        its rung already has.
        """
        px = max(1, int(px))
        bucket = int(round(lit_angle / (2 * math.pi) * LIT_BUCKETS)) % LIT_BUCKETS
        key = ("rings", tuple(tones), int(seed), px, bucket)
        hit = self._hit(key)
        if hit is not None:
            return hit
        angle = bucket * 2 * math.pi / LIT_BUCKETS
        surface = self._scaled(
            ("rings", tuple(tones), int(seed), bucket), px,
            lambda r: _ring_surfaces_at(r, tones, seed, angle),
            ring_size)
        return self._store(key, surface)

    def get_black_hole(self, seed, px, spin, disk):
        """A black hole sprite, cached like any other.

        Spin and disk are bucketed for the same reason ``get`` buckets the
        lighting angle: a continuous value would miss the cache on every frame
        the child holds a stepper down, and regenerating the sprite every frame
        is exactly what the cache exists to avoid.
        """
        px = max(1, int(px))
        spin_bucket = int(round(min(abs(float(spin)), units.MAX_SPIN)
                                / units.MAX_SPIN * 8))
        disk_bucket = int(round(min(max(float(disk), 0.0), 1.0) * 8))
        key = ("bh", int(seed) & 1, px, spin_bucket, disk_bucket)
        hit = self._hit(key)
        if hit is not None:
            return hit
        surface = _make_black_hole(px, spin_bucket / 8.0 * units.MAX_SPIN,
                                   disk_bucket / 8.0, seed)
        return self._store(key, surface)

    def get_craft(self, px, heading):
        """A shuttle sprite, cached like any other.

        Heading is bucketed into the same 16 compass points the lighting angle
        uses, and for exactly the same reason: a craft turning continuously
        along its orbit would miss the cache on every single frame, which is
        precisely what the cache exists to prevent.
        """
        px = max(1, int(px))
        bucket = int(round(heading / (2 * math.pi) * LIT_BUCKETS)) % LIT_BUCKETS
        key = ("craft", px, bucket)
        hit = self._hit(key)
        if hit is not None:
            return hit
        surface = _make_craft(px, bucket * 2 * math.pi / LIT_BUCKETS)
        return self._store(key, surface)
