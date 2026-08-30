"""What things are made of.

Composition is a small closed set rather than a free mixture: a child picks "rock" or
"ice", not a five-component fraction vector.  Each material carries a real uncompressed
density, which is what makes the mass/density/radius relationship in the inspector
physically honest -- squeeze a body denser and it really does shrink.

Colours live here rather than in the palette module because they are a property of the
substance, not of the theme.
"""

from dataclasses import dataclass

from . import units


@dataclass(frozen=True)
class Material:
    key: str
    name: str
    density_gcm3: float
    #: dark, mid, light -- the three tones a body is shaded with.
    tones: tuple
    #: How the surface is textured when a body is drawn large enough to show it.
    style: str
    #: Plain-English line for the inspector.
    blurb: str
    #: Scales the strength branch of the disruption threshold (units.py).
    #: Iron is tough, ice is brittle, and a gas giant has no meaningful
    #: strength at all -- its gravity branch dwarfs it regardless.
    toughness: float = 1.0
    #: Temperature at which the surface boils away near a star, in kelvin.
    vapour_temperature_k: float = 2000.0
    #: Fraction of incident light reflected rather than absorbed, 0..1.  Real
    #: bolometric values: fresh ice throws most of the light back, open water
    #: swallows nearly all of it, and that alone is tens of kelvin of difference
    #: between two worlds at the same distance from the same star.
    albedo: float = 0.15
    #: Specific heat capacity, J/kg/K -- what one kelvin costs.  Together with
    #: density it sets how long a body takes to warm up or cool down.
    heat_capacity_j_kgk: float = 800.0

    @property
    def density(self):
        """Density in simulation units (solar masses per AU^3)."""
        return units.from_gcm3(self.density_gcm3)


MATERIALS = (
    Material(
        "iron", "Iron", 7.87,
        ((0x3A, 0x3C, 0x42), (0x6B, 0x6E, 0x75), (0xC0, 0x8A, 0x5A)),
        "cratered",
        "Iron worlds are tiny for how heavy they are.",
        toughness=3.0, vapour_temperature_k=3130.0,
        albedo=0.20, heat_capacity_j_kgk=450.0,
    ),
    Material(
        "rock", "Rock", 3.30,
        ((0x4A, 0x33, 0x20), (0xB0, 0x80, 0x50), (0xE0, 0xB4, 0x84)),
        "cratered",
        "Rocky worlds are the kind you could stand on.",
        toughness=1.0, vapour_temperature_k=2700.0,
        albedo=0.15, heat_capacity_j_kgk=800.0,
    ),
    Material(
        "rock_ice", "Rock & Ice", 2.00,
        ((0x3E, 0x4E, 0x58), (0x8A, 0xA4, 0xB0), (0xCF, 0xE0, 0xE8)),
        "cracked",
        "Half rock, half frozen water.",
        toughness=0.6, vapour_temperature_k=1200.0,
        albedo=0.35, heat_capacity_j_kgk=1400.0,
    ),
    Material(
        "water", "Ocean", 1.00,
        ((0x11, 0x33, 0x5E), (0x2A, 0x6C, 0xB0), (0x7F, 0xC4, 0xE8)),
        "banded",
        "A world covered in deep water.",
        toughness=0.1, vapour_temperature_k=373.0,
        albedo=0.06, heat_capacity_j_kgk=4180.0,
    ),
    Material(
        "ice", "Ice", 0.93,
        ((0x5A, 0x86, 0x96), (0xA8, 0xD2, 0xE0), (0xEA, 0xF7, 0xFC)),
        "cracked",
        "Frozen solid, and light for its size.",
        toughness=0.4, vapour_temperature_k=180.0,
        albedo=0.60, heat_capacity_j_kgk=2000.0,
    ),
    Material(
        "gas", "Gas Giant", 1.33,
        ((0x7A, 0x55, 0x2E), (0xB9, 0x8D, 0x5A), (0xE0, 0xC0, 0x8A)),
        "banded",
        "A huge ball of gas with no surface to land on.",
        toughness=0.05, vapour_temperature_k=150.0,
        albedo=0.50, heat_capacity_j_kgk=10000.0,
    ),
    Material(
        "hydrogen", "Hydrogen & Helium", 0.70,
        ((0x8A, 0x9C, 0xC0), (0xC8, 0xD6, 0xEE), (0xEA, 0xF2, 0xFF)),
        "turbulent",
        "The same stuff stars are made of.",
        toughness=0.05, vapour_temperature_k=100.0,
        albedo=0.40, heat_capacity_j_kgk=14300.0,
    ),
)

#: A black hole is not made of anything, but every code path that draws, labels
#: or trails a body wants a Material.  This one exists to satisfy them, and its
#: tones are the accretion disk's colours so that trails and labels read hot.
#:
#: Deliberately **not** in MATERIALS: the Edit tool cycles that tuple and
#: nearest_by_density searches it, and neither may ever land here.  A black
#: hole's radius comes from its Schwarzschild radius, so density_gcm3 is never
#: read -- it is present only because the dataclass requires it.
SINGULARITY = Material(
    "singularity", "Singularity", 1.0,
    ((0x40, 0x08, 0x04), (0xE0, 0x70, 0x18), (0xFF, 0xF0, 0xD0)),
    "disk",
    "Not made of anything. Just gravity, wound too tight to let go.",
    toughness=1e9, vapour_temperature_k=float("inf"),
    albedo=0.0, heat_capacity_j_kgk=float("inf"),
)

BY_KEY = {m.key: m for m in MATERIALS}

# Reachable by key -- the Throw preset and collapse both ask for it by name --
# without being reachable by cycling or by density search.
BY_KEY[SINGULARITY.key] = SINGULARITY

#: Density is clamped to this range when the child drags the slider.  The floor is about
#: as puffy as a real planet gets; the ceiling is approaching degenerate matter.
MIN_DENSITY_GCM3 = 0.05
MAX_DENSITY_GCM3 = 25.0


def get(key):
    return BY_KEY[key]


def nearest_by_density(density):
    """The material whose density is closest to ``density`` (simulation units).

    Used when the child overrides density directly and the composition label has to
    follow along.
    """
    return min(MATERIALS, key=lambda m: abs(m.density - density))
