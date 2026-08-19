import math

from src import materials, units


def _si(q):
    """Sim specific energy -> J/kg."""
    return q * units.SPECIFIC_ENERGY_SI


def test_strength_branch_matches_published_basalt_fit():
    # Benz & Asphaug 1999 basalt, converted to SI: 608 * R**-0.38 J/kg.  Kept
    # to sizes where the gravity term is still negligible (it is already 19%
    # of the total by R=100m for this density).
    rho = units.from_gcm3(3.0)
    for r_m in (0.01, 0.1, 1.0, 10.0):
        r = r_m / units.AU_M
        got = _si(units.disruption_threshold(r, rho))
        assert math.isclose(got, 608.0 * r_m ** -0.38, rel_tol=0.02)


def test_gravity_branch_dominates_for_a_hundred_kilometre_body():
    rho = units.from_gcm3(3.0)
    r = 1.0e5 / units.AU_M
    got = _si(units.disruption_threshold(r, rho))
    # ~3e5 J/kg is the measured value for a 100 km basalt target.
    assert 1.0e5 < got < 1.0e6


def test_the_curve_has_a_minimum_between_a_boulder_and_a_mountain():
    rho = units.from_gcm3(3.0)
    lo = units.disruption_threshold(1.0 / units.AU_M, rho)
    hi = units.disruption_threshold(1.0e5 / units.AU_M, rho)
    mid = units.disruption_threshold(110.0 / units.AU_M, rho)
    assert mid < lo and mid < hi          # the curve has its minimum in between


def test_toughness_scales_only_the_strength_branch():
    rho = units.from_gcm3(3.0)
    small = 10.0 / units.AU_M
    big = 1.0e5 / units.AU_M
    assert units.disruption_threshold(small, rho, 3.0) > \
           2.5 * units.disruption_threshold(small, rho, 1.0)
    assert math.isclose(units.disruption_threshold(big, rho, 3.0),
                        units.disruption_threshold(big, rho, 1.0), rel_tol=0.01)


def test_every_material_declares_toughness_and_vapour_point():
    for m in materials.MATERIALS:
        assert m.toughness > 0.0
        assert m.vapour_temperature_k > 0.0
