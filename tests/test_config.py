import pytest

from src.app import config
from src.app.config import font_px, internal_size


@pytest.mark.parametrize("w,h,s,expected", [
    # A 16" MacBook Pro reports 1728x1117 logical points; see config.py.
    (1728, 1117, 1, (1728, 1117)),
    (1728, 1117, 2, (864, 558)),
    (1728, 1117, 3, (576, 372)),
    (1920, 1080, 2, (960, 540)),
])
def test_internal_size_divides_exactly(w, h, s, expected):
    assert internal_size(w, h, s) == expected


def test_internal_size_never_returns_zero():
    assert internal_size(10, 10, 4) == (2, 2)


def test_default_grid_is_the_finest_one():
    """SDL only exposes logical points, so one internal pixel per point is as
    sharp as this can get -- and lands as an exact 2x2 block on a Retina panel."""
    assert config.DEFAULT_S == min(config.GRID_CHOICES)
    assert config.DEFAULT_S == 1


def test_text_is_the_intended_size_at_the_default_grid():
    assert font_px(1) == config.TARGET_TEXT_POINTS


def test_text_stays_legible_at_every_grid():
    for s in config.GRID_CHOICES:
        assert font_px(s) >= config.MIN_FONT_PX
        # Apparent height in logical points.
        assert 12 <= font_px(s) * s <= 26, "s=%d looks wrong" % s


def test_coarser_grids_never_use_a_bigger_buffer():
    for finer, coarser in zip(config.GRID_CHOICES, config.GRID_CHOICES[1:]):
        a = internal_size(1728, 1117, finer)
        b = internal_size(1728, 1117, coarser)
        assert b[0] < a[0] and b[1] < a[1]
