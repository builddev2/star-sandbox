import pytest

from src.app import config
from src.app.config import font_px, internal_size, min_internal


@pytest.mark.parametrize("w,h,s,expected", [
    # A 16" MacBook Pro reports 1728x1117 logical points; see config.py.
    (1728, 1117, 1, (1728, 1117)),
    (1728, 1117, 2, (3456, 2234)),
    (1920, 1080, 2, (3840, 2160)),
])
def test_internal_size_is_the_window_times_the_pixels_per_point(w, h, s, expected):
    assert internal_size(w, h, s) == expected


def test_internal_size_never_returns_zero():
    assert internal_size(0, 0, 1) == (1, 1)


def test_text_is_the_intended_size_at_one_pixel_per_point():
    assert font_px(1) == config.TARGET_TEXT_POINTS


def test_text_doubles_with_the_pixel_density():
    """Apparent size is what stays constant, not the pixel count."""
    assert font_px(2) == 2 * font_px(1)


def test_the_interface_floor_scales_with_the_pixel_density():
    assert min_internal(2) == (2 * config.MIN_INTERNAL[0], 2 * config.MIN_INTERNAL[1])
    assert min_internal(1) == config.MIN_INTERNAL
