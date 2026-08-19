import numpy as np
import pytest

from src.render.camera import Camera


def test_centre_of_the_view_maps_to_the_camera_centre():
    cam = Camera(1728, 1084)
    cam.centre = np.array([3.0, -2.0])
    x, y = cam.world_to_screen(cam.centre)
    assert x == pytest.approx(1728 / 2)
    assert y == pytest.approx(1084 / 2)


def test_screen_to_world_inverts_world_to_screen():
    cam = Camera(1728, 1084)
    cam.centre = np.array([1.5, 0.25])
    point = np.array([4.2, -3.7])
    x, y = cam.world_to_screen(point)
    assert np.allclose(cam.screen_to_world(x, y), point)


def test_world_y_points_up_on_screen():
    """Screen y grows downward; world y grows upward.  Getting this backwards
    silently mirrors every orbit."""
    cam = Camera(800, 600)
    _, y_high = cam.world_to_screen(np.array([0.0, 1.0]))
    _, y_low = cam.world_to_screen(np.array([0.0, -1.0]))
    assert y_high < y_low


def test_default_view_spans_the_requested_width_in_au():
    cam = Camera(1400, 900, au_across=70.0)
    left = cam.screen_to_world(0, 450)
    right = cam.screen_to_world(1400, 450)
    assert (right[0] - left[0]) == pytest.approx(70.0, rel=1e-9)


def test_zoom_at_a_point_keeps_that_point_still():
    """Wheel-zoom must be anchored at the cursor, or the view slides away from
    whatever you were looking at."""
    cam = Camera(1728, 1084)
    sx, sy = 400.0, 300.0
    before = cam.screen_to_world(sx, sy)
    cam.zoom_by(3, sx, sy)
    after = cam.screen_to_world(sx, sy)
    assert np.allclose(before, after, atol=1e-9)


def test_zoom_in_increases_scale():
    cam = Camera(800, 600)
    z0 = cam.zoom
    cam.zoom_by(1)
    assert cam.zoom > z0
    cam.zoom_by(-1)
    assert cam.zoom == pytest.approx(z0, rel=1e-12)


def test_pan_moves_the_centre_by_the_right_number_of_au():
    cam = Camera(800, 600)
    start = cam.centre.copy()
    cam.pan_pixels(80, 0)
    assert cam.centre[0] == pytest.approx(start[0] - 80 / cam.zoom)


def test_follow_locks_the_centre_onto_a_body_by_id():
    from src import scenarios
    world = scenarios.solar_system()
    cam = Camera(800, 600)
    jupiter = next(b for b in world.bodies if b.name == "Jupiter")
    cam.follow_id = jupiter.id
    cam.update_follow(world)
    assert np.allclose(cam.centre, world.pos[world.index_of(jupiter.id)])


def test_follow_of_a_destroyed_body_is_dropped_not_crashed():
    from src import scenarios
    world = scenarios.solar_system()
    cam = Camera(800, 600)
    doomed = world.bodies[-1]
    cam.follow_id = doomed.id
    world.remove(world.index_of(doomed.id))
    cam.update_follow(world)          # must not raise
    assert cam.follow_id is None


def test_world_to_screen_array_matches_the_scalar_version():
    cam = Camera(800, 600)
    pts = np.array([[0.0, 0.0], [1.0, 2.0], [-3.0, 0.5]])
    batch = cam.world_to_screen_array(pts)
    for k, p in enumerate(pts):
        assert np.allclose(batch[k], cam.world_to_screen(p))


def test_framing_a_tiny_body_does_not_blow_past_the_zoom_limit():
    from src import scenarios
    from src.render.camera import MAX_ZOOM
    world = scenarios.solar_system()
    cam = Camera(1728, 1084)
    moon = next(b for b in world.bodies if b.name == "Moon")
    cam.frame_body(world, world.index_of(moon.id))
    assert cam.zoom <= MAX_ZOOM
    assert np.allclose(cam.centre, world.pos[world.index_of(moon.id)])


def test_the_default_view_shows_the_planets_that_visibly_move():
    """Opening on a 70 AU view is mostly empty space with the inner planets an
    indistinct cluster; nothing appears to happen."""
    cam = Camera(1728, 1084)
    span = cam.screen_to_world(1728, 542)[0] - cam.screen_to_world(0, 542)[0]
    assert 8.0 <= span <= 22.0, "default view spans %.1f AU" % span
    # Jupiter at 5.2 AU must be comfortably inside the frame.
    x, _ = cam.world_to_screen(np.array([5.2, 0.0]))
    assert 0 < x < 1728
