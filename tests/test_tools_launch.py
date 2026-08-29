import math
import os

import numpy as np
import pygame
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    pygame.init()
    pygame.display.set_mode((320, 200))
    yield
    pygame.quit()


@pytest.fixture
def state():
    from src.app.state import AppState
    from src.app.tools import LaunchTool
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.sim import Simulation
    st = AppState(Simulation(), Camera(1000, 700), ViewSettings())
    st.set_tool(LaunchTool)
    return st


def _down(x, y):
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(x, y), button=1)


def _move(x, y, rel=(0, 0)):
    return pygame.event.Event(pygame.MOUSEMOTION, pos=(x, y), rel=rel,
                              buttons=(1, 0, 0))


def _up(x, y):
    return pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(x, y), button=1)


def _keyup(key):
    return pygame.event.Event(pygame.KEYUP, key=key, mod=0)


# --- the conic maths --------------------------------------------------------------

def test_circular_orbit_traces_a_circle():
    from src import orbits, units
    mu = units.G
    pos, vel = orbits.state_from_elements(mu, 1.0, 0.0, 0.0, 0.0)
    points, bound = orbits.conic_points(mu, pos, vel, samples=64)
    assert bound is True
    radii = np.hypot(points[:, 0], points[:, 1])
    assert np.allclose(radii, 1.0, rtol=1e-9)


def test_ellipse_spans_periapsis_to_apoapsis():
    from src import orbits, units
    mu = units.G
    a, e = 2.0, 0.4
    pos, vel = orbits.state_from_elements(mu, a, e, 0.0, 0.0)
    points, bound = orbits.conic_points(mu, pos, vel, samples=256)
    radii = np.hypot(points[:, 0], points[:, 1])
    assert bound is True
    assert radii.min() == pytest.approx(a * (1 - e), rel=1e-4)
    assert radii.max() == pytest.approx(a * (1 + e), rel=1e-4)


def test_the_predicted_path_matches_where_the_body_actually_goes():
    """The whole point of drawing it: the preview must be the truth, not a hint."""
    from src import materials, orbits, scenarios, units
    from src.integrator import Integrator
    from src.world import World

    world = World()
    world.add("Star", materials.get("hydrogen"), 1.0, units.SUN_RADIUS,
              np.zeros(2), np.zeros(2))
    mu = units.G
    pos, vel = orbits.state_from_elements(mu, 1.6, 0.35, 0.4, 0.0)
    world.add("Rock", materials.get("rock"), 1e-14, units.EARTH_RADIUS * 0.01,
              pos, vel)
    world.zero_net_momentum()

    predicted, bound = orbits.conic_points(mu, pos, vel, samples=512)
    assert bound is True

    integrator = Integrator(world.capacity)
    integrator.refresh(world)
    for _ in range(4000):
        integrator.choose_k(world)
        integrator.step(world, integrator.dt)
        rel = world.pos[1] - world.pos[0]
        # Every actual position must lie on the predicted curve.
        gap = np.min(np.hypot(predicted[:, 0] - rel[0], predicted[:, 1] - rel[1]))
        assert gap < 0.02, "drifted %.4f AU off the predicted orbit" % gap


def test_escape_velocity_throw_is_reported_unbound():
    from src import orbits, units
    mu = units.G
    pos = np.array([1.0, 0.0])
    speed = math.sqrt(2.0 * mu / 1.0) * 1.3        # comfortably above escape
    vel = np.array([0.0, speed])
    points, bound = orbits.conic_points(mu, pos, vel, samples=64)
    assert bound is False
    assert len(points) >= 8
    assert np.isfinite(points).all()


def test_conic_of_a_body_at_rest_does_not_crash():
    from src import orbits, units
    points, bound = orbits.conic_points(units.G, np.array([1.0, 0.0]),
                                        np.zeros(2), samples=32)
    assert np.isfinite(points).all()


# --- the tool ---------------------------------------------------------------------

def test_presets_cover_the_promised_range():
    from src.app.tools import PRESETS
    labels = [p.label for p in PRESETS]
    assert len(PRESETS) == 6
    assert any("Asteroid" in x for x in labels)
    assert any("Star" in x for x in labels)
    assert any("Black hole" in x for x in labels)


def test_preset_bodies_have_sane_densities():
    from src.app.tools import PRESETS, preset_body
    from src.body import KIND_BLACK_HOLE
    from src import units
    for preset in PRESETS:
        mass, radius = preset_body(preset)
        assert mass > 0 and radius > 0
        # Black holes have extreme densities incomparable to ordinary matter.
        if preset.kind == KIND_BLACK_HOLE:
            continue
        density = units.to_gcm3(units.density_from_mass_radius(mass, radius))
        assert 0.05 < density < 25.0, "%s: %.3g g/cm3" % (preset.label, density)


def test_the_rocky_preset_is_about_an_earth():
    from src import units
    from src.app.tools import PRESETS, preset_body
    rocky = next(p for p in PRESETS if p.label == "Rocky planet")
    mass, radius = preset_body(rocky)
    assert radius == pytest.approx(units.EARTH_RADIUS, rel=1e-9)
    assert units.to_earth_masses(mass) == pytest.approx(1.0, rel=0.5)


def test_the_star_preset_actually_ignites():
    from src import units
    from src.app.tools import PRESETS, preset_body
    star = next(p for p in PRESETS if p.label == "Star")
    mass, _ = preset_body(star)
    assert mass >= units.STAR_MASS_THRESHOLD


def test_number_keys_choose_a_preset(state):
    state.tool.on_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_4, mod=0),
                        state)
    assert state.tool.preset_index == 3


def test_key_one_is_left_for_the_look_tool(state):
    """Pressing 1 while throwing must go back to Look, not pick preset 1."""
    before = state.tool.preset_index
    consumed = state.tool.on_event(
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_1, mod=0), state)
    assert consumed is False
    assert state.tool.preset_index == before


def test_drag_and_release_adds_a_body(state):
    before = state.sim.world.n
    state.tool.on_event(_down(500, 300), state)
    state.tool.on_event(_move(560, 300, (60, 0)), state)
    state.tool.on_event(_up(560, 300), state)
    assert state.sim.world.n == before + 1


def test_the_launched_body_is_selected_so_you_can_watch_it(state):
    state.tool.on_event(_down(500, 300), state)
    state.tool.on_event(_up(520, 300), state)
    assert state.selected_id == state.sim.world.bodies[-1].id


def test_launch_velocity_follows_the_drag_direction(state):
    state.tool.on_event(_down(500, 300), state)
    state.tool.on_event(_move(600, 300, (100, 0)), state)
    state.tool.on_event(_up(600, 300), state)
    assert state.sim.world.vel[state.sim.world.n - 1][0] > 0


def test_dragging_further_throws_harder(state):
    def speed_after(dx):
        state.tool.on_event(_down(500, 300), state)
        state.tool.on_event(_move(500 + dx, 300, (dx, 0)), state)
        state.tool.on_event(_up(500 + dx, 300), state)
        i = state.sim.world.n - 1
        v = float(np.hypot(*state.sim.world.vel[i]))
        state.sim.world.remove(i)
        return v

    assert speed_after(120) > speed_after(30)


def test_shift_snaps_to_a_circular_orbit(state):
    from src import orbits, units
    world = state.sim.world
    # Well clear of the planets, so the Sun is unambiguously the attractor.
    state.camera.centre = np.zeros(2)
    state.camera.log_zoom = math.log(1000 / 8.0)
    sx, sy = state.camera.world_to_screen(np.array([2.5, 0.0]))

    state.tool.shift = True
    state.tool.on_event(_down(int(sx), int(sy)), state)
    state.tool.on_event(_move(int(sx), int(sy) - 40, (0, -40)), state)
    state.tool.on_event(_up(int(sx), int(sy) - 40), state)

    index = world.n - 1
    attractor = orbits.dominant_attractor(world, index)
    mu = units.G * (world.mass[attractor] + world.mass[index])
    el = orbits.elements_from_state(mu, world.pos[index] - world.pos[attractor],
                                    world.vel[index] - world.vel[attractor])
    assert world.bodies[attractor].name == "Sun"
    assert el["e"] < 0.02, "shift should have produced a near-circular orbit, e=%.3f" % el["e"]


def test_overlay_draws_while_dragging(state):
    from src.render import palette
    from src.render.pixelfont import PixelFont
    surface = pygame.Surface((1000, 700))
    surface.fill(palette.SPACE)
    state.tool.on_event(_down(500, 300), state)
    state.tool.on_event(_move(600, 320, (100, 20)), state)
    state.tool.draw_overlay(surface, state, PixelFont(18))
    ink = sum(1 for x in range(0, 1000, 3) for y in range(0, 700, 3)
              if surface.get_at((x, y))[:3] != palette.SPACE[:3])
    assert ink > 20


def test_overlay_survives_a_near_parabolic_aim(state):
    """A path that reaches 1e12 pixels must not overflow the line rasteriser."""
    from src.render.pixelfont import PixelFont
    surface = pygame.Surface((1000, 700))
    state.camera.centre = np.zeros(2)
    sx, sy = state.camera.world_to_screen(np.array([1.0, 0.0]))
    state.tool.on_event(_down(int(sx), int(sy)), state)
    # Nudge the drag through the escape-velocity boundary a step at a time.
    for dx in range(0, 400, 7):
        state.tool.on_event(_move(int(sx) + dx, int(sy), (7, 0)), state)
        state.tool.draw_overlay(surface, state, PixelFont(18))


def test_every_preset_is_a_button_with_its_number_on_it(state):
    from src.app.tools import PRESETS
    from src.render.pixelfont import PixelFont
    from src.render.widgets import Metrics
    rows = state.tool.controls(state, Metrics(PixelFont(18), 1000, 700))
    presets = [c for row in rows for c in row if c.id == "throw.preset"]
    assert [c.label for c in presets] == [p.label for p in PRESETS]
    assert [c.hotkey for c in presets] == [str(p.key) for p in PRESETS]


def test_clicking_a_preset_button_selects_it(state):
    from src.render.pixelfont import PixelFont
    from src.render.widgets import Metrics
    rows = state.tool.controls(state, Metrics(PixelFont(18), 1000, 700))
    gas = next(c for row in rows for c in row
               if c.id == "throw.preset" and c.label == "Gas giant")
    assert state.tool.perform(gas, state)
    assert state.tool.preset_index == gas.payload
    assert state.tool.controls(state, Metrics(PixelFont(18), 1000, 700))


def test_the_circular_button_latches_what_holding_shift_does(state):
    from src.render.pixelfont import PixelFont
    from src.render.widgets import Metrics
    rows = state.tool.controls(state, Metrics(PixelFont(18), 1000, 700))
    button = next(c for row in rows for c in row if c.id == "throw.circular")

    assert not state.tool.circular
    state.tool.perform(button, state)
    assert state.tool.circular
    assert state.tool.lock_circular

    # Releasing a Shift that was never held must not cancel the latch.
    state.tool.on_event(_keyup(pygame.K_LSHIFT), state)
    assert state.tool.circular

    state.tool.perform(button, state)
    assert not state.tool.circular


def test_launching_into_a_full_world_is_refused_quietly(state):
    from src import materials, units
    world = state.sim.world
    while world.n < world.capacity:
        world.add("filler", materials.get("rock"), 1e-15, units.EARTH_RADIUS * 0.01,
                  np.array([500.0 + world.n, 0.0]), np.zeros(2))
    state.tool.on_event(_down(500, 300), state)
    state.tool.on_event(_up(520, 300), state)
    assert world.n == world.capacity


def test_a_quarter_screen_drag_throws_at_orbital_speed(state):
    """The gesture is measured against the local circular-orbit speed, so it
    means the same thing at every zoom."""
    from src import orbits, units
    from src.app.tools import DRAG_FRACTION
    world = state.sim.world
    state.camera.centre = np.zeros(2)
    state.camera.log_zoom = math.log(1000 / 8.0)
    sx, sy = state.camera.world_to_screen(np.array([3.0, 0.0]))
    drag = int(DRAG_FRACTION * state.camera.height)

    state.tool.on_event(_down(int(sx), int(sy)), state)
    state.tool.on_event(_move(int(sx), int(sy) - drag, (0, -drag)), state)
    state.tool.on_event(_up(int(sx), int(sy) - drag), state)

    index = world.n - 1
    attractor = orbits.dominant_attractor(world, index)
    mu = units.G * (world.mass[attractor] + world.mass[index])
    el = orbits.elements_from_state(mu, world.pos[index] - world.pos[attractor],
                                    world.vel[index] - world.vel[attractor])
    assert el["bound"] is True
    assert el["e"] < 0.35, "a circular-speed throw should be roughly circular"


def test_a_drag_of_the_same_length_means_the_same_at_any_zoom(state):
    """A fixed pixels-per-AU/day scale would make this fail badly."""
    from src.app.tools import DRAG_FRACTION
    world = state.sim.world
    drag = int(DRAG_FRACTION * state.camera.height)

    speeds = []
    for au_across in (8.0, 0.8, 0.08):
        state.camera.centre = np.array([3.0, 0.0])
        state.camera.log_zoom = math.log(state.camera.width / au_across)
        sx, sy = state.camera.world_to_screen(np.array([3.0, 0.0]))
        state.tool.on_event(_down(int(sx), int(sy)), state)
        state.tool.on_event(_move(int(sx), int(sy) - drag, (0, -drag)), state)
        state.tool.on_event(_up(int(sx), int(sy) - drag), state)
        i = world.n - 1
        speeds.append(float(np.hypot(*world.vel[i])))
        world.remove(i)

    assert max(speeds) / min(speeds) < 1.01, speeds


def test_dragging_far_enough_actually_escapes(state):
    """It must be possible to throw something free with one comfortable gesture."""
    from src import orbits, units
    from src.app.tools import DRAG_FRACTION
    world = state.sim.world
    state.camera.centre = np.zeros(2)
    state.camera.log_zoom = math.log(1000 / 8.0)
    sx, sy = state.camera.world_to_screen(np.array([3.0, 0.0]))
    drag = int(1.6 * DRAG_FRACTION * state.camera.height)
    assert drag < state.camera.height * 0.35, "gesture is too big to be comfortable"

    state.tool.on_event(_down(int(sx), int(sy)), state)
    state.tool.on_event(_move(int(sx), int(sy) - drag, (0, -drag)), state)
    state.tool.on_event(_up(int(sx), int(sy) - drag), state)

    index = world.n - 1
    attractor = orbits.dominant_attractor(world, index)
    mu = units.G * (world.mass[attractor] + world.mass[index])
    el = orbits.elements_from_state(mu, world.pos[index] - world.pos[attractor],
                                    world.vel[index] - world.vel[attractor])
    assert el["bound"] is False, "e=%.3f -- cannot throw anything free" % el["e"]


# --- target lock --------------------------------------------------------------

def _find(world, name):
    return next(b for b in world.bodies if b.name == name)


def test_pressing_x_then_clicking_a_body_locks_it_as_the_target(state):
    world = state.sim.world
    mars = _find(world, "Mars")
    sx, sy = state.camera.world_to_screen(world.pos[world.index_of(mars.id)])

    state.tool.on_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_x, mod=0), state)
    assert state.tool.picking_target
    state.tool.on_event(_down(int(sx), int(sy)), state)

    assert state.tool.target_id == mars.id
    assert not state.tool.picking_target


def test_clicking_empty_space_while_picking_clears_the_target(state):
    world = state.sim.world
    mars = _find(world, "Mars")
    state.tool.target_id = mars.id

    state.tool.on_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_x, mod=0), state)
    state.tool.on_event(_down(2, 2), state)          # a corner, nothing drawn there

    assert state.tool.target_id is None


def test_x_is_a_toggle(state):
    state.tool.on_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_x, mod=0), state)
    assert state.tool.picking_target
    state.tool.on_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_x, mod=0), state)
    assert not state.tool.picking_target


def test_a_locked_target_produces_a_launch_velocity_that_actually_intercepts_it(state):
    """Drives the tool's internal drag state directly, at an exact position
    and speed, rather than through pixel-rounded mouse events: direction is
    irrelevant once a target is locked, but position and speed are exactly
    what the solve uses, and even a sub-pixel difference in either can flip
    which of a mortar problem's two firing solutions comes back shortest."""
    from src import orbits, units
    from src.app.tools import DRAG_FRACTION
    world = state.sim.world
    mars = _find(world, "Mars")
    state.tool.target_id = mars.id

    # Launch from well clear of every planet, so the Sun is unambiguously
    # what the throw orbits -- matching the tool's own _nearest_attractor.
    launch_world = np.array([0.2, 0.0])
    mu = units.G * float(world.mass[0])
    rel = launch_world - world.pos[0]
    speed = 2.0 * math.sqrt(mu / float(np.hypot(*rel)))     # comfortably reachable

    reference_speed = state.tool._reference_speed(state, launch_world)
    span = max(1.0, DRAG_FRACTION * state.camera.height)
    length = speed / reference_speed * span
    origin_screen = state.camera.world_to_screen(launch_world)

    state.tool._origin = launch_world
    state.tool._cursor = (origin_screen[0], origin_screen[1] - length)
    state.tool._launch(state)

    index = world.n - 1
    assert world.bodies[index].name != "Mars"        # a new body was added

    target_index = world.index_of(mars.id)
    target_rel = world.pos[target_index] - world.pos[0]
    target_vel_rel = world.vel[target_index] - world.vel[0]
    expected = orbits.solve_intercept(mu, rel, speed, target_rel, target_vel_rel)
    assert expected["hit"], "test setup must pick a reachable geometry"

    got = world.vel[index] - world.vel[0]
    assert got == pytest.approx(expected["velocity"], abs=1e-6)


def test_target_lock_persists_across_multiple_throws(state):
    world = state.sim.world
    mars = _find(world, "Mars")
    state.tool.target_id = mars.id

    for _ in range(2):
        state.tool.on_event(_down(500, 300), state)
        state.tool.on_event(_move(560, 300, (60, 0)), state)
        state.tool.on_event(_up(560, 300), state)

    assert state.tool.target_id == mars.id


def test_a_target_that_gets_removed_clears_the_lock(state):
    world = state.sim.world
    mars = _find(world, "Mars")
    state.tool.target_id = mars.id
    world.remove(world.index_of(mars.id))

    assert state.tool._target_index(world) is None
    assert state.tool.target_id is None


def test_the_target_button_toggles_picking(state):
    from src.app.tools import LaunchTool
    control = next(c for row in state.tool.controls(state, None) for c in row
                   if c.id == "throw.target")
    assert state.tool.perform(control, state)
    assert state.tool.picking_target


def test_an_unreachable_target_falls_back_to_the_plain_drag(state):
    """A target too far away at this speed must not leave the throw unusable."""
    world = state.sim.world
    mars = _find(world, "Mars")
    state.tool.target_id = mars.id

    state.tool.on_event(_down(500, 300), state)
    state.tool.on_event(_move(600, 300, (100, 0)), state)   # a feeble, short drag
    state.tool.on_event(_up(600, 300), state)

    index = world.n - 1
    # Same behaviour as the plain (untargeted) drag test: rightward drag,
    # rightward launch velocity.
    assert world.vel[index][0] > 0
