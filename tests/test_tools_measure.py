"""The Measure tool: its panel, its keys, and where it sits."""

import math
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import pytest

from src import materials, measure, scenarios, units
from src.app.state import AppState
from src.app.tools import MeasureTool
from src.render import hud
from src.render.camera import Camera
from src.render.draw import ViewSettings
from src.sim import Simulation
from src.world import World

SIZE = (640, 360)


@pytest.fixture
def state():
    pygame.init()
    sim = Simulation(world=scenarios.solar_system(moons=False))
    app = AppState(sim, Camera(*SIZE), ViewSettings())
    app.set_tool(MeasureTool)
    return app


def _body(state, name):
    return next(b for b in state.sim.world.bodies if b.name == name)


def _key(key):
    return pygame.event.Event(pygame.KEYDOWN, key=key, mod=0)


def _controls(state):
    return {c.id: c for row in state.tool.controls(state, None) for c in row}


def test_the_tool_registers_under_key_five():
    from src.app.tools import TOOL_BY_NAME, TOOLS
    assert TOOLS[pygame.K_5] is MeasureTool
    assert TOOL_BY_NAME["measure"] is MeasureTool


def test_the_hotbar_has_a_measure_button():
    from src.app.ui import TOOL_BUTTONS
    assert ("tool.measure", "Measure") in {(b[0], b[1]) for b in TOOL_BUTTONS}


def test_timing_is_disabled_and_says_why_with_nothing_selected(state):
    time = _controls(state)["measure.time"]
    assert not time.enabled
    assert time.label == "Pick a body first"
    assert not _controls(state)["measure.stop"].enabled


def test_the_button_starts_timing_the_selection(state):
    # Deleting the measure.time branch of perform makes this fail.
    state.select(_body(state, "Mars").id)
    time = _controls(state)["measure.time"]
    assert time.enabled and time.hotkey == "P"
    assert state.tool.perform(time, state)
    assert state.period_timer.body_id == _body(state, "Mars").id
    assert state.tool.perform(_controls(state)["measure.stop"], state)
    assert state.period_timer is None


def test_the_target_button_arms_picking(state):
    # Deleting the measure.target branch of perform makes this fail.
    target = _controls(state)["measure.target"]
    assert state.tool.perform(target, state)
    assert state.tool.picking_target
    assert state.tool.perform(target, state)
    assert not state.tool.picking_target


def test_p_times_the_selection_and_does_nothing_without_one(state):
    assert state.tool.on_event(_key(pygame.K_p), state)
    assert state.period_timer is None
    state.select(_body(state, "Mars").id)
    state.tool.on_event(_key(pygame.K_p), state)
    assert state.period_timer.body_id == _body(state, "Mars").id


def test_x_then_click_locks_the_target(state):
    earth = _body(state, "Earth")
    state.camera.frame_body(state.sim.world, state.sim.world.index_of(earth.id))
    x, y = state.camera.world_to_screen(state.sim.world.pos[state.sim.world.index_of(earth.id)])
    state.tool.on_event(_key(pygame.K_x), state)
    assert state.tool.picking_target
    state.tool.on_event(pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, button=1, pos=(int(x), int(y))), state)
    assert state.tool.target_id == earth.id
    assert _controls(state)["measure.target"].label == "Target: Earth"


def test_a_plain_click_selects(state):
    mars = _body(state, "Mars")
    index = state.sim.world.index_of(mars.id)
    state.camera.frame_body(state.sim.world, index)
    pos = tuple(int(v) for v in state.camera.world_to_screen(state.sim.world.pos[index]))
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos), state)
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=pos), state)
    assert state.selected_id == mars.id


def test_every_control_explains_itself(state):
    state.select(_body(state, "Mars").id)
    state.start_timer()
    for control in _controls(state).values():
        assert control.hint(), control.id


def test_the_keys_are_documented():
    from src.render.hud import KEY_HELP
    combos = {combo for combo, _ in KEY_HELP}
    assert "P" in combos
    assert any("5" in combo.split() for combo in combos)


@pytest.mark.parametrize("name", ["measure", "clock"])
def test_the_new_icons_draw_something(name):
    # Deleting either draw_icon branch leaves the surface blank.
    from src.render.widgets import draw_icon
    surface = pygame.Surface((24, 24))
    surface.fill((0, 0, 0))
    draw_icon(surface, name, pygame.Rect(2, 2, 20, 20), (255, 255, 255))
    assert pygame.mask.from_threshold(surface, (255, 255, 255), (60, 60, 60, 255)).count() > 10


def _two_body_state():
    pygame.init()
    sim = Simulation(world=scenarios.two_body())
    app = AppState(sim, Camera(*SIZE), ViewSettings())
    app.set_tool(MeasureTool)
    return app


def _texts(state):
    return [text for text, _ in state.tool.readout_lines(state)]


def test_nothing_to_say_with_nothing_chosen(state):
    assert state.tool.readout_lines(state) == []


def test_the_ruler_reads_light_minutes_to_the_sun():
    """One AU exactly, from the two-body scenario: 8.3 light-minutes."""
    st = _two_body_state()
    st.select(st.sim.world.bodies[1].id)
    st.tool.target_id = st.sim.world.bodies[0].id
    texts = _texts(st)
    assert texts[0] == "Secondary to Primary"
    assert "Apart  1 AU" in texts
    assert "Light takes 8.3 min" in texts
    assert not any(t.startswith("Gap") for t in texts)      # too far to matter


def test_the_gap_appears_once_they_are_close():
    # Earth to the Moon is well inside 0.02 AU.
    pygame.init()
    sim = Simulation(world=scenarios.solar_system())
    st = AppState(sim, Camera(*SIZE), ViewSettings())
    st.set_tool(MeasureTool)
    st.select(_body(st, "Earth").id)
    st.tool.target_id = _body(st, "Moon").id
    assert any(t.startswith("Gap") and t.endswith("km") for t in _texts(st))


def test_the_timer_reads_measured_against_kepler():
    """After one lap: the measured period, Kepler's, how far apart (signed,
    F5), and the mass it implies.  Removing any of the four lines fails this."""
    st = _two_body_state()
    st.select(st.sim.world.bodies[1].id)
    st.start_timer()
    while st.period_timer.measured is None and st.sim.world.time < 1000.0:
        st.sim.advance(1.0, budget_s=1e9)
        st.sample_timer()
    texts = _texts(st)
    assert texts[0] == "Timing Secondary round Primary"
    assert "Measured    365.3 d" in texts
    assert "Kepler says 365.3 d" in texts
    # Measured lands a hair above Kepler at this frame rate -- "slower", not
    # the old unsigned "Off by".  Swapping the verb makes this fail.
    assert any(t.endswith("% slower than Kepler") for t in texts)
    assert "a^3/P^2 weighs 1 Suns" in texts


def test_too_fast_is_said_out_loud():
    st = _two_body_state()
    st.select(st.sim.world.bodies[1].id)
    st.start_timer()
    st.sim.advance(200.0, budget_s=1e9)
    st.sample_timer()
    assert "Too fast to time -- slow down" in _texts(st)


def test_a_line_is_drawn_between_the_two():
    """A pixel a quarter of the way along the line is lit only when there is
    a target.  Deleting the aaline call makes this fail."""
    from src.render.pixelfont import PixelFont
    st = _two_body_state()
    st.show_body_list = False
    st.camera.log_zoom = math.log(SIZE[0] / 3.0)     # pixels per AU: 3 AU across
    st.camera.centre[:] = 0.0
    st.select(st.sim.world.bodies[1].id)
    st.tool.target_id = st.sim.world.bodies[0].id
    world = st.sim.world
    a = st.camera.world_to_screen(world.pos[1])
    b = st.camera.world_to_screen(world.pos[0])
    probe = (int(round(a[0] + 0.25 * (b[0] - a[0]))), int(round(a[1] + 0.25 * (b[1] - a[1]))))
    # A probe off screen would make this test pass or fail for no reason.
    assert pygame.Rect(0, 0, *SIZE).collidepoint(probe)
    surface = pygame.Surface(SIZE)
    surface.fill((0, 0, 0))
    st.tool.draw_overlay(surface, st, PixelFont(12))
    lit = max(surface.get_at((probe[0] + dx, probe[1] + dy))[:3]
              for dx in (-1, 0, 1) for dy in (-1, 0, 1))
    assert sum(lit) > 150


# --- F3: closing vs opening --------------------------------------------------------

def _radial_pair_state(vel_b):
    """A at the origin, B one AU out on +x, moving along that same line."""
    pygame.init()
    world = World()
    rock = materials.get("rock")
    world.add("A", rock, units.EARTH_MASS, 0.0, np.zeros(2), np.zeros(2))
    world.add("B", rock, units.EARTH_MASS, 0.0, np.array([1.0, 0.0]), np.array(vel_b))
    st = AppState(Simulation(world=world), Camera(*SIZE), ViewSettings())
    st.set_tool(MeasureTool)
    st.select(world.bodies[1].id)
    st.tool.target_id = world.bodies[0].id
    return st


def test_a_target_moving_closer_reads_closing():
    # Swapping "Closing"/"Opening" in readout_lines makes this fail.
    texts = _texts(_radial_pair_state((-0.01, 0.0)))
    line = next(t for t in texts if t.startswith(("Closing", "Opening")))
    assert line.startswith("Closing at")


def test_a_target_moving_away_reads_opening():
    # Swapping "Closing"/"Opening" in readout_lines makes this fail.
    texts = _texts(_radial_pair_state((0.01, 0.0)))
    line = next(t for t in texts if t.startswith(("Closing", "Opening")))
    assert line.startswith("Opening at")


# --- F5: a signed comparison against Kepler ----------------------------------------

def test_measured_faster_than_kepler_says_so():
    """2-day frames on an e=0.8 orbit measure a shade short of the lap-mean
    Kepler prediction (measured, verified directly: 365.70 against 365.84).
    Flipping the faster/slower branch makes this fail."""
    world = scenarios.two_body(eccentricity=0.8)
    st = AppState(Simulation(world=world), Camera(*SIZE), ViewSettings())
    st.set_tool(MeasureTool)
    st.select(world.bodies[1].id)
    st.start_timer()
    while st.period_timer.measured is None and st.sim.world.time < 1000.0:
        st.sim.advance(2.0, budget_s=1e9)
        st.sample_timer()
    assert st.period_timer.measured < st.period_timer.lap_kepler
    texts = _texts(st)
    assert any(t.endswith("% faster than Kepler") for t in texts)


# --- F1 + F2: the readout uses the lap's frozen values, not the live ones ----------

def test_the_readout_uses_the_lap_values_once_one_exists():
    """Sun-perturbed Moon: well into the lap after the first one, the live
    osculating Kepler prediction has wandered measurably away from the
    frozen lap value.  The readout must show the frozen one.  Reading the
    live `measure.kepler_period` here instead of `timer.lap_kepler` makes
    this fail."""
    world = scenarios.solar_system(moons=True)
    for body in list(world.bodies):
        if body.name not in ("Sun", "Earth", "Moon"):
            world.remove(world.index_of(body.id))
    st = AppState(Simulation(world=world), Camera(*SIZE), ViewSettings())
    st.set_tool(MeasureTool)
    moon = next(b for b in world.bodies if b.name == "Moon")
    st.select(moon.id)
    st.start_timer()
    while len(st.period_timer.laps) < 3 and st.sim.world.time < 100.0:
        st.sim.advance(0.5, budget_s=1e9)
        st.sample_timer()
    for _ in range(20):
        st.sim.advance(0.5, budget_s=1e9)
        st.sample_timer()
    assert st.period_timer.lap_kepler is not None
    texts = _texts(st)
    assert ("Kepler says %s" % hud.format_period(st.period_timer.lap_kepler)) in texts


# --- F6: the attractor being gone reads differently from never having one ---------

def test_after_a_restart_kepler_is_the_live_prediction():
    """Complete a lap, go too fast (banking a stale Kepler number), quadruple
    the Primary's mass, then resume at 1-day frames. The readout's "Kepler
    says" line must track the live prediction for the new mass, not the
    dropped lap's old 365.3 d. Leaving lap_a/lap_kepler alone in _restart
    makes this fail."""
    st = _two_body_state()
    st.select(st.sim.world.bodies[1].id)
    st.start_timer()
    while st.period_timer.measured is None and st.sim.world.time < 1000.0:
        st.sim.advance(1.0, budget_s=1e9)
        st.sample_timer()
    assert st.period_timer.lap_kepler is not None

    st.sim.advance(200.0, budget_s=1e9)
    st.sample_timer()
    assert st.period_timer.status == "too fast"

    st.sim.world.mass[0] *= 4.0
    st.sim.refresh()
    for _ in range(5):
        st.sim.advance(1.0, budget_s=1e9)
        st.sample_timer()

    live, _ = measure.kepler_period(st.sim.world, 1, 0)
    texts = _texts(st)
    assert ("Kepler says %s" % hud.format_period(live)) in texts
    assert "Kepler says 365.3 d" not in texts


def test_the_attractor_being_deleted_reads_differently_from_never_orbiting():
    """The Sun (no attractor from the start) keeps the old wording; a body
    whose attractor was removed after timing started gets the new one.
    Making both branches share a message makes one of these two fail."""
    st = _two_body_state()
    sun = st.sim.world.bodies[0]
    st.select(sun.id)
    st.start_timer()
    assert st.period_timer.attractor_id is None
    assert "Primary is not orbiting anything" in _texts(st)

    st2 = _two_body_state()
    secondary = st2.sim.world.bodies[1]
    st2.select(secondary.id)
    st2.start_timer()
    st2.sim.world.remove(st2.sim.world.index_of(st2.sim.world.bodies[0].id))
    st2.sim.refresh()
    texts2 = _texts(st2)
    assert "What it orbited is gone" in texts2
    assert "Secondary is not orbiting anything" not in texts2
