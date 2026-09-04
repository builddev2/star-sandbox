"""The Fly tool: launching a craft and finding it again."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import pytest

from src import scenarios, units
from src.app.state import AppState
from src.app.tools import FlyTool
from src.body import KIND_CRAFT
from src.render.camera import Camera
from src.render.draw import ViewSettings
from src.sim import Simulation

SIZE = (640, 360)


@pytest.fixture
def state():
    pygame.init()
    sim = Simulation(world=scenarios.solar_system(moons=False))
    camera = Camera(SIZE[0], SIZE[1])
    app = AppState(sim, camera, ViewSettings())
    app.set_tool(FlyTool)
    return app


def _earth_index(state):
    names = [b.name for b in state.sim.world.bodies]
    return names.index("Earth")


def test_the_tool_registers_under_key_four():
    from src.app.tools import TOOLS
    assert TOOLS[pygame.K_4] is FlyTool
    from src.app.tools import TOOL_BY_NAME
    assert TOOL_BY_NAME["fly"] is FlyTool


def test_launch_places_a_craft_in_solar_orbit(state):
    """Not a parking orbit around Earth: that would drive the block timestep
    to its ceiling and step the whole solar system in seconds."""
    from src import orbits
    state.select(state.sim.world.bodies[_earth_index(state)].id)
    state.tool._launch(state)
    craft = state.selected_body
    assert craft.kind == KIND_CRAFT
    index = state.selected_index
    attractor = orbits.dominant_attractor(state.sim.world, index)
    assert state.sim.world.bodies[attractor].name == "Sun"


def test_a_launched_craft_has_a_full_tank(state):
    state.select(state.sim.world.bodies[_earth_index(state)].id)
    state.tool._launch(state)
    fuel = units.to_ms(state.selected_body.fuel_dv)
    assert abs(fuel - units.DEFAULT_FUEL_MS) < 1.0


def test_a_launched_craft_starts_near_the_body_it_launched_from(state):
    earth = _earth_index(state)
    earth_pos = np.array(state.sim.world.pos[earth])
    state.select(state.sim.world.bodies[earth].id)
    state.tool._launch(state)
    gap = float(np.hypot(*(state.sim.world.pos[state.selected_index] - earth_pos)))
    assert 0.0 < gap < 0.2


def test_launching_with_nothing_selected_does_nothing(state):
    before = state.sim.world.n
    state.selected_id = None
    state.tool._launch(state)
    assert state.sim.world.n == before


def test_the_craft_is_whatever_craft_is_selected(state):
    state.select(state.sim.world.bodies[_earth_index(state)].id)
    assert state.tool._craft(state) is None       # Earth is not a craft
    state.tool._launch(state)
    assert state.tool._craft(state) is state.selected_body


from src.app.ui import UiLayer
from src.render.hud import Hud
from src.render.pixelfont import PixelFont


@pytest.fixture
def ui():
    font = PixelFont(18)
    return UiLayer(font, Hud(font))


def _rows(state, ui):
    ui.build(state, SIZE)
    return [c for c in ui.controls if c.id.startswith("fly.")]


def test_the_panel_is_six_rows_and_no_more(state, ui):
    from src.app import config
    state.select(state.sim.world.bodies[_earth_index(state)].id)
    rows = state.tool.controls(state, ui.hud.metrics(*config.min_internal(3)))
    assert len(rows) == 6


def test_every_control_exists_with_no_craft_selected(state, ui):
    state.select(state.sim.world.bodies[_earth_index(state)].id)
    ids = {c.id for c in _rows(state, ui)}
    assert ids >= {"fly.launch", "fly.refuel", "fly.target", "fly.when",
                   "fly.prograde", "fly.radial", "fly.hohmann", "fly.clear"}


def test_node_controls_are_disabled_without_a_craft(state, ui):
    state.select(state.sim.world.bodies[_earth_index(state)].id)
    by_id = {c.id: c for c in _rows(state, ui)}
    assert not by_id["fly.prograde"].enabled
    assert not by_id["fly.refuel"].enabled
    assert by_id["fly.launch"].enabled


def test_scrubbing_prograde_creates_a_node_and_spends_nothing_yet(state, ui):
    state.select(state.sim.world.bodies[_earth_index(state)].id)
    state.tool._launch(state)
    fuel_before = state.selected_body.fuel_dv
    control = next(c for c in _rows(state, ui) if c.id == "fly.prograde")
    ui.perform(control, state)
    assert len(state.selected_body.plan) == 1
    assert state.selected_body.plan[0].prograde != 0.0
    assert state.selected_body.fuel_dv == fuel_before


def test_the_prograde_hotkey_does_nothing_without_a_craft(state):
    """A hotkey bypasses the enabled gate, so the key path re-checks."""
    state.select(state.sim.world.bodies[_earth_index(state)].id)
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT, mod=0)
    state.tool.field_index = 1
    state.tool.on_event(event, state)
    assert state.selected_body.plan == ()


def test_clearing_removes_the_node(state, ui):
    state.select(state.sim.world.bodies[_earth_index(state)].id)
    state.tool._launch(state)
    ui.perform(next(c for c in _rows(state, ui) if c.id == "fly.prograde"), state)
    assert state.selected_body.plan
    ui.perform(next(c for c in _rows(state, ui) if c.id == "fly.clear"), state)
    assert state.selected_body.plan == ()


def test_refuel_fills_the_tank(state, ui):
    state.select(state.sim.world.bodies[_earth_index(state)].id)
    craft = state.tool._launch(state)
    craft.fuel_dv = 0.0
    ui.perform(next(c for c in _rows(state, ui) if c.id == "fly.refuel"), state)
    assert abs(units.to_ms(craft.fuel_dv) - units.DEFAULT_FUEL_MS) < 1.0


def test_the_hohmann_button_plans_two_burns_to_the_target(state, ui):
    earth = _earth_index(state)
    names = [b.name for b in state.sim.world.bodies]
    state.select(state.sim.world.bodies[earth].id)
    craft = state.tool._launch(state)
    state.tool.target_id = state.sim.world.bodies[names.index("Mars")].id
    ui.perform(next(c for c in _rows(state, ui) if c.id == "fly.hohmann"), state)
    assert len(craft.plan) == 2
    assert craft.plan[0].due_time < craft.plan[1].due_time
    assert units.to_ms(craft.plan[0].prograde) > 2000.0


# --- the overlay --------------------------------------------------------------------


def _font():
    from src.render.pixelfont import PixelFont
    return PixelFont(18)


def _mars_id(state):
    names = [b.name for b in state.sim.world.bodies]
    return state.sim.world.bodies[names.index("Mars")].id


def _painted(surface):
    return pygame.image.tostring(surface, "RGB")


def test_the_overlay_draws_nothing_and_raises_nothing_without_a_craft(state):
    surface = pygame.Surface(SIZE)
    state.tool.draw_overlay(surface, state, _font())          # must not raise


def test_the_overlay_draws_the_predicted_orbit_of_a_planned_burn(state):
    from src import maneuver
    state.select(state.sim.world.bodies[_earth_index(state)].id)
    craft = state.tool._launch(state)
    surface = pygame.Surface(SIZE)
    before = _painted(surface)
    craft.plan = (maneuver.Maneuver(state.sim.world.time + 40.0,
                                    units.from_ms(2900.0)),)
    state.tool.draw_overlay(surface, state, _font())
    assert _painted(surface) != before


def test_the_overlay_draws_a_craft_with_no_plan_at_all(state):
    state.select(state.sim.world.bodies[_earth_index(state)].id)
    state.tool._launch(state)
    surface = pygame.Surface(SIZE)
    before = _painted(surface)
    state.tool.draw_overlay(surface, state, _font())
    # The current orbit and the fuel readout are drawn even with no node.
    assert _painted(surface) != before


def test_the_encounter_is_cached_between_identical_frames(state):
    state.select(state.sim.world.bodies[_earth_index(state)].id)
    state.tool._launch(state)
    state.tool.target_id = _mars_id(state)
    first = state.tool._encounter(state)
    again = state.tool._encounter(state)
    assert first is again


def test_the_encounter_cache_is_dropped_when_the_plan_changes(state):
    from src import maneuver
    state.select(state.sim.world.bodies[_earth_index(state)].id)
    craft = state.tool._launch(state)
    state.tool.target_id = _mars_id(state)
    first = state.tool._encounter(state)
    craft.plan = (maneuver.Maneuver(state.sim.world.time + 10.0,
                                    units.from_ms(2900.0)),)
    assert state.tool._encounter(state) is not first


def test_there_is_no_encounter_without_a_target(state):
    state.select(state.sim.world.bodies[_earth_index(state)].id)
    state.tool._launch(state)
    state.tool.target_id = None
    assert state.tool._encounter(state) is None


def test_the_encounter_finds_mars_within_a_sensible_distance(state):
    """A craft co-orbital with Earth and a locked Mars: the search must return
    a real closest approach, not None and not nonsense."""
    state.select(state.sim.world.bodies[_earth_index(state)].id)
    state.tool._launch(state)
    state.tool.target_id = _mars_id(state)
    found = state.tool._encounter(state)
    assert found is not None
    assert 0.0 < found["distance"] < 3.0
    assert found["time"] >= 0.0


def test_the_overlay_survives_the_target_being_deleted(state):
    state.select(state.sim.world.bodies[_earth_index(state)].id)
    state.tool._launch(state)
    state.tool.target_id = _mars_id(state)
    state.tool._encounter(state)
    world = state.sim.world
    world.remove(world.index_of(state.tool.target_id))
    state.sim.refresh()
    surface = pygame.Surface(SIZE)
    state.tool.draw_overlay(surface, state, _font())          # must not raise
    assert state.tool.target_id is None                        # lock cleared


def test_the_transfer_aims_at_the_targets_mean_radius_not_where_it_happens_to_be():
    """Mars's orbit is eccentric (e = 0.093), so its distance from the Sun
    swings between 1.38 and 1.67 AU.  A transfer aimed at wherever it is
    standing today arrives at a radius Mars has already left -- so the
    solver must use the target's semi-major axis, which is the mean radius
    the circular-orbit Hohmann arithmetic assumes anyway.
    """
    import pygame as _pg
    from src import maneuver, orbits, scenarios as _sc
    from src.app.state import AppState as _AS
    from src.app.tools import FlyTool as _FT
    from src.render.camera import Camera as _Cam
    from src.render.draw import ViewSettings as _VS
    from src.sim import Simulation as _Sim

    _pg.init()
    sim = _Sim(world=_sc.solar_system(moons=False))
    app = _AS(sim, _Cam(640, 360), _VS())
    app.set_tool(_FT)
    names = [b.name for b in sim.world.bodies]

    app.select(sim.world.bodies[names.index("Earth")].id)
    craft = app.tool._launch(app)
    app.tool.target_id = sim.world.bodies[names.index("Mars")].id
    app.tool._plan_transfer(app)
    assert len(craft.plan) == 2

    # Fly the whole plan.
    while craft.plan and sim.world.time < 2000.0:
        sim.advance(5.0, budget_s=2.0)
    assert craft.plan == ()

    index = sim.world.index_of(craft.id)
    _att, mu, rel_pos, rel_vel = maneuver.craft_state(sim.world, index)
    a_final = orbits.elements_from_state(mu, rel_pos, rel_vel)["a"]

    mars = sim.world.index_of(sim.world.bodies[names.index("Mars")].id)
    mars_a = orbits.elements_from_state(
        mu, sim.world.pos[mars] - sim.world.pos[0],
        sim.world.vel[mars] - sim.world.vel[0])["a"]

    # The craft should end on an orbit the size of Mars's, within a few percent.
    assert abs(a_final - mars_a) / mars_a < 0.05
