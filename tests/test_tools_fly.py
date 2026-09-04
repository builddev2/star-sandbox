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
