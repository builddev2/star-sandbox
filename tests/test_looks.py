"""A look is how a named body wants to be drawn -- Earth blue, Mars red --
pinned by the scenario and dropped the moment the child changes what the body
is made of.  Appearance only: the physics never reads it."""

import numpy as np
import pytest

from src import edits, materials, scenarios
from src.body import Body
from src.world import World


def _world():
    return scenarios.solar_system(moons=True)


def _index(world, name):
    return next(i for i in range(world.n) if world.bodies[i].name == name)


def test_a_fresh_body_has_no_look():
    assert Body(1, "x", materials.get("rock")).look is None
    world = World()
    body = world.add("thrown", materials.get("rock"), 1e-8, 1e-5, np.zeros(2), np.zeros(2))
    assert body.look is None


def test_world_add_takes_a_look():
    world = World()
    body = world.add("Earth", materials.get("rock"), 1e-8, 1e-5, np.zeros(2), np.zeros(2),
                     look="terrestrial")
    assert body.look == "terrestrial"


def test_the_solar_system_pins_a_look_on_every_planet_and_the_moon():
    world = _world()
    expected = {"Mercury": "mercury", "Venus": "venus", "Earth": "terrestrial",
                "Mars": "mars", "Jupiter": "jupiter", "Saturn": "saturn",
                "Uranus": "uranus", "Neptune": "neptune", "Moon": "moon"}
    for name, look in expected.items():
        assert world.bodies[_index(world, name)].look == look, name
    for name in ("Sun", "Io", "Europa", "Ganymede", "Callisto", "Titan"):
        assert world.bodies[_index(world, name)].look is None, name


def test_changing_the_material_drops_the_look():
    world = _world()
    earth = _index(world, "Earth")
    edits.set_material(world, earth, materials.get("iron"))
    assert world.bodies[earth].look is None


def test_cycling_the_material_drops_the_look():
    world = _world()
    mars = _index(world, "Mars")
    edits.cycle_material(world, mars)
    assert world.bodies[mars].look is None


def test_a_locked_material_keeps_the_look_too():
    world = _world()
    earth = _index(world, "Earth")
    world.bodies[earth].locked_fields = frozenset({"material"})
    before = world.bodies[earth].material
    edits.set_material(world, earth, materials.get("iron"))
    assert world.bodies[earth].material is before
    assert world.bodies[earth].look == "terrestrial"


def test_a_mass_edit_leaves_the_look_alone():
    world = _world()
    earth = _index(world, "Earth")
    edits.set_mass(world, earth, float(world.mass[earth]) * 2.0)
    assert world.bodies[earth].look == "terrestrial"


def test_a_density_scrub_that_changes_the_material_drops_the_look():
    world = _world()
    earth = _index(world, "Earth")
    before = world.bodies[earth].material
    # Scrub far enough to land on a different material (rock 3.30 g/cm3 ->
    # iron 7.87 g/cm3, MAX_DENSITY_GCM3 25.0 -- 1.06**60 clears the midpoint
    # many times over).
    for _ in range(60):
        edits.set_density(world, earth, edits.body_density(world, earth) * 1.06)
    assert world.bodies[earth].material is not before
    assert world.bodies[earth].look is None


def test_a_density_scrub_that_keeps_the_material_keeps_the_look():
    world = _world()
    earth = _index(world, "Earth")
    edits.set_density(world, earth, world.bodies[earth].material.density * 1.01)
    assert world.bodies[earth].look == "terrestrial"


def _session():
    from src import save
    return save.Session(scenario="solar_system", time_index=5, paused=False,
                        show_trails=True, real_size=False, show_labels=True,
                        explain_enabled=True, show_body_list=True, selected_id=None,
                        centre=(0.25, -1.5), follow_id=None, au_across=13.0)


def test_a_look_survives_a_save_and_a_file_without_one_loads_as_none():
    import json
    from src import save
    world = _world()
    data = json.loads(json.dumps(save.to_data(world, _session(), "x", "Solar")))
    earth = next(b for b in data["world"]["bodies"] if b["name"] == "Earth")
    assert earth["look"] == "terrestrial"
    fresh = World()
    save.from_data(data, fresh)
    assert fresh.bodies[_index(fresh, "Earth")].look == "terrestrial"
    for b in data["world"]["bodies"]:
        del b["look"]
    older = World()
    save.from_data(data, older)
    assert all(b.look is None for b in older.bodies[:older.n])
