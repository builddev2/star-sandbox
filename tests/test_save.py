"""A world goes to plain data and comes back the same, to the last bit."""

import json

import numpy as np
import pytest

from src import edits, materials, save, scenarios, units
from src.body import KIND_BLACK_HOLE, KIND_CRAFT, KIND_STAR
from src.maneuver import Maneuver
from src.world import World


def session(**over):
    base = dict(scenario="solar_system", time_index=5, paused=False,
                show_trails=True, real_size=False, show_labels=True,
                explain_enabled=True, show_body_list=True, selected_id=None,
                centre=(0.25, -1.5), follow_id=None, au_across=13.0)
    base.update(over)
    return save.Session(**base)


def busy_world():
    """Everything a save has to carry: a hole with spin, a craft with a plan,
    a locked planet, rings, a body with collision immunity."""
    world = scenarios.solar_system()
    sun = world.bodies[0]
    assert sun.kind == KIND_STAR
    edits.collapse(world, 0)
    sun.spin, sun.disk = 0.7, 0.6
    earth = next(i for i, b in enumerate(world.bodies) if b.name == "Earth")
    world.bodies[earth].locked_fields = frozenset({"mass", "temp"})
    world.temperature[earth] = 1234.5
    world.bodies[earth].event_id = 7
    world.bodies[earth].no_collide_until = world.time + 3.0
    saturn = next(b for b in world.bodies if b.name == "Saturn")
    assert saturn.rings
    craft = world.add("Shuttle", materials.SPACECRAFT, units.EARTH_MASS * 1e-18,
                      units.CRAFT_RADIUS, world.pos[earth] + (0.01, 0.0),
                      world.vel[earth] + (0.0, 0.001), kind=KIND_CRAFT,
                      temperature=250.0)
    craft.fuel_dv = 0.0123
    craft.plan = (Maneuver(10.0, prograde=0.001), Maneuver(20.0, radial=-0.0005))
    world.time = 812.5
    return world


def roundtrip(world, sess=None):
    data = json.loads(json.dumps(save.to_data(world, sess or session(),
                                               "Test", "Solar")))
    target = World()
    got = save.from_data(data, target)
    return data, target, got


def test_every_array_survives_to_the_last_bit():
    world = busy_world()
    _, target, _ = roundtrip(world)
    n = world.n
    assert target.n == n
    for name in ("pos", "vel", "mass", "radius", "temperature"):
        assert np.array_equal(getattr(world, name)[:n], getattr(target, name)[:n]), name
    assert np.array_equal(target.contact_radius[:n], target.radius[:n])
    assert target.time == world.time


def test_every_body_field_survives():
    world = busy_world()
    _, target, _ = roundtrip(world)
    for a, b in zip(world.bodies, target.bodies):
        assert (a.id, a.name, a.kind, a.material.key) == (b.id, b.name, b.kind, b.material.key)
        assert (a.born_day, a.event_id, a.no_collide_until, a.seed) == \
            (b.born_day, b.event_id, b.no_collide_until, b.seed)
        assert (a.rings, a.spin, a.disk, a.fuel_dv) == (b.rings, b.spin, b.disk, b.fuel_dv)
        assert a.locked_fields == b.locked_fields
        assert a.plan == b.plan
        assert isinstance(b.locked_fields, frozenset)
        assert isinstance(b.plan, tuple)
    hole = target.bodies[0]
    assert hole.kind == KIND_BLACK_HOLE and hole.spin == 0.7
    craft = target.bodies[-1]
    assert craft.kind == KIND_CRAFT and craft.plan[1].radial == -0.0005


def test_ids_and_the_counter_are_preserved():
    world = busy_world()
    _, target, _ = roundtrip(world)
    for body in world.bodies:
        assert target.index_of(body.id) is not None
        assert target.body_by_id(body.id).name == body.name
    assert target.next_id == world.next_id
    fresh = target.add("New", materials.ROCK, 1e-6, 1e-5, (5.0, 0.0), (0.0, 0.01))
    assert fresh.id not in {b.id for b in world.bodies}


def test_the_session_comes_back_and_the_camera_targets_resolve():
    world = busy_world()
    earth_id = next(b.id for b in world.bodies if b.name == "Earth")
    _, _, got = roundtrip(world, session(selected_id=earth_id, follow_id=earth_id,
                                         paused=True, time_index=8))
    assert got.selected_id == earth_id and got.follow_id == earth_id
    assert got.paused is True and got.time_index == 8
    assert got.centre == (0.25, -1.5) and got.au_across == 13.0


def test_a_camera_target_that_is_not_in_the_file_loads_as_none():
    world = busy_world()
    _, _, got = roundtrip(world, session(selected_id=999999, follow_id=999999))
    assert got.selected_id is None and got.follow_id is None


def test_the_data_is_plain_json_with_a_header():
    world = busy_world()
    from datetime import datetime
    data = save.to_data(world, session(), "Mars trip", "Solar",
                        now=datetime(2026, 9, 12, 14, 3))
    text = json.dumps(data)          # raises on any numpy scalar
    assert data["version"] == save.VERSION
    assert data["name"] == "Mars trip"
    assert data["saved_at"] == "2026-09-12T14:03"
    assert data["summary"] == {"scenario_label": "Solar", "days": 812.5,
                               "bodies": world.n}
    assert "numpy" not in text


def test_an_unknown_version_raises():
    world = busy_world()
    data = save.to_data(world, session(), "x", "Solar")
    data["version"] = 99
    with pytest.raises(save.SaveFormatError):
        save.from_data(data, World())


def test_an_unknown_kind_raises_and_leaves_the_target_untouched():
    world = busy_world()
    data = save.to_data(world, session(), "x", "Solar")
    data["world"]["bodies"][3]["kind"] = "teapot"
    target = scenarios.solar_system()
    before = target.pos[:target.n].copy()
    n = target.n
    with pytest.raises(save.SaveFormatError):
        save.from_data(data, target)
    assert target.n == n
    assert np.array_equal(target.pos[:n], before)


def test_an_unknown_material_falls_back_to_rock():
    world = busy_world()
    data = save.to_data(world, session(), "x", "Solar")
    data["world"]["bodies"][3]["material"] = "unobtainium"
    target = World()
    save.from_data(data, target)
    assert target.bodies[3].material is materials.ROCK


def test_too_many_bodies_raises():
    world = busy_world()
    data = save.to_data(world, session(), "x", "Solar")
    rec = data["world"]["bodies"][3]
    data["world"]["bodies"] = [dict(rec, id=1000 + k) for k in range(World().capacity + 1)]
    with pytest.raises(save.SaveFormatError):
        save.from_data(data, World())


def test_a_duplicate_id_raises():
    world = busy_world()
    data = save.to_data(world, session(), "x", "Solar")
    data["world"]["bodies"][2]["id"] = data["world"]["bodies"][1]["id"]
    with pytest.raises(save.SaveFormatError):
        save.from_data(data, World())


def test_garbage_raises_rather_than_crashing():
    for garbage in (None, [], {}, {"version": 1}, {"version": 1, "world": 3, "session": {}}):
        with pytest.raises(save.SaveFormatError):
            save.from_data(garbage, World())
