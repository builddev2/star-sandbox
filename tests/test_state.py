import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from src.render.camera import Camera
from src.render.draw import ViewSettings
from src.sim import Simulation


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    pygame.init()
    yield
    pygame.quit()


@pytest.fixture
def state():
    from src.app.state import AppState
    sim = Simulation()
    return AppState(sim, Camera(1000, 700), ViewSettings())


def test_time_ladder_is_monotonic_and_labelled():
    from src.app.state import TIME_LABELS, TIME_SCALES
    assert len(TIME_SCALES) == len(TIME_LABELS)
    assert list(TIME_SCALES) == sorted(TIME_SCALES)
    assert TIME_SCALES[0] < 1.0 < TIME_SCALES[-1]


def test_time_scale_clamps_at_both_ends(state):
    from src.app.state import TIME_SCALES
    for _ in range(50):
        state.nudge_time(-1)
    assert state.days_per_second == TIME_SCALES[0]
    for _ in range(50):
        state.nudge_time(1)
    assert state.days_per_second == TIME_SCALES[-1]


def test_pause_toggles_the_simulation(state):
    assert state.sim.paused is False
    state.toggle_pause()
    assert state.sim.paused is True
    state.toggle_pause()
    assert state.sim.paused is False


def test_selection_survives_bodies_being_removed(state):
    """Selection stores an id, so a merge elsewhere must not silently reselect
    a different planet."""
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    state.select(earth.id)
    state.sim.world.remove(0)          # remove the Sun; rows shuffle
    assert state.sim.world.bodies[state.selected_index].name == "Earth"


def test_selecting_a_removed_body_reports_none(state):
    doomed = state.sim.world.bodies[-1]
    state.select(doomed.id)
    state.sim.world.remove(state.sim.world.index_of(doomed.id))
    assert state.selected_index is None
    assert state.selected_body is None


def test_clicking_on_a_planet_picks_it(state):
    world = state.sim.world
    jupiter = next(b for b in world.bodies if b.name == "Jupiter")
    index = world.index_of(jupiter.id)
    state.camera.centre = world.pos[index].copy()
    sx, sy = state.camera.world_to_screen(world.pos[index])
    assert state.pick_body_at(sx, sy) == index


def test_clicking_empty_space_picks_nothing(state):
    assert state.pick_body_at(3.0, 3.0) is None


def test_clicking_jupiter_does_not_select_one_of_its_moons(state):
    """At system zoom the Galilean moons are a fraction of a pixel from
    Jupiter's centre, so a naive smallest-wins rule makes Jupiter unclickable."""
    world = state.sim.world
    jupiter = world.index_of(next(b for b in world.bodies if b.name == "Jupiter").id)
    state.camera.centre = world.pos[jupiter].copy()
    sx, sy = state.camera.world_to_screen(world.pos[jupiter])
    assert world.bodies[state.pick_body_at(sx, sy)].name == "Jupiter"


def test_a_moon_in_front_of_its_planet_is_still_selectable(state):
    """Picking the smallest match, not the first, is what makes this work."""
    world = state.sim.world
    moon = world.index_of(next(b for b in world.bodies if b.name == "Moon").id)
    state.camera.centre = world.pos[moon].copy()
    state.camera.log_zoom = np.log(1000 / 0.02)   # 0.02 AU across
    sx, sy = state.camera.world_to_screen(world.pos[moon])
    assert world.bodies[state.pick_body_at(sx, sy)].name == "Moon"


def test_cycle_selection_visits_different_bodies(state):
    state.cycle_selection()
    first = state.selected_id
    state.cycle_selection()
    assert state.selected_id != first


def test_follow_selected_sets_the_camera(state):
    mars = next(b for b in state.sim.world.bodies if b.name == "Mars")
    state.select(mars.id)
    state.follow_selected()
    assert state.camera.follow_id == mars.id


def test_delete_selected_removes_it_and_clears_selection(state):
    mars = next(b for b in state.sim.world.bodies if b.name == "Mars")
    before = state.sim.world.n
    state.select(mars.id)
    state.delete_selected()
    assert state.sim.world.n == before - 1
    assert state.selected_id is None
    assert all(b.name != "Mars" for b in state.sim.world.bodies)


def test_stopping_the_earth_makes_it_fall_into_the_sun(state):
    """The single best trick in the toy, so it gets a test."""
    world = state.sim.world
    earth = next(b for b in world.bodies if b.name == "Earth")
    sun = next(b for b in world.bodies if b.name == "Sun")
    state.select(earth.id)
    state.stop_selected()

    def distance():
        return float(np.linalg.norm(world.pos[world.index_of(earth.id)]
                                    - world.pos[world.index_of(sun.id)]))

    start = distance()
    for _ in range(30):
        state.sim.advance(2.0, budget_s=5.0)
    assert distance() < start * 0.9


def test_reset_restores_everything(state):
    state.select(state.sim.world.bodies[2].id)
    state.follow_selected()
    state.sim.advance(500.0, budget_s=5.0)
    state.reset()
    assert state.sim.world.time == 0.0
    assert state.selected_id is None
    assert state.camera.follow_id is None


def test_a_restart_clears_every_lock(state):
    """Restart rebuilds every Body from the scenario, so a lock cannot survive
    it -- the body it belonged to does not.  Right, but only accidentally so
    until something says it out loud: a scenario builder that reused Body
    objects instead of world.clear()-ing them would carry stale locks into a
    world the child asked to have put back."""
    from src import edits
    body = next(b for b in state.sim.world.bodies if b.name == "Earth")
    for field in edits.edit_fields(body):
        edits.toggle_lock(body, field)
    assert body.locked_fields

    state.reset()

    assert all(b.locked_fields == frozenset() for b in state.sim.world.bodies)
    assert all(b is not body for b in state.sim.world.bodies)


def test_elapsed_label_reports_years_and_days(state):
    state.sim.world.time = 400.0
    label = state.elapsed_label
    assert "Year" in label and "Day" in label


def test_hint_falls_back_when_no_tool_is_active(state):
    assert isinstance(state.hint, str) and state.hint


def test_the_inspector_starts_collapsed(state):
    assert state.inspector_expanded is False


def test_selecting_a_new_body_collapses_the_inspector_again(state):
    state.select(state.sim.world.bodies[0].id)
    state.toggle_inspector_expanded()
    assert state.inspector_expanded is True
    state.select(state.sim.world.bodies[1].id)
    assert state.inspector_expanded is False


def test_deselecting_collapses_the_inspector(state):
    state.select(state.sim.world.bodies[0].id)
    state.toggle_inspector_expanded()
    state.select(None)
    assert state.inspector_expanded is False


def test_selecting_stamps_the_selection_time(state):
    state.select(state.sim.world.bodies[0].id, now=1000)
    assert state.selected_at == 1000


def test_reselecting_the_same_body_restamps_the_time(state):
    """The tether line's 3-second countdown has to restart even when the
    click lands on the body that was already selected -- this is what makes
    that possible without watching selected_id for a change that never
    happens."""
    body = state.sim.world.bodies[0]
    state.select(body.id, now=1000)
    state.select(body.id, now=5000)
    assert state.selected_at == 5000


def test_cycling_selection_restamps_the_time(state):
    state.select(state.sim.world.bodies[0].id, now=1000)
    before = pygame.time.get_ticks()
    state.cycle_selection()
    assert state.selected_at >= before


# --- saving, loading, the menus ------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    from src.app.saves import SlotStore
    return SlotStore(str(tmp_path / "saves"))


@pytest.fixture
def saving_state(store):
    from src.app.state import AppState
    from src.app.tools import SelectTool
    st = AppState(Simulation(), Camera(1000, 700), ViewSettings(), store=store)
    st.set_tool(SelectTool)
    return st


def test_the_start_menu_freezes_time(saving_state):
    from src.app.state import DIALOG_START
    st = saving_state
    st.open_start()
    assert st.dialog.kind == DIALOG_START and st.sim.paused
    st.start_new()
    assert st.dialog is None and not st.sim.paused


def test_quit_from_the_start_menu_needs_no_confirmation(saving_state):
    saving_state.open_start()
    saving_state.quit_now()
    assert saving_state.running is False


def test_load_from_the_start_menu_and_back(saving_state):
    from src.app.state import DIALOG_LOAD, DIALOG_START
    st = saving_state
    st.open_start()
    st.open_load()
    assert st.dialog.kind == DIALOG_LOAD
    assert st.dialog.slots == [None, None, None]
    st.close_dialog()
    assert st.dialog.kind == DIALOG_START


def test_opening_the_saves_dialog_pauses_and_closing_restores(saving_state):
    st = saving_state
    for was_paused in (False, True):
        st.sim.paused = was_paused
        st.open_saves()
        assert st.sim.paused is True
        st.close_dialog()
        assert st.dialog is None
        assert st.sim.paused is was_paused


def test_toggle_saves_opens_and_closes(saving_state):
    from src.app.state import DIALOG_SAVES
    st = saving_state
    st.toggle_saves()
    assert st.dialog.kind == DIALOG_SAVES
    st.toggle_saves()
    assert st.dialog is None


def test_saving_into_an_empty_slot_takes_one_press(saving_state, store):
    st = saving_state
    st.open_saves()
    assert st.save_slot(2) is True
    assert store.list()[1].name == st.default_save_name()
    assert st.dialog.slots[1] is not None
    assert st.dialog.armed is None


def test_default_name_is_scenario_and_year(saving_state):
    saving_state.sim.world.time = 3.2 * 365.25
    assert saving_state.default_save_name() == "Solar Year 3.2"


def test_saving_over_a_full_slot_takes_two_presses_and_keeps_the_name(saving_state, store):
    st = saving_state
    st.open_saves()
    st.save_slot(1)
    store.rename(1, "Mars trip")
    st.dialog.slots = store.list()
    st.sim.world.time = 999.0
    assert st.save_slot(1) is False
    assert st.dialog.armed == 1
    assert store.read(1)["world"]["time"] != 999.0
    assert st.save_slot(1) is True
    assert st.dialog.armed is None
    assert store.read(1)["world"]["time"] == 999.0
    assert store.read(1)["name"] == "Mars trip"


def test_anything_else_disarms_an_overwrite(saving_state):
    st = saving_state
    st.open_saves()
    st.save_slot(1)
    st.save_slot(1)
    assert st.dialog.armed == 1
    st.disarm()
    assert st.dialog.armed is None
    st.save_slot(1)
    st.save_slot(3)                       # a different slot is "something else"
    assert st.dialog.armed is None
    assert st.dialog.slots[2] is not None


def test_save_over_an_unreadable_slot_uses_the_default_name(saving_state, store):
    import os
    os.makedirs(store.directory, exist_ok=True)
    with open(store.path(1), "w") as f:
        f.write("junk")
    st = saving_state
    st.open_saves()
    assert not st.dialog.slots[0].readable
    st.save_slot(1)
    assert st.save_slot(1) is True
    assert store.list()[0].name == st.default_save_name()


def test_without_a_store_nothing_is_saved(state):
    state.open_saves()
    assert state.dialog.slots == [None, None, None]
    assert state.save_slot(1) is False
    assert state.dialog.slots == [None, None, None]


def test_loading_restores_the_world_the_selection_and_the_camera(saving_state, store):
    st = saving_state
    world = st.sim.world
    earth_id = next(b.id for b in world.bodies if b.name == "Earth")
    st.select(earth_id)
    st.camera.follow_id = earth_id
    st.camera.centre[:] = (1.0, 2.0)
    st.camera.log_zoom = np.log(1000 / 4.0)          # 4 AU across
    st.time_index = 8
    st.view.show_trails = False
    st.sim.paused = True
    st.open_saves()
    st.save_slot(1)
    st.close_dialog()

    st.reset()
    st.sim.paused = False
    st.time_index = 2
    st.view.show_trails = True
    st.camera.log_zoom = np.log(1000 / 50.0)
    st.sim.world.time = 5000.0
    st.open_saves()
    assert st.load_slot(1) is True
    assert st.dialog is None
    assert st.sim.paused is True
    assert st.time_index == 8 and st.view.show_trails is False
    assert st.selected_id == earth_id and st.camera.follow_id == earth_id
    assert st.selected_body.name == "Earth"
    assert tuple(st.camera.centre) == (1.0, 2.0)
    assert abs(st.camera.width / st.camera.zoom - 4.0) < 1e-9
    assert st.sim.world.time != 5000.0


def test_a_save_made_from_a_running_world_loads_running(saving_state):
    st = saving_state
    st.sim.paused = False
    st.open_saves()                      # pauses mechanically
    st.save_slot(1)
    st.close_dialog()
    st.sim.paused = True
    st.open_saves()
    st.load_slot(1)
    assert st.sim.paused is False, "the dialog's own pause must not be saved"


def test_the_same_save_shows_the_same_width_of_sky_in_a_narrower_window(saving_state, store):
    from src.app.state import AppState
    st = saving_state
    st.camera.log_zoom = np.log(1000 / 4.0)
    st.open_saves()
    st.save_slot(1)
    narrow = AppState(Simulation(), Camera(400, 300), ViewSettings(), store=store)
    narrow.open_saves()
    assert narrow.load_slot(1)
    assert abs(narrow.camera.width / narrow.camera.zoom - 4.0) < 1e-9


def test_loading_refreshes_the_integrator_and_drops_the_stale_report(saving_state):
    st = saving_state
    st.open_saves()
    st.save_slot(1)
    st.explain.report = object()
    st.load_slot(1)
    assert st.explain.report is None
    world = st.sim.world
    # World.add zeroes acc for every body it adds, and nothing else fills it
    # back in on its own -- only Simulation.refresh() (-> Integrator.refresh)
    # recomputes it.  A skipped refresh does not raise: the first substep
    # would just half-kick every body from a silently wrong (zero) force,
    # so the only way to catch it is to look at world.acc directly.
    assert np.abs(world.acc[:world.n]).max() > 0


def test_a_slot_that_cannot_be_loaded_is_marked_and_the_world_survives(saving_state, store):
    import os
    st = saving_state
    st.open_saves()
    st.save_slot(1)
    data = store.read(1)
    data["world"]["bodies"][0]["kind"] = "teapot"
    store.write(1, data)
    st.dialog.slots = store.list()           # still lists as readable: header is fine
    n = st.sim.world.n
    assert st.load_slot(1) is False
    assert st.dialog is not None
    assert not st.dialog.slots[0].readable
    assert st.sim.world.n == n


def test_rename_commit_writes_and_cancel_does_not(saving_state, store):
    st = saving_state
    st.open_saves()
    st.save_slot(2)
    st.begin_rename(2)
    assert st.dialog.renaming == 2 and st.dialog.text == st.default_save_name()
    for ch in "!!":
        st.rename_type(ch)
    st.rename_backspace()
    st.rename_commit()
    assert st.dialog.renaming is None
    assert store.list()[1].name == st.default_save_name() + "!"
    st.begin_rename(2)
    st.rename_type("z")
    st.rename_cancel()
    assert store.list()[1].name == st.default_save_name() + "!"


def test_an_empty_name_keeps_the_old_one(saving_state, store):
    st = saving_state
    st.open_saves()
    st.save_slot(1)
    st.begin_rename(1)
    while st.dialog.text:
        st.rename_backspace()
    st.rename_type(" ")
    st.rename_commit()
    assert store.list()[0].name == st.default_save_name()


def test_names_are_capped_and_printable_ascii_only(saving_state):
    from src.app.state import NAME_MAX
    st = saving_state
    st.open_saves()
    st.save_slot(1)
    st.begin_rename(1)
    while st.dialog.text:
        st.rename_backspace()
    for _ in range(NAME_MAX + 10):
        st.rename_type("a")
    assert len(st.dialog.text) == NAME_MAX
    st.rename_backspace()
    st.rename_type("\t")
    st.rename_type("é")
    st.rename_type("\x1b")
    assert st.dialog.text == "a" * (NAME_MAX - 1)


def test_rename_and_arm_cancel_each_other(saving_state):
    st = saving_state
    st.open_saves()
    st.save_slot(1)
    st.save_slot(1)                          # armed
    st.begin_rename(1)
    assert st.dialog.armed is None and st.dialog.renaming == 1
    st.save_slot(1)                          # arms again, and leaves the field
    assert st.dialog.renaming is None and st.dialog.armed == 1


def test_describe_slot_reads_well(saving_state, store):
    from src.app.saves import SlotInfo, UNREADABLE_NAME
    st = saving_state
    assert st.describe_slot(None) == "Nothing saved here yet"
    bad = SlotInfo(1, UNREADABLE_NAME, "", {}, False)
    assert "can't be read" in st.describe_slot(bad)
    good = SlotInfo(1, "x", "2026-09-12T14:03",
                    {"scenario_label": "Solar", "days": 3.2 * 365.25, "bodies": 9}, True)
    assert st.describe_slot(good) == "Solar  Year 3.2  9 bodies  Sat 12 Sep 14:03"
    odd = SlotInfo(1, "x", "yesterday", {"scenario_label": "Solar", "days": 0, "bodies": 1}, True)
    assert st.describe_slot(odd).endswith("yesterday")


def test_restart_closes_nothing_but_loading_resets_the_tool(saving_state):
    from src.app.tools import FlyTool
    st = saving_state
    st.set_tool(FlyTool)
    st.open_saves()
    st.save_slot(1)
    old_tool = st.tool
    st.load_slot(1)
    assert isinstance(st.tool, FlyTool) and st.tool is not old_tool
