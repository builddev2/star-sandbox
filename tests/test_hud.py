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
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.sim import Simulation
    return AppState(Simulation(), Camera(1000, 700), ViewSettings())


def test_mass_is_shown_in_whichever_unit_is_readable():
    from src import units
    from src.render.hud import format_mass
    assert "Earth" in format_mass(units.EARTH_MASS)
    assert "Jupiter" in format_mass(units.JUPITER_MASS)
    assert "Sun" in format_mass(1.0)
    assert "kg" in format_mass(1e-20)


def test_length_switches_between_km_and_au():
    from src import units
    from src.render.hud import format_length
    assert "km" in format_length(units.EARTH_RADIUS)
    assert "AU" in format_length(5.2)


def test_sizes_never_use_scientific_notation():
    """"5.823e+04 km" is unreadable to an eight-year-old."""
    from src import units
    from src.render.hud import format_length
    for radius_km in (58232.0, 695700.0, 6371.0, 1737.4, 50.0):
        text = format_length(radius_km / units.AU_KM)
        assert "e+" not in text and "e-" not in text, text
    assert format_length(58232.0 / units.AU_KM) == "58,232 km"
    assert format_length(695700.0 / units.AU_KM) == "695,700 km"


def test_speed_is_in_km_per_second():
    from src import units
    from src.render.hud import format_speed
    text = format_speed(30.0 / units.AUD_TO_KMS)
    assert "km/s" in text and "30" in text


def test_orbit_facts_name_the_right_attractor(state):
    from src.render.hud import orbit_facts
    world = state.sim.world
    names = [b.name for b in world.bodies]
    facts = orbit_facts(world, names.index("Moon"))
    assert facts["attractor"] == "Earth"


def test_orbit_facts_get_earths_year_right(state):
    from src.render.hud import orbit_facts
    world = state.sim.world
    names = [b.name for b in world.bodies]
    facts = orbit_facts(world, names.index("Earth"))
    assert facts["a"] == pytest.approx(1.0, rel=0.02)
    # Under two years reads in days, which is the more concrete number for a
    # child: "Earth takes 365 days" beats "Earth takes 1.0 yr".
    assert facts["period_label"] == "365.3 days"


def test_outer_planets_report_their_period_in_years(state):
    from src.render.hud import orbit_facts
    world = state.sim.world
    names = [b.name for b in world.bodies]
    for planet, years in (("Jupiter", 11.86), ("Saturn", 29.46), ("Neptune", 164.79)):
        label = orbit_facts(world, names.index(planet))["period_label"]
        assert label.endswith(" yr"), label
        assert float(label[:-3]) == pytest.approx(years, rel=0.005), \
            "%s reads %s, should be about %.2f yr" % (planet, label, years)


def test_the_moon_reports_its_month(state):
    from src.render.hud import orbit_facts
    world = state.sim.world
    names = [b.name for b in world.bodies]
    label = orbit_facts(world, names.index("Moon"))["period_label"]
    assert label.endswith("days")
    assert 27.0 <= float(label.split()[0]) <= 28.0, label


def test_orbit_facts_report_an_escaping_body(state):
    """A rock thrown past escape velocity must not claim to have a period."""
    from src import materials, units
    from src.render.hud import orbit_facts
    world = state.sim.world
    fast = np.array([0.0, np.sqrt(2 * units.G / 3.0) * 1.6])
    body = world.add("Runaway", materials.get("rock"), 1e-12,
                     units.EARTH_RADIUS * 0.1, np.array([3.0, 0.0]), fast)
    facts = orbit_facts(world, world.index_of(body.id))
    assert facts["period_label"] == "escaping"
    assert facts["apoapsis"] is None


def test_orbit_facts_of_a_lone_body_are_none():
    from src import materials, units
    from src.render.hud import orbit_facts
    from src.world import World
    world = World()
    world.add("lonely", materials.get("rock"), units.EARTH_MASS,
              units.EARTH_RADIUS, np.zeros(2), np.zeros(2))
    assert orbit_facts(world, 0) is None


def test_hud_draws_without_a_selection(state):
    from src.render import palette
    from src.render.hud import Hud
    from src.render.pixelfont import PixelFont
    surface = pygame.Surface((1000, 700))
    surface.fill(palette.SPACE)
    Hud(PixelFont(14)).draw(surface, state)
    assert pygame.transform.average_color(surface)[:3] != palette.SPACE[:3]


def test_hud_draws_the_inspector_when_something_is_selected(state):
    from src.render import palette
    from src.render.hud import Hud
    from src.render.pixelfont import PixelFont
    hud = Hud(PixelFont(14))

    plain = pygame.Surface((1000, 700))
    plain.fill(palette.SPACE)
    hud.draw(plain, state)

    state.select(next(b for b in state.sim.world.bodies if b.name == "Saturn").id)
    with_panel = pygame.Surface((1000, 700))
    with_panel.fill(palette.SPACE)
    hud.draw(with_panel, state)

    assert pygame.image.tostring(plain, "RGB") != pygame.image.tostring(with_panel, "RGB")


def test_inspector_survives_every_body_in_the_scenario(state):
    """Stars have no ignition row, the Sun has no attractor, moons have both."""
    from src.render.hud import Hud
    from src.render.pixelfont import PixelFont
    hud = Hud(PixelFont(14))
    for body in list(state.sim.world.bodies):
        state.select(body.id)
        assert hud._rows(state, state.selected_index)


def test_hud_survives_the_selected_body_being_destroyed(state):
    from src.render.hud import Hud
    from src.render.pixelfont import PixelFont
    doomed = state.sim.world.bodies[-1]
    state.select(doomed.id)
    state.sim.world.remove(state.sim.world.index_of(doomed.id))
    Hud(PixelFont(14)).draw(pygame.Surface((1000, 700)), state)


def test_help_overlay_lists_every_binding():
    from src.render.hud import KEY_HELP, Hud
    from src.render.pixelfont import PixelFont
    assert len(KEY_HELP) >= 12
    surface = pygame.Surface((1000, 700))
    Hud(PixelFont(14)).draw_help(surface)


def _labelled(rows, label):
    """The single inspector row starting with ``label``, or None."""
    for text, _colour in rows:
        if text.startswith(label):
            return text
    return None


def test_anything_that_orbits_something_reports_a_farthest_distance(state):
    """Every body the inspector says orbits another must show how far it gets.

    "Closest" without "Farthest" reads as a missing number rather than as an
    unbounded orbit, and there is no body in the scenario for which the pair
    should be split up.
    """
    from src.render.hud import Hud, orbit_facts
    from src.render.pixelfont import PixelFont
    hud = Hud(PixelFont(14))
    world = state.sim.world
    checked = 0
    for body in list(world.bodies):
        state.select(body.id)
        index = state.selected_index
        rows = hud._rows(state, index)
        if orbit_facts(world, index) is None:
            assert _labelled(rows, "Farthest") is None, body.name
            continue
        checked += 1
        assert _labelled(rows, "Farthest") is not None, body.name
    assert checked > 0


def test_an_escaping_body_still_says_how_far_it_goes(state):
    """A hyperbolic body is still shown as orbiting something, so it still
    needs the row -- and the honest answer is that it never comes back."""
    from src import materials, units
    from src.render.hud import Hud
    from src.render.pixelfont import PixelFont
    world = state.sim.world
    fast = np.array([0.0, np.sqrt(2 * units.G / 3.0) * 1.6])
    body = world.add("Runaway", materials.get("rock"), 1e-12,
                     units.EARTH_RADIUS * 0.1, np.array([3.0, 0.0]), fast)
    state.select(body.id)
    rows = Hud(PixelFont(14))._rows(state, state.selected_index)

    assert _labelled(rows, "Orbits") is not None
    farthest = _labelled(rows, "Farthest")
    assert farthest is not None
    assert "never" in farthest


def test_an_escaping_body_does_not_report_a_negative_distance(state):
    """Semi-major axis is negative on a hyperbola; printing it gave
    "Distance   -1.44e+11 m", which is both wrong and in exponent notation."""
    from src import materials, units
    from src.render.hud import Hud
    from src.render.pixelfont import PixelFont
    world = state.sim.world
    fast = np.array([0.0, np.sqrt(2 * units.G / 3.0) * 1.6])
    body = world.add("Runaway", materials.get("rock"), 1e-12,
                     units.EARTH_RADIUS * 0.1, np.array([3.0, 0.0]), fast)
    state.select(body.id)
    rows = Hud(PixelFont(14))._rows(state, state.selected_index)

    distance = _labelled(rows, "Distance")
    assert "-" not in distance, distance
    assert "e+" not in distance and "e-" not in distance, distance


def test_the_farthest_row_fits_the_inspector_panel(state):
    """The panel is a fixed character width; a row wider than it is clipped."""
    from src import materials, units
    from src.render.hud import Hud, PANEL_WIDTH_CHARS, orbit_facts
    from src.render.pixelfont import PixelFont
    world = state.sim.world
    fast = np.array([0.0, np.sqrt(2 * units.G / 3.0) * 1.6])
    world.add("Runaway", materials.get("rock"), 1e-12,
              units.EARTH_RADIUS * 0.1, np.array([3.0, 0.0]), fast)
    hud = Hud(PixelFont(14))
    for body in list(world.bodies):
        state.select(body.id)
        if orbit_facts(world, state.selected_index) is None:
            continue
        row = _labelled(hud._rows(state, state.selected_index), "Farthest")
        assert len(row) <= PANEL_WIDTH_CHARS, row
