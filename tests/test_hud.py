import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import pytest


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


def test_escape_velocity_uses_grouped_digits_for_the_speed_of_light():
    """format_speed must never use exponent notation, per the file's principle.

    The speed of light (299,792 km/s) is a black hole's escape velocity and
    must render with grouped digits, not as 2.998e+05.
    """
    from src import units
    from src.render.hud import format_speed
    c_au_day = units.C  # speed of light in AU/day
    text = format_speed(c_au_day)
    assert "299,792" in text or "299793" in text, \
        f"Speed of light should be grouped: {text}"
    assert "e+" not in text and "e-" not in text, \
        f"Speed of light must not use exponent notation: {text}"


def test_hawking_temperature_of_a_ten_solar_mass_hole_is_unchanged():
    """Fixing format_temperature must not change output for the ordinary range.

    A 10-solar-mass hole (common test fixture) has T = 6.169e-9 K, which
    should still render as 6.17 billionths of a K after the fix.
    """
    from src.render.hud import format_temperature
    temp_10_sun = 6.169e-9
    text = format_temperature(temp_10_sun)
    assert "billionths" in text, f"Should show billionths for 6.169e-9 K: {text}"
    assert "e+" not in text and "e-" not in text, \
        f"No exponent notation allowed: {text}"
    # The number should be around 6.17 billionths
    assert "6.17" in text or "6.2" in text, \
        f"Should show approximately 6.17 billionths: {text}"


# --- body temperature ---------------------------------------------------------

def test_temperature_is_shown_in_kelvin_with_celsius_alongside():
    """Kelvin because that is the unit the physics is in; Celsius because that
    is the unit the audience has."""
    from src.render.hud import format_body_temperature
    assert format_body_temperature(273.15) == "273 K  (0 C)"
    assert format_body_temperature(254.0) == "254 K  (-19 C)"


def test_a_hot_temperature_uses_grouped_digits_not_an_exponent():
    from src.render.hud import format_body_temperature
    text = format_body_temperature(5772.0)
    assert "e+" not in text
    assert "5,772 K" in text


def test_the_inspector_shows_a_temperature_for_a_planet(state):
    from src.render.hud import Hud
    from src.render.pixelfont import PixelFont
    state.select(next(b for b in state.sim.world.bodies if b.name == "Earth").id)
    rows = Hud(PixelFont(10))._rows(state, state.selected_index)
    assert any(text.startswith("Temp") for text, _ in rows)


def test_a_star_shows_the_surface_temperature_its_mass_gives_it(state):
    from src import units
    from src.render.hud import Hud
    from src.render.pixelfont import PixelFont
    state.select(next(b for b in state.sim.world.bodies if b.name == "Sun").id)
    rows = Hud(PixelFont(10))._rows(state, state.selected_index)
    temp_row = next(text for text, _ in rows if text.startswith("Temp"))
    assert "{:,.0f} K".format(units.star_temperature(1.0)) in temp_row


def test_headline_stats_for_a_planet_are_mass_and_temperature(state):
    from src.render.hud import Hud
    from src.render.pixelfont import PixelFont
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    state.select(earth.id)
    stats = Hud(PixelFont(14)).headline_stats(state.sim.world, state.selected_index)
    labels = [label for label, _ in stats]
    assert labels == ["Mass", "Temp"]


def test_headline_stats_for_a_star_are_mass_only(state):
    from src.render.hud import Hud
    from src.render.pixelfont import PixelFont
    sun = next(b for b in state.sim.world.bodies if b.name == "Sun")
    state.select(sun.id)
    stats = Hud(PixelFont(14)).headline_stats(state.sim.world, state.selected_index)
    assert [label for label, _ in stats] == ["Mass"]


def test_headline_stats_for_a_black_hole_are_mass_and_horizon(state):
    from src import edits
    from src.render.hud import Hud
    from src.render.pixelfont import PixelFont
    sun = next(b for b in state.sim.world.bodies if b.name == "Sun")
    state.select(sun.id)
    edits.collapse(state.sim.world, state.selected_index)
    stats = Hud(PixelFont(14)).headline_stats(state.sim.world, state.selected_index)
    assert [label for label, _ in stats] == ["Mass", "Horizon"]


def test_a_collapsed_inspector_shows_far_fewer_rows_than_expanded(state):
    from src.render.hud import Hud
    from src.render.pixelfont import PixelFont
    hud = Hud(PixelFont(14))
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    state.select(earth.id)
    state.inspector_expanded = False
    collapsed = hud.visible_rows(state, state.selected_index, 700)
    state.inspector_expanded = True
    expanded = hud.visible_rows(state, state.selected_index, 700)
    assert len(collapsed) < len(expanded)


def test_the_inspector_rect_is_shorter_collapsed_than_expanded(state):
    from src.render.hud import Hud
    from src.render.pixelfont import PixelFont
    hud = Hud(PixelFont(14))
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    state.select(earth.id)
    state.inspector_expanded = False
    collapsed = hud.inspector_rect(state, 1000, 700)
    state.inspector_expanded = True
    expanded = hud.inspector_rect(state, 1000, 700)
    assert collapsed.height < expanded.height
    # Both still reserve the same fixed actions band above them, so they start
    # at the same top -- only the bottom moves.
    assert collapsed.top == expanded.top


def test_bar_specs_include_the_ignition_row_for_a_light_planet(state):
    from src.render.hud import Hud
    from src.render.pixelfont import PixelFont
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    state.select(earth.id)
    hud = Hud(PixelFont(14))
    rows = hud._rows(state, state.selected_index)
    specs = hud.bar_specs(state.sim.world, state.selected_index, rows)
    ignition_i = next(i for i, (text, _) in enumerate(rows)
                      if "heavier to ignite" in text)
    assert any(i == ignition_i for i, _frac, _colour in specs)


def test_bar_specs_include_the_temperature_row(state):
    from src.render.hud import Hud
    from src.render.pixelfont import PixelFont
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    state.select(earth.id)
    hud = Hud(PixelFont(14))
    rows = hud._rows(state, state.selected_index)
    specs = hud.bar_specs(state.sim.world, state.selected_index, rows)
    temp_i = next(i for i, (text, _) in enumerate(rows) if text.startswith("Temp"))
    assert any(i == temp_i for i, _frac, _colour in specs)
    frac = next(f for i, f, _c in specs if i == temp_i)
    assert 0.0 <= frac <= 1.0


def test_a_star_has_no_ignition_bar(state):
    from src.render.hud import Hud
    from src.render.pixelfont import PixelFont
    sun = next(b for b in state.sim.world.bodies if b.name == "Sun")
    state.select(sun.id)
    hud = Hud(PixelFont(14))
    rows = hud._rows(state, state.selected_index)
    specs = hud.bar_specs(state.sim.world, state.selected_index, rows)
    assert not any("heavier to ignite" in rows[i][0] for i, _f, _c in specs)


def test_the_material_blurb_is_the_dimmest_row(state):
    from src.render import palette
    from src.render.hud import Hud
    from src.render.pixelfont import PixelFont
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    state.select(earth.id)
    rows = Hud(PixelFont(14))._rows(state, state.selected_index)
    text, colour = rows[-1]
    assert text == earth.material.blurb
    assert colour == palette.TEXT_FAINT


def test_the_collapsed_card_draws_a_raised_background_band(state):
    from src.render import palette
    from src.render.hud import Hud
    from src.render.pixelfont import PixelFont
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    state.select(earth.id)
    state.inspector_expanded = False
    hud = Hud(PixelFont(14))
    surface = pygame.Surface((1000, 700))
    surface.fill(palette.SPACE)
    hud.draw(surface, state)
    rect = hud.inspector_rect(state, 1000, 700)
    band = [surface.get_at((x, rect.centery))[:3]
           for x in range(rect.left, rect.right)]
    assert palette.PANEL_ROW in band


def test_the_temperature_row_fits_the_inspector_panel():
    """The panel is a fixed number of characters wide; a row that overruns it
    is a number nobody can read."""
    from src import units
    from src.render.hud import PANEL_WIDTH_CHARS, format_body_temperature
    hottest = "Temp       %s" % format_body_temperature(units.MAX_TEMPERATURE_K)
    assert len(hottest) <= PANEL_WIDTH_CHARS


def test_light_time_reads_in_the_unit_a_book_uses():
    from src import units
    from src.render.hud import format_light_time
    assert format_light_time(1.0 / units.C) == "8.3 min"          # the Sun
    assert format_light_time(30.07 / units.C) == "4.2 hr"         # Neptune
    assert format_light_time(0.00257 / units.C) == "1.3 s"        # the Moon
    assert format_light_time(2.0) == "2.0 days"


def test_a_period_is_four_figures_in_a_sensible_unit():
    from src.render.hud import format_period
    assert format_period(365.2564) == "365.3 d"
    assert format_period(87.969) == "87.97 d"
    assert format_period(0.5) == "12 hr"
    assert format_period(60190.0) == "164.8 yr"
