"""Black hole rendering: visual radii, the sprite, and the swallow effect."""

import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    pygame.init()
    pygame.display.set_mode((320, 200))
    yield
    pygame.quit()


@pytest.fixture
def hole_world():
    """A 10 solar mass black hole and the Earth, for comparison."""
    from src import edits, materials, units
    from src.body import KIND_STAR
    from src.world import World
    w = World()
    w.add("Hole", materials.get("hydrogen"), 10.0, units.SUN_RADIUS,
          np.zeros(2), np.zeros(2), kind=KIND_STAR)
    edits.collapse(w, 0)
    w.add("Earth", materials.get("rock"), units.EARTH_MASS, units.EARTH_RADIUS,
          np.array([1.0, 0.0]), np.zeros(2))
    return w


# --- visual radii ---------------------------------------------------------------

def test_visual_radii_replaces_only_the_black_hole(hole_world):
    from src import units
    from src.render import scale
    visual = scale.visual_radii(hole_world)
    assert visual[0] == pytest.approx(units.apparent_radius(10.0))
    assert visual[1] == pytest.approx(units.EARTH_RADIUS)


def test_visual_radii_does_not_mutate_the_world(hole_world):
    from src import units
    from src.render import scale
    scale.visual_radii(hole_world)
    assert float(hole_world.radius[0]) == pytest.approx(
        units.schwarzschild_radius(10.0))


def test_without_the_floor_fix_a_black_hole_draws_as_one_pixel(hole_world):
    """The problem this task exists to solve, pinned so it cannot come back."""
    from src.render import scale
    naive = scale.display_radii_px(hole_world.radii, zoom=64.0,
                                   internal_height=360.0)
    assert naive[0] == pytest.approx(1.0)


def test_with_the_floor_fix_a_black_hole_draws_as_large_as_a_star(hole_world):
    from src.render import scale
    fixed = scale.display_radii_px(hole_world.radii, zoom=64.0,
                                   internal_height=360.0,
                                   floor_radii=scale.visual_radii(hole_world))
    assert fixed[0] == pytest.approx(scale.FLOOR_MAX)


def test_real_size_mode_collapses_a_black_hole_to_a_dot(hole_world):
    """Pressing R must still show the honest truth: the horizon is nothing."""
    from src.render import scale
    real = scale.display_radii_px(hole_world.radii, zoom=64.0,
                                  internal_height=360.0, real_size=True,
                                  floor_radii=scale.visual_radii(hole_world))
    assert real[0] == pytest.approx(1.0)


def test_the_floor_argument_leaves_ordinary_bodies_alone(hole_world):
    from src.render import scale
    plain = scale.display_radii_px(hole_world.radii, 64.0, 360.0)
    fixed = scale.display_radii_px(hole_world.radii, 64.0, 360.0,
                                   floor_radii=scale.visual_radii(hole_world))
    assert fixed[1] == pytest.approx(plain[1])


# --- aiming ---------------------------------------------------------------------

def test_without_the_aim_fix_a_black_hole_is_impossible_to_hit(hole_world):
    """8x the true horizon is 1.6e-6 AU: about 0.0001 pixels at system zoom."""
    from src.render import scale
    naive = scale.contact_radii_au(hole_world.radii, zoom=64.0,
                                   internal_height=360.0)
    assert naive[0] * 64.0 < 0.001          # sub-thousandth of a pixel


def test_with_the_aim_fix_a_black_hole_is_hittable(hole_world):
    from src.render import scale
    fixed = scale.contact_radii_au(hole_world.radii, zoom=64.0,
                                   internal_height=360.0,
                                   aim_radii=scale.visual_radii(hole_world))
    assert fixed[0] * 64.0 >= scale.FLOOR_MAX - 1e-9


def test_contact_radius_never_falls_below_the_true_radius(hole_world):
    from src.render import scale
    for zoom in (1e-3, 1.0, 64.0, 1e9):
        fixed = scale.contact_radii_au(hole_world.radii, zoom, 360.0,
                                       aim_radii=scale.visual_radii(hole_world))
        assert np.all(fixed >= hole_world.radii - 1e-30)


def test_the_aim_argument_leaves_ordinary_bodies_alone(hole_world):
    from src.render import scale
    plain = scale.contact_radii_au(hole_world.radii, 64.0, 360.0)
    fixed = scale.contact_radii_au(hole_world.radii, 64.0, 360.0,
                                   aim_radii=scale.visual_radii(hole_world))
    assert fixed[1] == pytest.approx(plain[1])


# --- the sprite -----------------------------------------------------------------

@pytest.fixture
def cache():
    from src.render.sprites import SpriteCache
    return SpriteCache(limit=64)


def _centre(surf):
    return surf.get_width() // 2, surf.get_height() // 2


def test_the_centre_is_opaque_black(cache):
    """The one thing a black hole must get right."""
    surf = cache.get_black_hole(seed=1, px=12, spin=0.5, disk=0.8)
    cx, cy = _centre(surf)
    r, g, b, a = surf.get_at((cx, cy))
    assert (r, g, b) == (0, 0, 0)
    assert a == 255


def test_a_bright_photon_ring_sits_at_the_shadow_edge(cache):
    px = 12
    surf = cache.get_black_hole(seed=1, px=px, spin=0.0, disk=0.0)
    cx, cy = _centre(surf)
    # Straight up from the centre: dark inside, bright at px.
    inside = surf.get_at((cx, cy - px + 3))
    rim = max(sum(surf.get_at((cx, cy - px + d))[:3]) for d in (-1, 0, 1))
    assert sum(inside[:3]) < 30
    assert rim > 300


def test_a_starved_hole_draws_no_disk(cache):
    """disk=0 is a real state: a hole with nothing left to eat."""
    px = 12
    surf = cache.get_black_hole(seed=1, px=px, spin=0.9, disk=0.0)
    cx, cy = _centre(surf)
    # Well outside the photon ring, along the disk plane.
    far = surf.get_at((cx + int(px * 2.2), cy))
    assert far[3] == 0


def test_a_feeding_hole_draws_a_disk_along_the_horizontal(cache):
    px = 12
    surf = cache.get_black_hole(seed=1, px=px, spin=0.0, disk=1.0)
    cx, cy = _centre(surf)
    band = [surf.get_at((cx + int(px * 2.0), cy + dy))[3] for dy in (-1, 0, 1)]
    assert max(band) > 0


def test_the_disk_is_squashed_not_circular(cache):
    """Near-edge-on: much wider than it is tall."""
    px = 12
    surf = cache.get_black_hole(seed=1, px=px, spin=0.0, disk=1.0)
    cx, cy = _centre(surf)
    out = int(px * 2.4)
    along = surf.get_at((cx + out, cy))[3]
    across = surf.get_at((cx, cy + out))[3]
    assert along > across


def test_spin_pulls_the_inner_edge_of_the_disk_inward(cache):
    """The ISCO runs from 3 horizon radii to 0.62 as spin rises.  Exaggerated
    for legibility -- at 6px the true range is under one pixel -- but the
    ordering and direction are honest."""
    from src.render import sprites
    still = sprites._inner_disk_px(20, spin=0.0)
    spun = sprites._inner_disk_px(20, spin=0.998)
    assert spun < still
    assert still == pytest.approx(20 * sprites.BH_INNER_MAX)
    assert spun == pytest.approx(20 * sprites.BH_INNER_MIN, rel=1e-3)


def test_doppler_makes_one_side_of_the_disk_brighter(cache):
    px = 14
    surf = cache.get_black_hole(seed=0, px=px, spin=0.0, disk=1.0)
    cx, cy = _centre(surf)
    out = int(px * 2.0)
    left = sum(surf.get_at((cx - out, cy))[:3])
    right = sum(surf.get_at((cx + out, cy))[:3])
    assert left != right


def test_the_doppler_side_is_stable_for_a_given_seed(cache):
    """It comes from the body's seed, so a hole never flips direction when
    the sprite cache evicts it."""
    a = cache.get_black_hole(seed=7, px=14, spin=0.0, disk=1.0)
    cache.clear()
    b = cache.get_black_hole(seed=7, px=14, spin=0.0, disk=1.0)
    cx, cy = _centre(a)
    out = 28
    assert a.get_at((cx - out, cy)) == b.get_at((cx - out, cy))


def test_a_spinning_feeding_hole_grows_jets(cache):
    px = 14
    spun = cache.get_black_hole(seed=1, px=px, spin=0.9, disk=1.0)
    still = cache.get_black_hole(seed=1, px=px, spin=0.0, disk=1.0)
    assert spun.get_height() > still.get_height()

    # The canvas grows from the jet-length arithmetic alone, so height proves
    # nothing about whether a jet was painted.  Sample the axis itself: the
    # root is hidden inside the shadow, and the beam emerges past the ring.
    cx, cy = _centre(spun)
    assert spun.get_at((cx, cy - px + 2))[:3] == (0, 0, 0)
    lit = [spun.get_at((cx, cy - d))[3] for d in range(px + 1, px + 12)]
    assert max(lit) > 0, "spin and disk are on, but no jet pixel was drawn"


def test_a_tiny_hole_is_still_a_dark_dot_with_a_rim(cache):
    """At system zoom a hole draws at 6px; below that it must not vanish."""
    surf = cache.get_black_hole(seed=1, px=2, spin=0.5, disk=0.5)
    cx, cy = _centre(surf)
    assert surf.get_at((cx, cy))[:3] == (0, 0, 0)
    assert surf.get_at((cx, cy))[3] == 255


def test_the_cache_distinguishes_spin_and_disk(cache):
    a = cache.get_black_hole(seed=1, px=10, spin=0.0, disk=1.0)
    b = cache.get_black_hole(seed=1, px=10, spin=0.9, disk=1.0)
    c = cache.get_black_hole(seed=1, px=10, spin=0.0, disk=0.2)
    assert a is not b
    assert a is not c
    assert cache.get_black_hole(seed=1, px=10, spin=0.0, disk=1.0) is a


# --- wiring ---------------------------------------------------------------------

def _draw(world, zoom=64.0, size=(320, 200)):
    from src.render.camera import Camera
    from src.render.draw import ViewSettings, WorldRenderer
    from src.render.pixelfont import PixelFont
    from src.render.sprites import SpriteCache
    from src.sim import Simulation

    surface = pygame.Surface(size)
    surface.fill((40, 0, 60))          # not black: the shadow must be drawn,
                                       # not merely left as background
    au_across = size[0] / zoom
    camera = Camera(size[0], size[1], au_across=au_across)
    sim = Simulation(world=world)
    renderer = WorldRenderer(SpriteCache(limit=64), PixelFont(14))
    renderer.draw(surface, sim, camera, ViewSettings(show_trails=False,
                                                     show_labels=False))
    return surface, camera


def test_a_black_hole_is_drawn_as_a_dark_disc_not_a_bright_one(hole_world):
    """The star path blits additively; a hole must not, or it draws white."""
    hole_world.pos[0] = (0.0, 0.0)
    surface, camera = _draw(hole_world)
    x, y = camera.world_to_screen(hole_world.pos[0])
    assert surface.get_at((int(x), int(y)))[:3] == (0, 0, 0)


def test_a_black_hole_is_drawn_large_enough_to_see(hole_world):
    """The opaque black shadow is six pixels across, not one."""
    hole_world.pos[0] = (0.0, 0.0)
    surface, camera = _draw(hole_world)
    x, y = camera.world_to_screen(hole_world.pos[0])
    # Count the run of opaque black pixels (0,0,0) through the centre.
    # Without the fix the hole draws at px<2.0 as a single body.material.tones[2]
    # pixel (hot white for the singularity material), not black.
    # The disk halo is a separate effect; measure only the shadow.
    black_run = [dx for dx in range(-12, 13)
                 if surface.get_at((int(x) + dx, int(y)))[:3] == (0, 0, 0)]
    assert black_run, "the opaque black shadow was not drawn"
    assert max(abs(dx) for dx in black_run) >= 2


def test_clicking_where_a_black_hole_appears_selects_it(hole_world):
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.app.state import AppState
    from src.sim import Simulation
    hole_world.pos[0] = (0.0, 0.0)
    camera = Camera(1280, 720, au_across=20.0)  # zoom = 1280 / 20.0 = 64.0
    state = AppState(Simulation(world=hole_world), camera, ViewSettings())
    x, y = camera.world_to_screen(hole_world.pos[0])
    # Fifteen pixels off centre: inside the drawn disc with the fix (~12px
    # radius + 8px margin), outside the unfixed one (~2px + 8px).
    # This discriminates whether the visual radius is being used.
    assert state.pick_body_at(int(x) + 15, int(y)) == 0


def test_the_contact_refresh_makes_a_black_hole_hittable():
    """App.step_once() refreshes contact_radius with visual_radii as aim_radii.

    This tests the actual line in main.py:98-101 that wires aim_radii into the
    frame loop. Without this line, the contact radius stays at the true (useless)
    value and the black hole cannot be hit by a thrown projectile.
    """
    import os
    from src.app.main import App

    app = App(fullscreen=False)
    try:
        world = app.state.sim.world
        # Collapse the first body (Earth in the default scenario).
        from src import edits
        edits.collapse(world, 0)
        initial_contact = float(world.contact_radius[0])

        # One frame of the app.
        app.step_once()

        # The contact refresh in step_once should have inflated contact_radius
        # to match the visual radius so a projectile can hit it.
        new_contact = float(world.contact_radius[0])
        true_radius = float(world.radius[0])
        assert new_contact > initial_contact, \
            "contact refresh did not enlarge the contact radius"
        assert new_contact > true_radius * 1000.0, \
            "visual radius basis was not applied; contact radius is still tiny"
    finally:
        pygame.quit()


# --- the swallow effect ---------------------------------------------------------

def _swallow_event(energy=1e-6):
    from src.collisions import SWALLOW, ImpactEvent
    from src import units
    return ImpactEvent(
        pos=(0.0, 0.0), vel=(0.0, 0.0), energy=energy, outcome=SWALLOW,
        target_radius=units.schwarzschild_radius(10.0),
        projectile_radius=units.EARTH_RADIUS,
        tones=((0x40, 0x08, 0x04), (0xE0, 0x70, 0x18), (0xFF, 0xF0, 0xD0)),
        on_star=False,
    )


def test_a_swallow_spawns_particles():
    from src.render.effects import EffectField
    field = EffectField(seed=1)
    field.spawn(_swallow_event())
    assert field.count > 0


def test_a_swallow_collapses_inward_rather_than_exploding_outward():
    """Every other effect grows.  This one is the only one that shrinks."""
    from src.render.effects import EffectField
    field = EffectField(seed=1)
    field.spawn(_swallow_event())
    rings = [p for p in field.particles if p.kind == "ring"]
    assert rings
    assert all(p.r1 < p.r0 for p in rings)


def test_a_swallow_throws_no_debris_outward():
    """Nothing comes back out of a horizon."""
    from src.render.effects import EffectField
    field = EffectField(seed=1)
    field.spawn(_swallow_event(energy=1.0))
    assert not [p for p in field.particles if p.kind == "plume"]


def test_swallow_sparks_travel_inward():
    from src.render.effects import EffectField
    import numpy as np
    field = EffectField(seed=1)
    field.spawn(_swallow_event())
    sparks = [p for p in field.particles if p.kind == "spark"]
    assert sparks
    for p in sparks:
        # Position is offset from the centre; velocity points back at it.
        assert float(np.dot(p.pos, p.vel)) < 0.0


def test_an_ordinary_impact_still_explodes_outward():
    import dataclasses
    from src.collisions import ACCRETE
    from src.render.effects import EffectField
    ordinary = dataclasses.replace(_swallow_event(), outcome=ACCRETE)
    field = EffectField(seed=1)
    field.spawn(ordinary)
    rings = [p for p in field.particles if p.kind == "ring"]
    assert rings
    assert all(p.r1 > p.r0 for p in rings)


# --- throwing -------------------------------------------------------------------

def test_there_is_a_black_hole_preset_on_key_six():
    from src.app.tools import PRESETS
    hole = [p for p in PRESETS if p.label == "Black hole"]
    assert len(hole) == 1
    assert hole[0].key == 6


def test_the_preset_derives_its_radius_from_the_horizon():
    from src import units
    from src.app.tools import PRESETS, preset_body
    hole = next(p for p in PRESETS if p.label == "Black hole")
    mass, radius = preset_body(hole)
    assert radius == pytest.approx(units.schwarzschild_radius(mass))


def test_existing_presets_are_unaffected():
    from src.app.tools import PRESETS, preset_body
    rocky = next(p for p in PRESETS if p.label == "Rocky planet")
    mass, radius = preset_body(rocky)
    assert radius == pytest.approx(6371.0 / 1.495978707e8)
    assert mass > 0.0


def test_launching_the_preset_creates_a_feeding_black_hole():
    import numpy as np
    from src.body import (DEFAULT_BLACK_HOLE_DISK, DEFAULT_BLACK_HOLE_SPIN,
                          KIND_BLACK_HOLE)
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.app.state import AppState
    from src.app.tools import PRESETS, LaunchTool
    from src.sim import Simulation
    from src import scenarios

    sim = Simulation(world=scenarios.two_body())
    state = AppState(sim, Camera(320, 200), ViewSettings())
    tool = LaunchTool()
    tool.preset_index = next(k for k, p in enumerate(PRESETS)
                             if p.label == "Black hole")
    tool._origin = np.array([3.0, 0.0])
    tool._cursor = (170, 100)
    before = sim.world.n
    tool._launch(state)

    assert sim.world.n == before + 1
    body = sim.world.bodies[-1]
    assert body.kind == KIND_BLACK_HOLE
    assert body.spin == pytest.approx(DEFAULT_BLACK_HOLE_SPIN)
    assert body.disk == pytest.approx(DEFAULT_BLACK_HOLE_DISK)
    assert float(sim.world.radius[-1]) == pytest.approx(
        __import__("src.units", fromlist=["x"]).schwarzschild_radius(
            float(sim.world.mass[-1])))


# --- editing --------------------------------------------------------------------

def _edit_state(world, select=0):
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.app.state import AppState
    from src.app.tools import EditTool
    from src.sim import Simulation
    sim = Simulation(world=world)
    state = AppState(sim, Camera(320, 200), ViewSettings())
    state.set_tool(EditTool)
    state.selected_id = world.bodies[select].id
    return state


def _ids(rows):
    return [c.id for row in rows for c in row]


def _labels(rows):
    return [c.label for row in rows for c in row]


def test_a_star_is_offered_a_collapse_button(hole_world):
    from src.render.widgets import Metrics
    from src.render.pixelfont import PixelFont
    from src import edits, materials, units
    from src.body import KIND_STAR
    import numpy as np
    hole_world.add("Sun", materials.get("hydrogen"), 1.0, units.SUN_RADIUS,
                   np.array([5.0, 0.0]), np.zeros(2), kind=KIND_STAR)
    state = _edit_state(hole_world, select=2)
    rows = state.tool.controls(state, Metrics(PixelFont(18), 320, 200))
    assert "edit.collapse" in _ids(rows)


def test_a_planet_is_not_offered_a_collapse_button(hole_world):
    from src.render.widgets import Metrics
    from src.render.pixelfont import PixelFont
    state = _edit_state(hole_world, select=1)      # Earth
    rows = state.tool.controls(state, Metrics(PixelFont(18), 320, 200))
    collapse = [c for row in rows for c in row if c.id == "edit.collapse"]
    assert not collapse or not collapse[0].enabled


def test_a_black_hole_is_offered_spin_and_disk_not_density(hole_world):
    from src.render.widgets import Metrics
    from src.render.pixelfont import PixelFont
    state = _edit_state(hole_world, select=0)
    rows = state.tool.controls(state, Metrics(PixelFont(18), 320, 200))
    labels = " ".join(_labels(rows))
    assert "Spin" in labels
    assert "Disk" in labels
    assert "Density" not in labels
    assert "Material" not in labels


def test_the_collapse_button_collapses_the_selected_star(hole_world):
    from src.render.widgets import Control
    from src import materials, units
    from src.body import KIND_BLACK_HOLE, KIND_STAR
    import numpy as np
    hole_world.add("Sun", materials.get("hydrogen"), 1.0, units.SUN_RADIUS,
                   np.array([5.0, 0.0]), np.zeros(2), kind=KIND_STAR)
    state = _edit_state(hole_world, select=2)
    assert state.tool.perform(Control("edit.collapse"), state)
    assert hole_world.bodies[2].kind == KIND_BLACK_HOLE


def test_the_c_key_collapses_a_selected_star(hole_world):
    from src import materials, units
    from src.body import KIND_BLACK_HOLE, KIND_STAR
    import numpy as np
    hole_world.add("Sun", materials.get("hydrogen"), 1.0, units.SUN_RADIUS,
                   np.array([5.0, 0.0]), np.zeros(2), kind=KIND_STAR)
    state = _edit_state(hole_world, select=2)
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c, mod=0)
    assert state.tool.on_event(event, state)
    assert hole_world.bodies[2].kind == KIND_BLACK_HOLE


def test_the_c_key_does_not_collapse_a_planet(hole_world):
    from src.body import KIND_PLANET
    state = _edit_state(hole_world, select=1)      # Earth
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c, mod=0)
    state.tool.on_event(event, state)
    assert hole_world.bodies[1].kind == KIND_PLANET


def test_scrubbing_spin_changes_it(hole_world):
    from src import edits
    state = _edit_state(hole_world, select=0)
    state.tool.field_index = edits.BLACK_HOLE_EDIT_FIELDS.index("spin")
    before = hole_world.bodies[0].spin
    state.tool._scrub(state, +3)
    assert hole_world.bodies[0].spin > before


def test_scrubbing_disk_changes_it(hole_world):
    from src import edits
    state = _edit_state(hole_world, select=0)
    state.tool.field_index = edits.BLACK_HOLE_EDIT_FIELDS.index("disk")
    before = hole_world.bodies[0].disk
    state.tool._scrub(state, -3)
    assert hole_world.bodies[0].disk < before


def test_scrubbing_mass_still_works_on_a_black_hole(hole_world):
    from src import edits, units
    state = _edit_state(hole_world, select=0)
    state.tool.field_index = edits.BLACK_HOLE_EDIT_FIELDS.index("mass")
    state.tool._scrub(state, +5)
    assert float(hole_world.radius[0]) == pytest.approx(
        units.schwarzschild_radius(float(hole_world.mass[0])))


def test_the_c_key_is_documented():
    """tests/test_ui.py reads KEY_HELP and fails if a documented key has no
    button; this asserts the other direction."""
    from src.render.hud import KEY_HELP
    assert any(keys == "C" for keys, _ in KEY_HELP)


# --- the inspector ---------------------------------------------------------------

def _inspector_text(world, index):
    from src.render.hud import Hud
    from src.render.pixelfont import PixelFont
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.app.state import AppState
    from src.sim import Simulation
    state = AppState(Simulation(world=world), Camera(320, 200), ViewSettings())
    state.selected_id = world.bodies[index].id
    return "\n".join(text for text, _ in Hud(PixelFont(18))._rows(state, index))


def test_the_inspector_names_a_black_hole(hole_world):
    assert "black hole" in _inspector_text(hole_world, 0)


def test_the_inspector_reports_escape_velocity_as_the_speed_of_light(hole_world):
    """Falls out of escape_velocity(M, r_s) with no special case."""
    text = _inspector_text(hole_world, 0)
    assert "299792" in text.replace(",", "") or "2.998e+05" in text


def test_the_inspector_shows_the_horizon_shadow_and_isco(hole_world):
    text = _inspector_text(hole_world, 0)
    for label in ("Horizon", "Shadow", "Orbit edge"):
        assert label in text


def test_the_inspector_shows_spin_and_efficiency(hole_world):
    text = _inspector_text(hole_world, 0)
    assert "Spin" in text
    assert "%" in text


def test_the_inspector_never_offers_to_ignite_a_black_hole(hole_world):
    """"0x heavier to ignite" would be nonsense."""
    assert "ignite" not in _inspector_text(hole_world, 0)


def test_an_ordinary_planet_still_offers_to_ignite(hole_world):
    assert "ignite" in _inspector_text(hole_world, 1)


def test_the_inspector_still_describes_an_ordinary_planet(hole_world):
    text = _inspector_text(hole_world, 1)
    assert "Density" in text
    assert "Gravity" in text


# --- the scenario ----------------------------------------------------------------

def test_the_black_hole_scenario_is_registered():
    from src import scenarios
    assert "black_hole" in scenarios.REGISTRY


def test_the_scenario_contains_exactly_one_black_hole():
    from src import scenarios
    from src.body import KIND_BLACK_HOLE
    world = scenarios.black_hole()
    holes = [b for b in world.bodies if b.kind == KIND_BLACK_HOLE]
    assert len(holes) == 1
    assert holes[0].disk > 0.0


def test_the_scenario_shreds_something_almost_immediately():
    """It exists to show a tidal disruption; requiring the child to arrange
    one would defeat the point.  Bounded to about 30 days -- three times the
    brief's own ~10-day estimate for the plunge -- so a regression that made
    the plunge implausibly slow, while still nominally shredding something
    eventually, could not hide behind a loose bound."""
    from src import scenarios
    from src.sim import Simulation
    sim = Simulation(world=scenarios.black_hole())
    before = sim.world.n
    for _ in range(60):
        sim.advance(0.5, budget_s=1.0)
        if sim.world.n != before:
            break
    assert sim.world.n != before, "nothing was disrupted in 30 days"


def test_the_scenario_does_not_cascade_the_world_to_capacity():
    """Pins that the disruption itself stays a large, slow, visible event,
    not a chain reaction that fills the world on the very next substep --
    it does NOT claim the world never fills. Fragments are born a
    body-radius from where their parent died, deep inside the tidal radius
    that killed it, and a surviving fragment can still return on a later
    orbit and be shredded again: that longer-timescale refilling is real
    physics (a scale-invariant tidal ratio plus a periapsis that has not
    moved), not a bug this test is meant to catch."""
    from src import scenarios
    from src.sim import Simulation
    sim = Simulation(world=scenarios.black_hole())
    before = sim.world.n
    for _ in range(60):
        sim.advance(0.5, budget_s=1.0)
        if sim.world.n != before:
            break
    assert sim.world.n != before, "nothing was disrupted in 30 days"
    assert sim.world.n < 40, \
        f"cascaded to {sim.world.n} bodies -- near the 200-body cap, not a modest event"


def test_the_scenario_is_stable_enough_to_run():
    """No NaNs, nothing flung to infinity."""
    import numpy as np
    from src import scenarios
    from src.sim import Simulation
    sim = Simulation(world=scenarios.black_hole())
    for _ in range(200):
        sim.advance(1.0, budget_s=1.0)
    assert np.all(np.isfinite(sim.world.positions))


def test_cycling_the_scenario_changes_what_restart_loads():
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.app.state import AppState
    from src.body import KIND_BLACK_HOLE
    from src.sim import Simulation
    state = AppState(Simulation(), Camera(320, 200), ViewSettings())
    assert state.scenario == "solar_system"
    while state.scenario != "black_hole":
        state.cycle_scenario()
    state.reset()
    assert any(b.kind == KIND_BLACK_HOLE for b in state.sim.world.bodies)


def test_cycling_the_scenario_does_not_restart():
    """Cycling only changes what Restart will load next time it is pressed --
    it must not touch the running simulation.  A stray click on the cycle
    button must cost nothing."""
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.app.state import AppState
    from src.sim import Simulation
    state = AppState(Simulation(), Camera(320, 200), ViewSettings())
    state.reset()
    state.sim.advance(5.0, budget_s=1.0)
    time_before = state.sim.world.time
    n_before = state.sim.world.n
    for _ in range(5):
        state.cycle_scenario()
    assert state.sim.world.time == time_before
    assert state.sim.world.n == n_before


def test_restart_reloads_the_current_scenario():
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.app.state import AppState
    from src.sim import Simulation
    state = AppState(Simulation(), Camera(320, 200), ViewSettings())
    state.reset()
    assert any(b.name == "Sun" for b in state.sim.world.bodies)
