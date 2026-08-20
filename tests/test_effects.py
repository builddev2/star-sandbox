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


def _event(energy=1e-8, target_radius=4.26e-5, projectile_radius=4.26e-7,
          on_star=False, pos=(0.0, 0.0), vel=(0.01, 0.0)):
    from src.collisions import ACCRETE, ImpactEvent
    return ImpactEvent(
        pos=pos, vel=vel, energy=energy, outcome=ACCRETE,
        target_radius=target_radius, projectile_radius=projectile_radius,
        tones=((0x4A, 0x33, 0x20), (0xB0, 0x80, 0x50), (0xE0, 0xB4, 0x84)),
        on_star=on_star,
    )


def _camera():
    from src.render.camera import Camera
    return Camera(640, 400)


def test_an_impact_spawns_particles():
    from src.render.effects import EffectField
    field = EffectField(seed=1)
    assert field.count == 0
    field.spawn(_event())
    assert field.count > 0


def test_particles_expire_on_wall_clock_time_not_simulated_time():
    from src.render.effects import EffectField
    field = EffectField(seed=1)
    field.spawn(_event())
    assert field.count > 0
    # A generous number of real seconds -- everything has a life under 2s.
    field.update(5.0)
    assert field.count == 0


def test_the_particle_cap_holds_under_a_flood_of_impacts():
    from src.render.effects import EffectField, MAX_PARTICLES
    field = EffectField(seed=1)
    for _ in range(50):
        field.spawn(_event(energy=1.0))   # big, energetic: lots of sparks each
    assert field.count <= MAX_PARTICLES


def test_drawing_an_empty_field_is_a_no_op():
    from src.render.effects import EffectField
    surface = pygame.Surface((320, 200))
    before = surface.copy()
    field = EffectField(seed=1)
    field.draw(surface, _camera())
    assert bytes(surface.get_buffer()) == bytes(before.get_buffer())


def test_a_star_impact_spawns_a_flare_and_a_planet_impact_does_not():
    from src.render.effects import EffectField
    field = EffectField(seed=1)
    field.spawn(_event(on_star=True))
    assert any(p.kind == "flare" for p in field.particles)

    field2 = EffectField(seed=1)
    field2.spawn(_event(on_star=False))
    assert not any(p.kind == "flare" for p in field2.particles)


def test_effects_drift_with_the_impact_velocity():
    from src.render.effects import EffectField
    # A large event.vel (AU/day, a physics quantity) must never leak into a
    # particle's real-second screen velocity -- that was a real bug caught
    # by this test: a flash inheriting event.vel directly as an AU-per-
    # real-second drift shot clean off-screen within a single frame, even
    # though a flash is supposed to mark a fixed point.
    field = EffectField(seed=1)
    field.spawn(_event(pos=(0.0, 0.0), vel=(5.0, 0.0)))
    flash = next(p for p in field.particles if p.kind == "flash")
    field.update(0.2)
    assert flash.pos[0] == pytest.approx(0.0, abs=1e-9), \
        "the flash must stay put regardless of the impact's physics velocity"

    # A spark, by contrast, does move outward -- on its own purpose-built
    # visual speed, not the physics one.
    spark = next(p for p in field.particles if p.kind == "spark")
    start = spark.pos.copy()
    field.update(0.2)
    assert not np.allclose(spark.pos, start)


def test_update_and_draw_survive_an_empty_or_full_field_without_raising():
    from src.render.effects import EffectField
    surface = pygame.Surface((320, 200))
    field = EffectField(seed=1)
    field.update(0.016)
    field.draw(surface, _camera())
    for _ in range(30):
        field.spawn(_event(energy=1.0))
    field.update(0.016)
    field.draw(surface, _camera())
