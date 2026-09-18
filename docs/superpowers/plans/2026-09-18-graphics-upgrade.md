# Graphics and Resolution Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Native-Retina, smooth, shaded rendering -- per-pixel lit spheres with real-looking surfaces, a corona, a coloured pre-rendered sky, antialiased trails and text -- without changing the architecture.

**Architecture:** `DisplayContext` swaps `pygame.display.set_mode` for a high-DPI `pygame.Window` + `Renderer` + streaming `Texture`, uploading the one software surface once a frame; `s` becomes internal pixels per point. A new `render/shade.py` (pure numpy) generates every planet, moon and star sprite; `sprites.py` keeps the cache and adds a `look` per body so Earth is blue. Sky, trails, text and effects go antialiased.

**Tech Stack:** Python 3.11, pygame-ce 2.5.8 (drop-in for pygame; `import pygame` unchanged), numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-18-graphics-upgrade-design.md`

## Global Constraints

- The test suite is 1253 tests today and must end at least that size; the suite takes ~4 minutes -- **always run pytest with a 600000 ms Bash timeout**, never the default.
- `src/*.py` (physics) never imports pygame; `tests/test_layering.py` enforces it in a subprocess.
- Every test that touches pygame sets `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")` at module top before importing pygame.
- Nothing in `src/` may call `pygame.Window.from_display_module()` -- that object destroys the real window when garbage-collected.
- `World.MAX_BODIES` is 200; `sim.DEFAULT_BUDGET_S` is 0.012; neither changes.
- Simulation units are AU, solar masses, days; `src/units.py` is the only place a conversion constant lives.
- Nothing may cache a pixel size across frames except through `App._build_render_stack`, which is rebuilt whenever the buffer changes shape.
- `render/shade.py` and every sprite generator must return a surface built with `pygame.image.frombuffer(...).copy()` -- **never `convert()` / `convert_alpha()`** -- because once the window is a `pygame.Window` there is no display surface to convert to.
- Every commit ends with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` on its own line, after a blank line, and is made on branch `graphics-upgrade` (already created and holding the spec).
- The Python to use is `./.venv/bin/python` (Task 1 recreates the venv with pygame-ce).
- Commit messages follow the repo's style: `feat:`, `fix:`, `docs:`, `test:`, lower-case, a short subject that says what and why.

---

## Phase 1 -- Display

### Task 1: Swap the dependency to pygame-ce and prove the old code still runs on it

**Files:**
- Modify: `requirements.txt`
- Modify: `run.command`
- Modify: `CLAUDE.md` (Commands block, lines 7-18)

**Interfaces:**
- Produces: a `.venv` with `pygame-ce>=2.5.8` where `import pygame` gives `pygame.version.ver` starting `2.5`, and a green suite on it.

- [ ] **Step 1: Point requirements at pygame-ce**

Replace the whole of `requirements.txt` with:

```
pygame-ce>=2.5.8
numpy>=1.26
pytest>=7.4
```

- [ ] **Step 2: Teach the launcher to evict the old package**

Replace the whole of `run.command` with:

```bash
#!/bin/bash
# Double-click me in Finder.
cd "$(dirname "$0")" || exit 1

if [ ! -d .venv ]; then
    echo "First run: setting up..."
    python3 -m venv .venv || exit 1
    ./.venv/bin/pip install --quiet --upgrade pip
fi

# pygame and pygame-ce unpack into the same "pygame" directory and cannot
# coexist.  A venv built before the switch to pygame-ce has the old one; it
# must go before the requirements are installed, or pip leaves a broken mix.
if ./.venv/bin/pip show pygame >/dev/null 2>&1; then
    echo "Replacing pygame with pygame-ce..."
    ./.venv/bin/pip uninstall --quiet -y pygame
fi
./.venv/bin/pip install --quiet -r requirements.txt || exit 1

exec ./.venv/bin/python -m src.app.main
```

- [ ] **Step 3: Recreate the venv**

Run:
```bash
./.venv/bin/pip uninstall -y pygame && ./.venv/bin/pip install -q -r requirements.txt && ./.venv/bin/python -c "import pygame; print(pygame.version.ver)"
```
Expected: the last line starts with `2.5`.

- [ ] **Step 4: Run the whole suite on pygame-ce, before any code changes**

Run (600000 ms timeout): `./.venv/bin/python -m pytest -q tests 2>&1 | tail -15`
Expected: `1253 passed` (deprecation warnings about `pygame.image.tostring` are fine). If anything fails, fix only what the package swap broke, and say what it was in the commit message.

- [ ] **Step 5: Update the Commands block in CLAUDE.md**

In `CLAUDE.md`, replace the paragraph starting "The venv at `.venv/` is created by `run.command`" with:

```
The venv at `.venv/` is created by `run.command`; use `./.venv/bin/python` if the system
Python lacks pygame/numpy. The dependency is **pygame-ce** (`import pygame` is unchanged);
it cannot share a venv with the original `pygame`, so `run.command` uninstalls that first
whenever it finds it.
```

- [ ] **Step 6: Commit**

```bash
git add requirements.txt run.command CLAUDE.md
git commit -m "chore: switch to pygame-ce, the fork that can open a high-DPI window

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: `DisplayContext` on a high-DPI Window, Renderer and Texture

**Files:**
- Modify: `src/app/config.py` (whole module below the docstring)
- Modify: `tests/test_config.py` (rewrite)
- Modify: `tests/test_window.py` (rewrite)

**Interfaces:**
- Produces: `DisplayContext(s=None, fullscreen=False, size=None, fraction_index=DEFAULT_FRACTION_INDEX, pixels_per_point=None)` with attributes `native` (int), `s` (int), `width`, `height`, `surface` (software `Surface`), `window` (size tuple, points), `window_size`, `usable_grids` (tuple), and methods `present()`, `to_internal(pos)`, `map_event(event)`, `mouse_pos()`, `set_grid(s)`, `resize(size=None)`, `set_window_fraction(i)`, `cycle_window_size()`, `toggle_fullscreen()`, `size_changed()` (bool), `close()`; properties `upscale`, `font_px`, `resizable`, `window_size_label`. Module functions `internal_size(w, h, s)`, `font_px(s)`, `min_internal(s)`, `clamp_window`, `preset_size`, `open_display(s=None, fullscreen=False, size=None)`. Constants `TARGET_TEXT_POINTS = 11`, `MIN_INTERNAL`, `MIN_WINDOW`, `WINDOW_FRACTIONS`, `DEFAULT_FRACTION_INDEX`, `MENU_BAR_POINTS`, `TITLE_BAR_POINTS`.
- Removed: `DEFAULT_S`, `GRID_CHOICES`, `MIN_FONT_PX`, `grid_fits`, `ctx.display`, `ctx.origin`.

- [ ] **Step 1: Write the new config tests**

Replace the whole of `tests/test_config.py` with:

```python
import pytest

from src.app import config
from src.app.config import font_px, internal_size, min_internal


@pytest.mark.parametrize("w,h,s,expected", [
    # A 16" MacBook Pro reports 1728x1117 logical points; see config.py.
    (1728, 1117, 1, (1728, 1117)),
    (1728, 1117, 2, (3456, 2234)),
    (1920, 1080, 2, (3840, 2160)),
])
def test_internal_size_is_the_window_times_the_pixels_per_point(w, h, s, expected):
    assert internal_size(w, h, s) == expected


def test_internal_size_never_returns_zero():
    assert internal_size(0, 0, 1) == (1, 1)


def test_text_is_the_intended_size_at_one_pixel_per_point():
    assert font_px(1) == config.TARGET_TEXT_POINTS


def test_text_doubles_with_the_pixel_density():
    """Apparent size is what stays constant, not the pixel count."""
    assert font_px(2) == 2 * font_px(1)


def test_the_interface_floor_scales_with_the_pixel_density():
    assert min_internal(2) == (2 * config.MIN_INTERNAL[0], 2 * config.MIN_INTERNAL[1])
    assert min_internal(1) == config.MIN_INTERNAL
```

- [ ] **Step 2: Write the new window tests**

Replace the whole of `tests/test_window.py` with:

```python
"""The window: resizing it, the size presets, and the pixels inside it.

The bug the old version of these existed for was invisible from inside the
program: ``SCALED`` pinned the window's minimum size to the buffer and every
second ``set_mode`` failed silently.  The lesson survives the move to
``pygame.Window``: check the *outcome* -- the buffer, the window, the
coordinates -- and never merely that a call returned.

The dummy driver reports one pixel per point.  Tests that need a Retina
display say so with ``pixels_per_point=2``, which is what the constructor
offers in place of measuring.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest

from src.app import config
from src.app.config import (DisplayContext, MIN_WINDOW, WINDOW_FRACTIONS,
                            clamp_window, font_px, internal_size,
                            min_internal, preset_size)

#: A 16" MacBook Pro, the machine the measurements in config.py come from.
LAPTOP = (1728, 1117)


@pytest.fixture
def ctx():
    pygame.display.init()
    context = DisplayContext(size=(800, 600))
    yield context
    context.close()
    pygame.display.quit()


@pytest.fixture
def retina():
    """A context that believes the display has two pixels per point."""
    pygame.display.init()
    context = DisplayContext(size=(800, 600), pixels_per_point=2)
    yield context
    context.close()
    pygame.display.quit()


# --- the numbers ----------------------------------------------------------------

def test_the_smallest_window_is_smaller_than_a_third_of_a_laptop_screen():
    """Otherwise the one-third preset clamps back up and stops being one third."""
    third = preset_size(LAPTOP, 1 / 3)
    assert MIN_WINDOW[0] < LAPTOP[0] // 3
    assert third[0] == round(LAPTOP[0] / 3)


def test_a_preset_leaves_room_for_the_menu_bar_and_the_title_bar():
    for fraction, _ in WINDOW_FRACTIONS:
        w, h = preset_size(LAPTOP, fraction)
        assert h < LAPTOP[1] - config.MENU_BAR_POINTS
        assert w <= LAPTOP[0]


def test_the_presets_are_the_fractions_they_say_they_are():
    widths = [preset_size(LAPTOP, f)[0] for f, _ in WINDOW_FRACTIONS]
    assert widths == sorted(widths, reverse=True)
    assert widths[0] == LAPTOP[0]
    assert widths[2] == LAPTOP[0] // 2


def test_clamp_holds_a_window_between_the_floor_and_the_screen():
    assert clamp_window((10, 10)) == MIN_WINDOW
    assert clamp_window((9999, 9999), LAPTOP) == LAPTOP
    assert clamp_window((900, 700), LAPTOP) == (900, 700)


# --- the window itself -----------------------------------------------------------

def test_the_window_is_resizable_and_not_pinned_to_the_screen(ctx):
    assert ctx.resizable
    assert not ctx.fullscreen
    assert ctx.window[0] < ctx.desktop[0] or ctx.window[1] < ctx.desktop[1]


def test_resizing_moves_the_buffer_with_the_window(ctx):
    ctx.resize((640, 480))
    assert ctx.window == (640, 480)
    assert (ctx.width, ctx.height) == (640, 480)
    ctx.resize((1000, 700))
    assert (ctx.width, ctx.height) == (1000, 700)


def test_a_drag_below_the_floor_is_pushed_back_out_to_it(ctx):
    ctx.resize((100, 80))
    assert ctx.window == MIN_WINDOW
    assert (ctx.width, ctx.height) == MIN_WINDOW


def test_every_preset_lands_on_a_window_the_interface_fits_in(ctx):
    seen = []
    for _ in range(len(WINDOW_FRACTIONS)):
        label = ctx.cycle_window_size()
        seen.append(label)
        assert ctx.width >= min_internal(ctx.s)[0]
        assert ctx.height >= min_internal(ctx.s)[1]
    assert sorted(seen) == sorted(name for _, name in WINDOW_FRACTIONS)


def test_the_presets_cycle_back_round(ctx):
    first = ctx.cycle_window_size()
    for _ in range(len(WINDOW_FRACTIONS) - 1):
        ctx.cycle_window_size()
    assert ctx.cycle_window_size() == first


def test_a_preset_leaves_fullscreen(ctx):
    ctx.toggle_fullscreen()
    assert ctx.fullscreen
    ctx.cycle_window_size()
    assert not ctx.fullscreen


def test_fullscreen_and_back_returns_to_the_size_it_left(ctx):
    ctx.resize((720, 540))
    before = ctx.window
    ctx.toggle_fullscreen()
    assert ctx.fullscreen
    ctx.toggle_fullscreen()
    assert not ctx.fullscreen
    assert ctx.window == before


def test_size_changed_notices_a_resize_the_context_was_not_told_about(ctx):
    """The once-a-frame safety poll: a maximise or a macOS tile reaches the
    window without an event we listen for."""
    assert not ctx.size_changed()
    ctx._window.size = (900, 650)
    pygame.event.pump()
    assert ctx.size_changed()
    ctx.resize()
    assert not ctx.size_changed()
    assert ctx.window == (900, 650)


# --- pixels per point ----------------------------------------------------------------

def test_the_dummy_driver_has_one_pixel_per_point(ctx):
    assert ctx.native == 1
    assert ctx.s == 1
    assert ctx.usable_grids == (1,)
    assert ctx.surface.get_size() == ctx.window


def test_a_retina_display_opens_at_native_density(retina):
    assert retina.native == 2
    assert retina.s == 2
    assert retina.usable_grids == (2, 1)
    assert (retina.width, retina.height) == internal_size(800, 600, 2)
    assert retina.surface.get_size() == (1600, 1200)
    assert (retina.texture.width, retina.texture.height) == (1600, 1200)


def test_standard_density_is_the_escape_hatch(retina):
    """Half the pixels for when the physics budget is saturated at fullscreen."""
    retina.set_grid(1)
    assert retina.s == 1
    assert retina.surface.get_size() == (800, 600)
    assert (retina.texture.width, retina.texture.height) == (800, 600)
    assert retina.upscale == 1.0


def test_a_density_the_display_does_not_have_is_refused(retina):
    retina.set_grid(3)
    assert retina.s == 2
    retina.set_grid(1)
    retina.set_grid(7)
    assert retina.s == 2, "refused, and back to native rather than left wherever it was"


def test_a_resize_rebuilds_the_texture_to_match(retina):
    retina.resize((900, 650))
    assert retina.surface.get_size() == (1800, 1300)
    assert (retina.texture.width, retina.texture.height) == (1800, 1300)


def test_present_runs_after_every_kind_of_change(retina):
    retina.present()
    retina.resize((640, 480))
    retina.present()
    retina.set_grid(1)
    retina.present()
    retina.toggle_fullscreen()
    retina.present()


def test_the_font_and_the_floor_follow_the_density(retina):
    assert retina.font_px == 2 * font_px(1)
    assert min_internal(retina.s) == (2 * MIN_WINDOW[0], 2 * MIN_WINDOW[1])


# --- window points to internal pixels ----------------------------------------------

def test_at_one_pixel_per_point_a_click_needs_no_translation(ctx):
    assert ctx.to_internal((37, 41)) == (37, 41)


def test_at_two_pixels_per_point_a_click_is_doubled(retina):
    assert retina.to_internal((100, 50)) == (200, 100)


def test_a_mouse_event_is_translated_on_its_way_in(retina):
    event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": (60, 80), "button": 1})
    assert retina.map_event(event).pos == (120, 160)


def test_a_translated_event_keeps_everything_else_it_carried(retina):
    event = pygame.event.Event(pygame.MOUSEMOTION,
                               {"pos": (100, 100), "rel": (4, 8), "buttons": (1, 0, 0)})
    mapped = retina.map_event(event)
    assert mapped.buttons == (1, 0, 0)
    assert mapped.rel == (8, 16)


def test_an_event_with_no_position_passes_straight_through(retina):
    event = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_SPACE, "mod": 0})
    assert retina.map_event(event) is event


def test_at_one_pixel_per_point_an_event_is_not_copied(ctx):
    event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": (60, 80), "button": 1})
    assert ctx.map_event(event) is event


# --- the keys ---------------------------------------------------------------------

def _key(key, mod=0):
    return pygame.event.Event(pygame.KEYDOWN, {"key": key, "mod": mod, "unicode": ""})


def test_w_steps_the_window_size(ctx):
    from src.app.input import handle_event
    from src.app.state import AppState
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.sim import Simulation

    state = AppState(Simulation(), Camera(ctx.width, ctx.height), ViewSettings())
    before = ctx.window
    handle_event(_key(pygame.K_w), state, ctx)
    assert ctx.window != before
    assert (state.camera.width, state.camera.height) == (ctx.width, ctx.height)


def test_g_toggles_the_density_on_a_retina_display(retina):
    from src.app.input import handle_event
    from src.app.state import AppState
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.sim import Simulation

    state = AppState(Simulation(), Camera(retina.width, retina.height), ViewSettings())
    handle_event(_key(pygame.K_g), state, retina)
    assert retina.s == 1
    assert (state.camera.width, state.camera.height) == (800, 600)
    handle_event(_key(pygame.K_g), state, retina)
    assert retina.s == 2
    assert (state.camera.width, state.camera.height) == (1600, 1200)


def test_command_w_asks_to_quit_rather_than_resizing(ctx):
    """Cmd+W closes a window everywhere else on this platform, and nothing here
    is saved, so falling through to the resize would be the one unrecoverable
    guess."""
    from src.app.input import handle_event
    from src.app.state import AppState
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.sim import Simulation

    state = AppState(Simulation(), Camera(ctx.width, ctx.height), ViewSettings())
    before = ctx.window
    handle_event(_key(pygame.K_w, pygame.KMOD_META), state, ctx)
    assert ctx.window == before
    assert state.quit_armed
    assert state.running, "one press arms it, it does not go"


def test_a_drag_is_not_capped_at_the_primary_display(ctx):
    """``desktop`` is the *primary* screen.  A window dragged onto a bigger
    external monitor is entitled to be bigger than the laptop it started on, so
    only the presets -- which are fractions of a screen -- are capped to one."""
    huge = (ctx.desktop[0] + 400, ctx.desktop[1] + 300)
    ctx.resize(huge)
    assert ctx.window_size == huge


def test_close_destroys_the_window_and_can_be_called_twice(ctx):
    ctx.close()
    ctx.close()
    assert ctx._window is None
```

- [ ] **Step 3: Run the new tests to see them fail**

Run: `./.venv/bin/python -m pytest -q tests/test_config.py tests/test_window.py 2>&1 | tail -5`
Expected: ImportError / failures (no `pixels_per_point`, no `native`, `GRID_CHOICES` gone).

- [ ] **Step 4: Rewrite `src/app/config.py`**

Replace everything in `src/app/config.py` from the line `import pygame` to the end of the file with the following. Leave the module docstring above it in place for now (Step 5 rewrites it).

```python
import os

import pygame
from pygame._sdl2.video import Renderer, Texture

# When the buffer is drawn at standard density onto a Retina window the GPU
# scales it up; this asks SDL for linear filtering rather than nearest, so
# the picture is soft rather than blocky.  Read when the texture is created.
os.environ.setdefault("SDL_RENDER_SCALE_QUALITY", "1")

#: Text height in points at one pixel per point -- the font's point size.
#: Sized for a child reading alongside an adult on a laptop screen, not for
#: maximum information density.  (Atkinson Hyperlegible at 11 renders a 14 px
#: line, a shade taller and narrower than the built-in face used to be.)
TARGET_TEXT_POINTS = 11

#: Height macOS keeps for itself above a window, and for the window's own title
#: bar.  A preset that ignores them puts the transport bar under the Dock.
MENU_BAR_POINTS = 37
TITLE_BAR_POINTS = 28

#: The smallest window the whole interface lays itself out inside, in points.
#: ``tests/test_layout.py`` is what keeps these two numbers honest: it builds
#: every panel of every tool at exactly this size (at every pixel density) and
#: fails if a single control or panel lands off the edge.
#:
#: The width has to stay below a third of a laptop desktop (1728 / 3 = 576) or
#: the one-third preset would clamp straight back up and stop being one third.
#: The height is set by the help panel, which is the tallest thing here.
MIN_INTERNAL = (480, 440)
MIN_WINDOW = MIN_INTERNAL

#: The window sizes offered as presets, as a fraction of the desktop *width*.
#: Height is always the whole usable desktop, because that is what half-screen
#: means to everyone who has ever dragged a window against the side of a
#: screen -- half as wide, full height -- and what macOS's own tiling does.
WINDOW_FRACTIONS = (
    (1.0, "Full width"),
    (2 / 3, "Two thirds"),
    (1 / 2, "Half screen"),
    (1 / 3, "One third"),
)
#: Opens at two thirds: unmistakably a window, with room for the whole interface.
DEFAULT_FRACTION_INDEX = 1

WINDOW_TITLE = "Solar System Sandbox"


def internal_size(window_w, window_h, s):
    """Internal buffer size for a window in points at ``s`` pixels per point."""
    return (max(1, int(window_w) * s), max(1, int(window_h) * s))


def font_px(s):
    """Font size in internal pixels: the point size times the pixel density,
    so the text is the same physical size at every density."""
    return TARGET_TEXT_POINTS * max(1, int(s))


def min_internal(s):
    """Smallest buffer the interface lays out in, at ``s`` pixels per point."""
    return (MIN_INTERNAL[0] * max(1, int(s)), MIN_INTERNAL[1] * max(1, int(s)))


def clamp_window(size, desktop=None):
    """A window size the interface can actually be drawn in, and the screen hold."""
    w, h = max(MIN_WINDOW[0], int(size[0])), max(MIN_WINDOW[1], int(size[1]))
    if desktop is not None:
        w, h = min(w, max(1, desktop[0])), min(h, max(1, desktop[1]))
    return (w, h)


def preset_size(desktop, fraction):
    """Window size for one of ``WINDOW_FRACTIONS`` on this desktop."""
    return clamp_window(
        (round(desktop[0] * fraction),
         desktop[1] - MENU_BAR_POINTS - TITLE_BAR_POINTS),
        desktop)


class DisplayContext:
    """The window, the renderer that reaches its pixels, and the one software
    surface everything draws into.

    ``surface`` is a plain ``pygame.Surface`` sized ``window * s``; every
    frame ``present`` uploads it into a streaming ``Texture`` and the renderer
    draws that over the whole window.  At native density the texture is the
    window's pixel size exactly, so nothing is scaled; at standard density
    (``s = 1`` on a Retina display) the GPU scales it up with linear filtering.

    ``s`` is **internal pixels per point**.  ``native`` is what the display
    actually has -- measured from the renderer, never guessed -- and the only
    values ever offered are ``native`` and 1.
    """

    def __init__(self, s=None, fullscreen=False, size=None,
                 fraction_index=DEFAULT_FRACTION_INDEX, pixels_per_point=None):
        self.fullscreen = fullscreen
        self.desktop = self._desktop_size()
        self.fraction_index = fraction_index % len(WINDOW_FRACTIONS)
        #: Tests pass this to stand in for a display the dummy driver lacks.
        self._pixels_per_point = pixels_per_point
        self._window = None
        self.renderer = None
        self.texture = None
        self.surface = None
        self.native = 1
        self.s = s
        if size is None:
            size = preset_size(self.desktop, WINDOW_FRACTIONS[self.fraction_index][0])
        self.window_size = clamp_window(size, self.desktop)
        self.window = self.window_size
        self._open()

    @staticmethod
    def _desktop_size():
        sizes = pygame.display.get_desktop_sizes()
        return sizes[0] if sizes else (1920, 1080)

    def _open(self):
        """Build the window once, then put it in the state asked for and derive
        the buffer from the size we actually got."""
        if self._window is None:
            self._window = pygame.Window(WINDOW_TITLE, self.window_size,
                                         resizable=True, allow_high_dpi=True)
            try:
                self.renderer = Renderer(self._window, vsync=True)
            except pygame.error:
                # The dummy driver, and any other that cannot promise vsync.
                self.renderer = Renderer(self._window, vsync=False)
        if self.fullscreen:
            self._window.set_fullscreen(True)
        else:
            self._window.set_windowed()
            self._window.size = self.window_size
        pygame.event.pump()
        self._adopt()

    def _measure_native(self):
        """Pixels per point, from the renderer.  1 on the dummy driver."""
        if self._pixels_per_point is not None:
            return max(1, int(self._pixels_per_point))
        w, h = self.window
        dx, dy = self.renderer.coordinates_from_window((w, h))
        return max(1, int(round(float(dx) / max(1, w))))

    @property
    def usable_grids(self):
        """The densities on offer, native first: the display's own, and 1."""
        return (self.native, 1) if self.native > 1 else (1,)

    def _adopt(self):
        """Re-derive the buffer from the window that is actually open."""
        self.window = tuple(int(v) for v in self._window.size)
        if not self.fullscreen:
            self.window_size = self.window
        self.native = self._measure_native()
        if self.s not in self.usable_grids:
            self.s = self.usable_grids[0]
        self.width, self.height = internal_size(self.window[0], self.window[1], self.s)
        if self.surface is None or self.surface.get_size() != (self.width, self.height):
            self.surface = pygame.Surface((self.width, self.height))
            self.texture = Texture(self.renderer, (self.width, self.height),
                                   streaming=True)

    # --- the frame's last line ---------------------------------------------------

    def present(self):
        """Put the internal surface on the screen.  Replaces ``display.flip``."""
        self.texture.update(self.surface)
        self.renderer.clear()
        self.texture.draw()
        self.renderer.present()

    # --- window points to internal pixels ----------------------------------------

    @property
    def _identity(self):
        return self.s == 1

    def to_internal(self, pos):
        """A window point in internal pixels."""
        if self._identity:
            return (int(pos[0]), int(pos[1]))
        return (int(pos[0]) * self.s, int(pos[1]) * self.s)

    def map_event(self, event):
        """The same event with its mouse coordinates in internal pixels.

        Every reader of ``event.pos`` -- the interface, all the tools, the
        effects -- works in internal pixels, so the translation happens once,
        here, rather than at each of the two dozen places that would otherwise
        have to remember it.
        """
        if self._identity or not hasattr(event, "pos"):
            return event
        fields = dict(event.__dict__)
        fields["pos"] = self.to_internal(event.pos)
        if "rel" in fields:
            fields["rel"] = (int(fields["rel"][0]) * self.s,
                             int(fields["rel"][1]) * self.s)
        return pygame.event.Event(event.type, fields)

    def mouse_pos(self):
        """Where the pointer is, in internal pixels."""
        return self.to_internal(pygame.mouse.get_pos())

    @property
    def resizable(self):
        """True while the window has a drag handle and a title bar of its own."""
        return not self.fullscreen

    @property
    def upscale(self):
        """Window points per internal pixel."""
        return 1.0 / self.s

    @property
    def font_px(self):
        return font_px(self.s)

    # --- changing the window -----------------------------------------------------

    def size_changed(self):
        """Has the window been resized behind our back?  One cheap call a frame."""
        return tuple(int(v) for v in self._window.size) != self.window

    def set_grid(self, s):
        """Change the pixel density live.  Callers must clear any sprite cache.

        Only the display's native density and 1 are ever on offer; anything
        else is refused and lands on native.
        """
        self.s = s if s in self.usable_grids else self.usable_grids[0]
        self._adopt()

    def resize(self, size=None):
        """Take a new window size from the window manager.

        Below ``MIN_WINDOW`` the drag hits a floor and the window is pushed back
        out to it, because the interface has a smallest size it can be laid out
        in and a window under it would hide its own controls.  There is no
        ceiling: see below.
        """
        if self.fullscreen:
            self._adopt()
            return
        current = tuple(int(v) for v in self._window.size)
        # Floored, but deliberately not capped at ``self.desktop``: that is the
        # *primary* display, and a window dragged onto a bigger external monitor
        # is entitled to be bigger than the laptop screen.  Only the presets,
        # which are computed as fractions of a screen, are capped to one.
        wanted = clamp_window(size if size is not None else current)
        self.window_size = wanted
        if wanted != current:
            self._window.size = wanted
            pygame.event.pump()
        self._adopt()

    def set_window_fraction(self, index):
        """Snap the window to one of the presets.  Leaves fullscreen behind."""
        self.fraction_index = index % len(WINDOW_FRACTIONS)
        self.fullscreen = False
        self.window_size = preset_size(
            self.desktop, WINDOW_FRACTIONS[self.fraction_index][0])
        self._open()

    def cycle_window_size(self):
        """Step to the next preset.  Returns the label of the one now in force."""
        self.set_window_fraction(self.fraction_index + 1)
        return self.window_size_label

    @property
    def window_size_label(self):
        return WINDOW_FRACTIONS[self.fraction_index][1]

    def toggle_fullscreen(self):
        """Fullscreen and back.  Windowed returns to the size it left at."""
        self.fullscreen = not self.fullscreen
        self._open()

    def close(self):
        """Drop the texture and renderer, then destroy the window.  Idempotent."""
        self.texture = None
        self.renderer = None
        if self._window is not None:
            self._window.destroy()
            self._window = None


def open_display(s=None, fullscreen=False, size=None):
    pygame.display.init()
    return DisplayContext(s, fullscreen, size)
```

- [ ] **Step 5: Rewrite the module docstring**

Replace the whole docstring at the top of `src/app/config.py` (everything between the first `"""` and its closing `"""`) with:

```python
"""Display setup and the constants that depend on the pixel density.

Measured on a 16" MacBook Pro (macOS 26, pygame-ce 2.5.8, SDL 2.32, cocoa
driver) on 2026-09-18::

    panel                                   3456 x 2234
    get_desktop_sizes()                     1728 x 1117   logical points
    Window((1152, 720), allow_high_dpi)     1152 x 720    points
    renderer.coordinates_from_window(...)   2304 x 1440   device pixels
    fullscreen                              1728 x 1084   points, 3456 x 2168 pixels

**Only the renderer sees device pixels.**  A window's *surface* is always in
points, on every SDL2 platform, so drawing into ``Window.get_surface()`` could
never be sharper than one pixel per point.  The frame is therefore drawn into
our own surface at ``window * s`` and uploaded into a streaming ``Texture``
each frame (3.4 ms at 2304x1440; measured 7.8 ms for a whole realistic frame).
The original pygame (not the -ce fork) cannot set ``SDL_WINDOW_ALLOW_HIGHDPI``
at all -- the string does not occur in its source -- which is why the
dependency is pygame-ce.

**``s`` is internal pixels per point**, ``native`` is what the display has.
It used to mean the opposite -- points per pixel, the coarse "pixel grid" of
the pixel-art era.  ``G`` now toggles between native and 1: half the pixels is
the escape hatch when the physics budget is saturated at fullscreen.

**Text is sized in points** because points are the unit the platform is
calibrated in; ``font_px`` multiplies by ``s``.

**A fullscreen window is not the desktop size.**  The menu-bar/notch strip
costs 33 points of height, so the buffer is derived from the window we
actually got, after the fact, in ``_adopt``.

**Nothing here may call ``pygame.Window.from_display_module()``.**  The object
it hands back destroys the real SDL window when it is garbage-collected, and
the program segfaults some frames later inside ``pygame.event.get``, a long
way from the call that did it.  This context creates the one ``Window`` for
the process itself, keeps it for its whole life, and destroys it in ``close``.
"""
```

- [ ] **Step 6: Run the tests**

Run: `./.venv/bin/python -m pytest -q tests/test_config.py tests/test_window.py 2>&1 | tail -5`
Expected: all pass. (Other test files will be broken until Tasks 3-4; do not run the whole suite yet.)

- [ ] **Step 7: Commit**

```bash
git add src/app/config.py tests/test_config.py tests/test_window.py
git commit -m "feat: the display is a high-DPI Window, a Renderer and a streaming Texture

s is pixels per point now, native or 1; the software surface is unchanged.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Wire the app to the new context -- resize events, the frame poll, shutdown, probe

**Files:**
- Modify: `src/app/input.py:22-31`
- Modify: `src/app/main.py:23-42, 84-92, 135-139`
- Modify: `tests/test_smoke.py:177-185, 340-345`
- Test: `tests/test_input_resize.py` (new)

**Interfaces:**
- Consumes: `DisplayContext.resize`, `size_changed`, `close`, `native`, `s` from Task 2.

- [ ] **Step 1: Write the failing resize-event test**

Create `tests/test_input_resize.py`:

```python
"""A resize reaches the context and the camera through the event it actually
arrives as.  ``pygame.Window`` posts WINDOWSIZECHANGED; the old VIDEORESIZE
never fires for it, and a listener for the wrong one leaves the interface laid
out against a size the window no longer has."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest

from src.app.config import DisplayContext
from src.app.input import handle_event
from src.app.state import AppState
from src.render.camera import Camera
from src.render.draw import ViewSettings
from src.sim import Simulation


@pytest.fixture
def ctx():
    pygame.display.init()
    context = DisplayContext(size=(800, 600))
    yield context
    context.close()
    pygame.display.quit()


def _state(ctx):
    return AppState(Simulation(), Camera(ctx.width, ctx.height), ViewSettings())


def test_a_window_size_changed_event_resizes_the_context_and_the_camera(ctx):
    state = _state(ctx)
    event = pygame.event.Event(pygame.WINDOWSIZECHANGED, {"x": 640, "y": 480})
    handle_event(event, state, ctx)
    assert ctx.window == (640, 480)
    assert (state.camera.width, state.camera.height) == (640, 480)


def test_a_size_changed_event_is_consumed_before_the_interface(ctx):
    """Nothing else may see it: every control was laid out against the old size."""
    class Ui:
        def on_event(self, *_):
            raise AssertionError("the interface must not see a resize")
    event = pygame.event.Event(pygame.WINDOWSIZECHANGED, {"x": 700, "y": 500})
    handle_event(event, _state(ctx), ctx, Ui())
    assert ctx.window == (700, 500)


def test_the_old_videoresize_event_is_ignored(ctx):
    state = _state(ctx)
    event = pygame.event.Event(pygame.VIDEORESIZE, {"size": (640, 480), "w": 640, "h": 480})
    handle_event(event, state, ctx)
    assert ctx.window == (800, 600)
```

- [ ] **Step 2: Run it to see it fail**

Run: `./.venv/bin/python -m pytest -q tests/test_input_resize.py 2>&1 | tail -5`
Expected: the first two fail (`ctx.window` still `(800, 600)`).

- [ ] **Step 3: Listen for WINDOWSIZECHANGED in `src/app/input.py`**

Replace lines 22-31 (the comment and the `VIDEORESIZE` branch) with:

```python
    # Before the interface, not after: every control was laid out against the
    # old size, and one of them would otherwise get a click meant for a rectangle
    # that has already moved.  A pygame.Window reports its new size as
    # WINDOWSIZECHANGED (``x``, ``y``); VIDEORESIZE never fires for it.
    if event.type == pygame.WINDOWSIZECHANGED:
        if ctx is not None:
            ctx.resize((event.x, event.y))
            state.on_resize(ctx)
        return
```

- [ ] **Step 4: Update `src/app/main.py`**

Replace the `probe` function (lines 23-42) with:

```python
def probe(fullscreen=False):
    """Report what SDL actually gives us.  Answers the Retina question."""
    pygame.init()
    ctx = open_display(fullscreen=fullscreen)
    info = {
        "desktop_size": pygame.display.get_desktop_sizes(),
        "window_size": ctx.window,
        "driver": pygame.display.get_driver(),
        "pixels_per_point": ctx.native,
        "s": ctx.s,
        "buffer": (ctx.width, ctx.height),
        "font_px": ctx.font_px,
        "resizable": ctx.resizable,
        "window_preset": ctx.window_size_label,
    }
    for key, value in info.items():
        print("%-16s %s" % (key, value))
    ctx.close()
    pygame.quit()
    return info
```

Change the import line `from .config import DEFAULT_S, open_display` to `from .config import open_display`.

Replace the safety-net poll (the comment block and `if` at lines 84-92) with:

```python
        # Safety net for a resize that reached us through a path we do not
        # listen for -- a maximise, a macOS tile, a window moved by script.
        # One cheap SDL call a frame, and a no-op whenever nothing moved.
        if self.ctx.size_changed():
            self.ctx.resize()
            self.state.on_resize(self.ctx)
```

Replace `run` with:

```python
    def run(self):
        while self.state.running:
            self.step_once()
        self.ctx.close()
        pygame.quit()
        return 0
```

- [ ] **Step 5: Fix the smoke tests that speak the old event and the old grid**

In `tests/test_smoke.py`, replace `test_the_grid_can_be_changed_live` (lines 177-185) with:

```python
def test_the_density_can_be_changed_live(app):
    for s in (1, 2, 1):
        app.ctx._pixels_per_point = 2
        app.ctx.set_grid(s)
        app.state.on_resize(app.ctx)
        app._build_render_stack()
        for _ in range(3):
            app.step_once()
        assert app.ctx.s == s
        assert app.camera.width == app.ctx.width
        assert app.ctx.surface.get_size() == (app.ctx.width, app.ctx.height)


def test_a_frame_at_native_retina_density_agrees_with_itself(app):
    """The buffer, the texture and the camera must all be the window times two,
    or present() raises when the upload does not fit."""
    app.ctx._pixels_per_point = 2
    app.ctx.set_grid(2)
    app.state.on_resize(app.ctx)
    app._build_render_stack()
    for _ in range(3):
        app.step_once()
    w, h = app.ctx.window
    assert (app.ctx.width, app.ctx.height) == (2 * w, 2 * h)
    assert (app.ctx.texture.width, app.ctx.texture.height) == (2 * w, 2 * h)
    assert (app.camera.width, app.camera.height) == (2 * w, 2 * h)
```

Replace the `_resize` helper (lines 340-345) with:

```python
def _resize(size):
    return pygame.event.Event(pygame.WINDOWSIZECHANGED, {"x": size[0], "y": size[1]})
```

Also change the `app` fixture's teardown from `pygame.quit()` to:

```python
    instance.ctx.close()
    pygame.quit()
```

and in `test_the_app_boots_onto_the_start_menu` change the `finally: pygame.quit()` to `finally: instance.ctx.close(); pygame.quit()` (two statements on two lines).

- [ ] **Step 6: Run the affected files**

Run: `./.venv/bin/python -m pytest -q tests/test_input_resize.py tests/test_smoke.py tests/test_window.py 2>&1 | tail -5`
Expected: all pass. If `test_smoke.py` fails inside `ui.build` on `ctx.s != 1` or `usable_grids`, that is Task 4's job -- note it and continue.

- [ ] **Step 7: Commit**

```bash
git add src/app/input.py src/app/main.py tests/test_smoke.py tests/test_input_resize.py
git commit -m "feat: resize arrives as WINDOWSIZECHANGED; the frame polls the Window; close on exit

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: `G` is the resolution toggle; layout tests run at both densities

**Files:**
- Modify: `src/app/ui.py:82` (VIEW_BUTTONS entry), `:475` (flags)
- Modify: `src/app/state.py:719-730` (`cycle_grid` docstring)
- Modify: `src/render/hud.py:49`
- Modify: `tests/test_layout.py:49-56`
- Modify: `tests/test_ui.py` (only if a test names the tooltip "Chunkier pixels" -- grep first)
- Modify: `tests/test_layering.py` (add one test)

- [ ] **Step 1: Add the layering test for the forbidden call**

Append to `tests/test_layering.py`:

```python
def test_nothing_calls_from_display_module():
    """``pygame.Window.from_display_module()`` hands back an object that
    destroys the real window when it is garbage-collected; the program then
    segfaults some frames later, far from the call that did it."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "src"
    offenders = [p for p in root.rglob("*.py") if "from_display_module" in p.read_text()]
    assert offenders == []
```

- [ ] **Step 2: Rename the control's tooltip and key help**

In `src/app/ui.py` line 82 change `("view.grid", "grid", "", "G", "Chunkier pixels"),` to:

```python
    ("view.grid", "grid", "", "G", "Standard or Retina resolution"),
```

In `src/render/hud.py` line 49 change `("G", "pixel grid size"),` to `("G", "resolution -- Retina or standard"),`.

In `src/app/ui.py` line 475 leave `"view.grid": ctx is not None and ctx.s != 1,` as it is (native density is the "on" state, and it is the only state other than 1).

In `src/app/state.py` replace the `cycle_grid` docstring (lines 720-724) with:

```python
        """Toggle the pixel density.  Sprites are sized in pixels, so the cache dies.

        Cycles what the display actually offers: its native density and 1.
        On a display with one pixel per point there is nothing to cycle and
        the button is disabled.
        """
```

- [ ] **Step 3: Parametrise the layout tests over densities 1 and 2**

In `tests/test_layout.py` replace `cases()` (lines 49-54) with:

```python
def cases():
    for window in WINDOWS:
        for s in (1, 2):
            for tool in TOOLS:
                yield window, s, tool
```

Grep the file for any other `GRID_CHOICES` or `grid_fits` and replace `config.GRID_CHOICES` with `(1, 2)`; a `grid_fits` guard is simply deleted (every density fits every legal window now).

- [ ] **Step 4: Run the whole suite**

Run (600000 ms timeout): `./.venv/bin/python -m pytest -q tests 2>&1 | tail -15`
Expected: all pass, count >= 1253. Fix any straggler that still names `GRID_CHOICES`, `DEFAULT_S`, `MIN_FONT_PX`, `grid_fits`, `ctx.display` or `ctx.origin` (grep `tests/` and `src/` for each; the only legitimate remaining mentions are in this plan and the spec).

- [ ] **Step 5: Commit**

```bash
git add -A src tests
git commit -m "feat: G toggles Retina and standard resolution; layout holds at both densities

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Run the real app at Retina and record the phase-1 screenshots; update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (display invariants)
- Create: `docs/superpowers/plans/screenshots/` is **not** committed -- screenshots go to the scratchpad only.

- [ ] **Step 1: Take headless screenshots with the shot script**

Write this script to the scratchpad directory named in your system prompt as `shots.py` (adjust `OUT`), then run it with `./.venv/bin/python shots.py`:

```python
import os, sys
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
OUT = sys.argv[1]
import pygame
from src.app.main import App
app = App(saves_dir=os.path.join(OUT, "saves"))
app.ctx._pixels_per_point = 2
app.ctx.resize((1152, 720))
app.ctx.set_grid(2)
app.state.on_resize(app.ctx)
app._build_render_stack()
app.state.start_new()
for _ in range(90):
    app.step_once()
pygame.image.save(app.ctx.surface, os.path.join(OUT, "shot_default.png"))
w = app.state.sim.world
for name, radii in (("Earth", 6.0), ("Sun", 4.0), ("Jupiter", 3.0)):
    idx = [i for i in range(w.n) if w.bodies[i].name == name][0]
    app.state.camera.frame_body(w, idx, radii=radii)
    for _ in range(5):
        app.step_once()
    pygame.image.save(app.ctx.surface, os.path.join(OUT, "shot_%s.png" % name.lower()))
print(app.ctx.width, app.ctx.height, app.ctx.s)
app.ctx.close(); pygame.quit()
```

Expected: prints `2304 1440 2` and four PNGs of 2304x1440 exist. Look at `shot_default.png` (Read tool): the composition matches the pre-upgrade app -- body list top-left, transport bar bottom, Sun centred -- at twice the pixels, with text the same physical size (the status bar hint fits on one line).

- [ ] **Step 2: Run the app for real for five seconds**

Run: `timeout 5 ./.venv/bin/python -m src.app.main; echo "exit $?"` -- macOS has no `timeout`; use `./.venv/bin/python -m src.app.main & sleep 5; kill %1` instead.
Expected: a window opens and closes with no traceback on stderr.

- [ ] **Step 3: Update the display invariants in CLAUDE.md**

In `CLAUDE.md`, under "### Design decisions worth knowing before changing them":

Replace the bullet beginning `- **`DEFAULT_S = 1`** (`app/config.py`).` with:

```
- **`s` is internal pixels per point, and only `native` or 1 is ever offered** (`app/config.py`).
  The renderer reports the display's density (`coordinates_from_window`); the software
  surface is `window * s`, uploaded into a streaming `Texture` every frame. `G` toggles
  native and 1 -- half the pixels is the escape hatch when the physics budget is saturated
  at fullscreen. Text is sized in points; `font_px` multiplies by `s`. See the module
  docstring for the measured numbers.
```

Replace the bullet beginning `- **The window is `RESIZABLE`, and the upscale is ours, not SDL's**` with:

```
- **The window is a `pygame.Window(allow_high_dpi=True)`, never `display.set_mode`.**
  Only a `Renderer` reaches device pixels; a window surface is in points on every
  platform, which is why the original pygame (no high-DPI flag at all) was replaced by
  pygame-ce. A resize arrives as `WINDOWSIZECHANGED` (`event.x`, `event.y`), not
  `VIDEORESIZE`; `step_once` also polls `ctx.size_changed()` once a frame for the
  resizes no event announces.
```

Replace the bullet beginning `- **Nothing may touch `pygame._sdl2.video.Window`.**` with:

```
- **Nothing may call `pygame.Window.from_display_module()`.** The object it returns
  destroys the real SDL window when it is garbage-collected, and the program segfaults
  some frames later inside `pygame.event.get`. `DisplayContext` creates the one `Window`
  for the process, keeps it for its whole life and destroys it in `close()`;
  `tests/test_layering.py` greps `src/` for the forbidden call.
```

Replace the bullet beginning `- **The interface has a smallest buffer it fits in, and a coarser grid is a divisor**` with:

```
- **The interface has a smallest window it fits in, in points** (`config.MIN_INTERNAL`,
  `min_internal(s)` scales it by the density). `tests/test_layout.py` builds every panel
  of every tool at the floor at both densities and fails if anything lands outside.
```

In the Invariants list, change `- Pixel-size constants scale by `internal_height / scale.REF_HEIGHT` (360); interface sizes come from `widgets.Metrics`, which derives every one of them from the font. Neither may hard-code a pixel count, because the `G` grid switch changes both.` to end `..., because the `G` density toggle changes both.`

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: the display rules after the move to a high-DPI Window

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 2 -- Bodies

### Task 6: `render/shade.py` -- noise, geometry, lighting, and the sphere

**Files:**
- Create: `src/render/shade.py`
- Test: `tests/test_shade.py` (new)

**Interfaces:**
- Produces: `shade.fbm(xs, ys, seed, octaves=4, cells=4, gain=0.5) -> ndarray`, `shade.shade_radius(px) -> int`, `shade.sphere_size(px, atmosphere) -> int`, `shade.sphere(px, lit_angle, tones, style, seed, atmosphere=None) -> Surface`, constants `LIGHT_TILT`, `AMBIENT`, `SHADE_MAX_PX`, `LADDER_START`, `LADDER_RATIO`, `ATMOSPHERE_HALO`. Styles: `"cratered"`, `"banded"`, `"cracked"`, `"turbulent"`, `"terrestrial"`; any other string draws a plain mid-toned ball. `tones` is three RGB tuples `(dark, mid, light)`; for `"terrestrial"` they mean `(ocean, land, cloud)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_shade.py`:

```python
"""The per-pixel sphere shader.  Every assertion is about a pixel, because the
whole point of the module is what the pixels look like."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import math

import numpy as np
import pygame
import pytest

from src.render import shade

TONES = ((0x4A, 0x33, 0x20), (0xB0, 0x80, 0x50), (0xE0, 0xB4, 0x84))
STYLES = ("cratered", "banded", "cracked", "turbulent", "terrestrial")


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    pygame.init()
    yield
    pygame.quit()


def _lum(colour):
    return 0.2126 * colour[0] + 0.7152 * colour[1] + 0.0722 * colour[2]


def _at(surface, dx, dy):
    """The pixel ``dx, dy`` from the centre of a sprite."""
    c = surface.get_width() // 2
    return surface.get_at((c + int(dx), c + int(dy)))


# --- noise ---------------------------------------------------------------------

def test_fbm_is_deterministic_and_in_range():
    xs, ys = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 1, 40))
    a = shade.fbm(xs, ys, seed=3)
    b = shade.fbm(xs, ys, seed=3)
    c = shade.fbm(xs, ys, seed=4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.min() >= 0.0 and a.max() <= 1.0
    assert a.std() > 0.02, "flat noise is no noise"


# --- the ladder ------------------------------------------------------------------

def test_small_sprites_are_shaded_at_their_own_size():
    for px in (1, 2, 7, shade.LADDER_START):
        assert shade.shade_radius(px) == px


def test_big_sprites_are_shaded_at_a_rung_and_never_above_the_cap():
    assert shade.shade_radius(shade.LADDER_START + 1) > shade.LADDER_START
    assert shade.shade_radius(700) == shade.SHADE_MAX_PX
    assert shade.shade_radius(1_000_000) == shade.SHADE_MAX_PX
    # A zoom notch is 18%; the ladder is 25%, so a notch re-uses a rung.
    assert shade.shade_radius(100) == shade.shade_radius(int(100 * 1.18))


# --- the rim ---------------------------------------------------------------------

def test_the_rim_is_antialiased():
    surf = shade.sphere(20, 0.0, TONES, "cratered", seed=1)
    assert surf.get_size() == (41, 41)
    assert _at(surf, 18, 0)[3] == 255
    assert 0 < _at(surf, 20, 0)[3] < 255
    assert 0 < _at(surf, 0, 20)[3] < 255
    # The corners, two pixels outside the disc diagonally, are nothing.
    assert surf.get_at((0, 0))[3] == 0
    assert surf.get_at((40, 40))[3] == 0


def test_a_one_pixel_body_is_still_a_sprite():
    for px in (1, 2, 3):
        surf = shade.sphere(px, 0.0, TONES, "cratered", seed=1)
        assert surf.get_size() == (2 * px + 1, 2 * px + 1)
        assert _at(surf, 0, 0)[3] == 255


# --- lighting --------------------------------------------------------------------

def test_the_lit_side_faces_the_light():
    surf = shade.sphere(30, 0.0, TONES, "cratered", seed=1)
    assert _lum(_at(surf, 18, 0)) > _lum(_at(surf, -18, 0)) * 1.5
    flipped = shade.sphere(30, math.pi, TONES, "cratered", seed=1)
    assert _lum(_at(flipped, -18, 0)) > _lum(_at(flipped, 18, 0)) * 1.5


def test_the_light_direction_is_screen_up_for_pi_over_two():
    """lit_angle is a world angle; +y in the world is *up* on screen."""
    surf = shade.sphere(30, math.pi / 2, TONES, "cratered", seed=1)
    assert _lum(_at(surf, 0, -18)) > _lum(_at(surf, 0, 18)) * 1.5


def test_the_night_side_is_never_black():
    surf = shade.sphere(30, 0.0, TONES, "cratered", seed=1)
    darkest = min(_lum(_at(surf, dx, dy))
                  for dx in range(-27, 28, 3) for dy in range(-27, 28, 3)
                  if dx * dx + dy * dy < 27 * 27)
    assert darkest > 0.0


def test_the_limb_is_darker_than_the_centre():
    surf = shade.sphere(40, 0.0, TONES, "banded", seed=2)
    # Straight above the centre, perpendicular to the light: same n.L on both,
    # so only limb darkening separates them.
    assert _lum(_at(surf, 0, 38)) < _lum(_at(surf, 0, 4))


# --- styles and sizes ---------------------------------------------------------------

@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("px", (3, 40, 600))
def test_every_style_renders_at_every_size(style, px):
    surf = shade.sphere(px, 1.2, TONES, style, seed=7)
    assert surf.get_size() == (2 * px + 1, 2 * px + 1)
    assert _at(surf, 0, 0)[3] == 255


def test_a_style_has_texture():
    """Two pixels at the same lighting must differ, or the albedo is flat."""
    surf = shade.sphere(60, 0.0, TONES, "cratered", seed=7)
    ring = [_at(surf, 0, dy)[:3] for dy in range(-40, 41, 5)]
    assert len(set(ring)) > 3


def test_the_same_seed_gives_the_same_picture_and_a_different_seed_does_not():
    a = shade.sphere(24, 0.3, TONES, "cracked", seed=5)
    b = shade.sphere(24, 0.3, TONES, "cracked", seed=5)
    c = shade.sphere(24, 0.3, TONES, "cracked", seed=6)
    assert pygame.image.tobytes(a, "RGBA") == pygame.image.tobytes(b, "RGBA")
    assert pygame.image.tobytes(a, "RGBA") != pygame.image.tobytes(c, "RGBA")


def test_an_unknown_style_is_a_plain_ball_not_an_error():
    surf = shade.sphere(10, 0.0, TONES, "nonsense", seed=1)
    assert _at(surf, 0, 0)[3] == 255


# --- atmosphere --------------------------------------------------------------------

def test_an_atmosphere_adds_a_halo_outside_the_rim():
    px = 30
    bare = shade.sphere(px, 0.0, TONES, "cratered", seed=1)
    hazy = shade.sphere(px, 0.0, TONES, "cratered", seed=1, atmosphere=(120, 170, 255))
    assert hazy.get_width() > bare.get_width()
    assert hazy.get_width() == shade.sphere_size(px, True)
    assert _at(hazy, px + 1, 0)[3] > 0
    assert _at(hazy, 0, 0)[3] == 255


def test_the_atmosphere_tints_the_limb():
    px = 40
    bare = shade.sphere(px, 0.0, TONES, "cratered", seed=1)
    hazy = shade.sphere(px, 0.0, TONES, "cratered", seed=1, atmosphere=(120, 170, 255))
    assert _at(hazy, 37, 0)[2] > _at(bare, 37, 0)[2] + 10, "the rim should go blue"
```

- [ ] **Step 2: Run to see them fail**

Run: `./.venv/bin/python -m pytest -q tests/test_shade.py 2>&1 | tail -3`
Expected: `ModuleNotFoundError: No module named 'src.render.shade'`.

- [ ] **Step 3: Create `src/render/shade.py`**

```python
"""Per-pixel shaded spheres and stars, in numpy.

A body is a lit sphere seen from above the plane of the system.  The light
comes from the nearest star, in the plane -- which, taken literally, is a
permanent half-moon: the terminator is a straight line through the centre of
every disc.  So the light is **tilted ``LIGHT_TILT`` toward the viewer**, which
lights about seventy percent of the disc and gives the terminator a curve.  It
is a cheat of exactly the kind the drawn-radius floor is: the direction stays
honest, the picture reads.

**The size ladder.**  A zoom notch is 18%, and a sprite is keyed by its pixel
radius, so a zoom from 100 px to 700 px would regenerate a body a dozen times.
Above ``LADDER_START`` the shader runs at the next rung of a 25% ladder and
the result is scaled to the requested size (``shade_radius``); a notch then
lands on the rung it already has.  ``SHADE_MAX_PX`` caps the rung: a 384 px
sphere is 590,000 pixels and takes ~180 ms once, and above that ``smoothscale``
cannot be told from the real thing.

Everything here is vectorised over the whole sprite; nothing loops over
pixels in Python.  Surfaces come from ``pygame.image.frombuffer(...).copy()``
-- never ``convert_alpha`` -- because once the window is a ``pygame.Window``
there is no display surface to convert to.
"""

import math

import numpy as np
import pygame

#: Degrees the light is lifted out of the plane toward the viewer.
LIGHT_TILT = 40.0
#: Night-side floor, as a fraction of full light.  Never black.
AMBIENT = 0.08
#: Terminator softness: n.L runs from this to the next value while the light
#: fades in.
TERMINATOR_LO = -0.08
TERMINATOR_HI = 0.25
#: Limb darkening exponent on the view-facing component of the normal.
LIMB_POWER = 0.6
#: Largest radius the shader runs at; bigger sprites are shaded here and scaled.
SHADE_MAX_PX = 384
#: Above this radius a sprite is shaded at the next rung of a geometric ladder.
LADDER_START = 48
LADDER_RATIO = 1.25
#: Atmosphere rim strength on the disc, and the halo's width as a fraction of
#: the radius.
ATMOSPHERE_STRENGTH = 0.55
ATMOSPHERE_HALO = 0.09


# --- noise --------------------------------------------------------------------

def _lattice(n, seed):
    """A seeded n x n lattice of random values in 0..1."""
    return np.random.default_rng(int(seed) & 0xFFFFFFFF).random((n, n))


def value_noise(xs, ys, seed, cells):
    """Smooth value noise sampled at the coordinate grids ``xs``, ``ys``
    (0..1).  ``cells`` lattice points per unit; wraps, so a band profile
    tiles."""
    lattice = _lattice(cells, seed)
    fx = (xs * cells) % cells
    fy = (ys * cells) % cells
    x0 = fx.astype(int) % cells
    y0 = fy.astype(int) % cells
    x1 = (x0 + 1) % cells
    y1 = (y0 + 1) % cells
    tx = fx - np.floor(fx)
    ty = fy - np.floor(fy)
    tx = tx * tx * (3 - 2 * tx)
    ty = ty * ty * (3 - 2 * ty)
    a = lattice[y0, x0]
    b = lattice[y0, x1]
    c = lattice[y1, x0]
    d = lattice[y1, x1]
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty


def fbm(xs, ys, seed, octaves=4, cells=4, gain=0.5):
    """Fractal Brownian motion: octaves of value noise, each finer and fainter.
    Values in 0..1."""
    total = np.zeros_like(xs, dtype=np.float64)
    amplitude = 1.0
    norm = 0.0
    for k in range(octaves):
        total += amplitude * value_noise(xs, ys, seed + 101 * k, cells << k)
        norm += amplitude
        amplitude *= gain
    return total / norm


def _smoothstep(lo, hi, x):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _rgb(tone):
    return np.asarray(tone, dtype=np.float64)


def _lerp3(a, b, t):
    """Blend two RGB tones by a per-pixel weight ``t`` of shape (h, w)."""
    a, b = _rgb(a), _rgb(b)
    return a[None, None, :] + (b - a)[None, None, :] * t[..., None]


def _toward(colour, tone, t):
    """Move a per-pixel colour array toward one tone by weight ``t``."""
    return colour + (_rgb(tone)[None, None, :] - colour) * t[..., None]


# --- the ladder ---------------------------------------------------------------

def shade_radius(px):
    """The radius the shader actually runs at for a request of ``px``."""
    px = max(1, int(px))
    if px <= LADDER_START:
        return px
    rung = LADDER_START
    while rung < px:
        rung = int(math.ceil(rung * LADDER_RATIO))
    return min(rung, SHADE_MAX_PX)


def sphere_size(px, atmosphere):
    """Side of the square surface ``sphere`` returns: the disc plus, with an
    atmosphere, room for the halo."""
    px = max(1, int(px))
    pad = int(math.ceil(ATMOSPHERE_HALO * px)) + 1 if atmosphere else 0
    return 2 * (px + pad) + 1


# --- the sphere ---------------------------------------------------------------

def _geometry(px, pad):
    """Per-pixel distance, unit-sphere coordinates and coverage alpha for a
    disc of radius ``px`` centred in a square with ``pad`` extra pixels."""
    half = px + pad
    coords = np.arange(2 * half + 1, dtype=np.float64) - half
    xs, ys = np.meshgrid(coords, coords)
    r = np.hypot(xs, ys)
    alpha = np.clip(px + 0.5 - r, 0.0, 1.0)
    u = np.clip(xs / px, -1.0, 1.0)
    v = np.clip(ys / px, -1.0, 1.0)
    z = np.sqrt(np.clip(1.0 - u * u - v * v, 0.0, 1.0))
    return r, u, v, z, alpha


def _lighting(u, v, z, lit_angle):
    """Diffuse light per pixel, 0..1, with the terminator softened and the
    night side held at AMBIENT.  ``v`` grows downward on screen while the
    world's y grows upward, hence the minus."""
    tilt = math.radians(LIGHT_TILT)
    lx = math.cos(lit_angle) * math.cos(tilt)
    ly = -math.sin(lit_angle) * math.cos(tilt)
    lz = math.sin(tilt)
    ndotl = u * lx + v * ly + z * lz
    diffuse = _smoothstep(TERMINATOR_LO, TERMINATOR_HI, ndotl) * np.clip(ndotl, 0, 1)
    limb = np.power(z, LIMB_POWER)
    return (AMBIENT + (1.0 - AMBIENT) * diffuse) * limb


def _albedo(u, v, z, style, tones, seed):
    """Surface colour per pixel before lighting: (h, w, 3) floats, 0..255."""
    dark, mid, light = tones
    # Texture coordinates: a longitude from u with the z term stretching the
    # texture toward the limb like a real globe, and a latitude from v.
    lon = (np.arctan2(u, np.maximum(z, 1e-6)) / math.pi + 1.0) * 0.5
    lat = (v + 1.0) * 0.5

    if style == "cratered":
        base = fbm(lon, lat, seed, octaves=4, cells=6)
        colour = _lerp3(dark, mid, _smoothstep(0.3, 0.7, base))
        rng = np.random.default_rng(seed + 7)
        for _ in range(int(rng.integers(6, 13))):
            cx, cy = rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)
            radius = rng.uniform(0.04, 0.14)
            d = np.hypot(u - cx, v - cy) / radius
            floor = _smoothstep(1.0, 0.6, d)
            rim = _smoothstep(0.75, 1.0, d) * _smoothstep(1.25, 1.0, d)
            colour = colour * (1 - 0.3 * floor)[..., None]
            colour = _toward(colour, light, 0.22 * rim)
        return colour

    if style == "banded":
        warp = (fbm(lon, lat, seed + 3, octaves=3, cells=3) - 0.5) * 0.12
        profile = fbm(np.zeros_like(lat), lat * 0.5 + warp, seed, octaves=4, cells=6)
        turb = fbm(lon * 2.0, lat * 2.0 + warp, seed + 11, octaves=3, cells=8)
        t = np.clip(profile * 0.8 + turb * 0.2, 0, 1)
        colour = _lerp3(dark, mid, _smoothstep(0.25, 0.55, t))
        upper = _lerp3(mid, light, _smoothstep(0.55, 0.85, t))
        colour = colour + (upper - colour) * _smoothstep(0.5, 0.6, t)[..., None]
        # One storm oval.
        rng = np.random.default_rng(seed + 5)
        sx, sy = rng.uniform(-0.6, 0.6), rng.uniform(-0.5, 0.5)
        d = np.hypot((u - sx) / 0.22, (v - sy) / 0.12)
        storm_tone = _rgb(light) * 0.6 + _rgb(dark) * 0.4
        return _toward(colour, storm_tone, _smoothstep(1.0, 0.6, d))

    if style == "cracked":
        base = fbm(lon, lat, seed, octaves=3, cells=4)
        colour = _lerp3(dark, mid, 0.6 + 0.4 * base)
        ridge = 1.0 - np.abs(fbm(lon, lat, seed + 17, octaves=4, cells=8) * 2 - 1)
        return _toward(colour, light, _smoothstep(0.86, 0.96, ridge))

    if style == "turbulent":
        cells = fbm(lon, lat, seed, octaves=3, cells=10)
        colour = _lerp3(mid, light, _smoothstep(0.45, 0.7, cells))
        return _toward(colour, dark, _smoothstep(0.45, 0.3, cells))

    if style == "terrestrial":
        ocean, land, cloud = tones
        height = fbm(lon, lat, seed, octaves=5, cells=4)
        landmask = _smoothstep(0.5, 0.56, height)
        mountain = _smoothstep(0.66, 0.78, height)
        colour = _lerp3(ocean, land, landmask)
        colour = _toward(colour, _rgb(land) * 0.55, mountain * landmask)
        shallow = _smoothstep(0.44, 0.5, height) * (1 - landmask)
        colour = _toward(colour, _rgb(ocean) * 1.4, 0.5 * shallow)
        caps = _smoothstep(0.8, 0.9, np.abs(v)) * (0.6 + 0.4 * height)
        colour = _toward(colour, cloud, caps)
        clouds = _smoothstep(0.55, 0.75, fbm(lon * 1.5, lat, seed + 23, octaves=4, cells=6))
        return _toward(colour, cloud, 0.55 * clouds)

    # Unknown style: a plain mid-toned ball.
    return np.broadcast_to(_rgb(mid), u.shape + (3,)).copy()


def to_surface(rgb, alpha):
    """(h, w, 3) colour floats and (h, w) alpha floats 0..1 to a SRCALPHA
    surface that owns its pixels."""
    h, w = alpha.shape
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return pygame.image.frombuffer(rgba.tobytes(), (w, h), "RGBA").copy()


def sphere(px, lit_angle, tones, style, seed, atmosphere=None):
    """A shaded planet or moon of radius ``px``, lit from ``lit_angle``
    (a world angle: 0 is +x, pi/2 is up on screen).  Shaded at
    ``shade_radius(px)`` and scaled to ``sphere_size(px, atmosphere)``."""
    px = max(1, int(px))
    shade_px = shade_radius(px)
    pad = int(math.ceil(ATMOSPHERE_HALO * shade_px)) + 1 if atmosphere else 0
    r, u, v, z, alpha = _geometry(shade_px, pad)
    rgb = _albedo(u, v, z, style, tones, seed) * _lighting(u, v, z, lit_angle)[..., None]

    if atmosphere is not None:
        atm = _rgb(atmosphere)
        # A Fresnel rim on the disc, brighter on the day side.
        fresnel = np.power(1.0 - z, 3.0)
        light = _lighting(u, v, z, lit_angle)
        day = 0.15 + 0.85 * np.clip(light, 0, 1)
        rgb = rgb + atm[None, None, :] * (ATMOSPHERE_STRENGTH * fresnel * day * alpha)[..., None]
        # A soft halo just outside the rim, carried in alpha, brighter toward
        # the light.
        halo_w = max(1.0, ATMOSPHERE_HALO * shade_px)
        outside = np.clip((r - shade_px + 0.5) / halo_w, 0.0, 1.0)
        halo = (1.0 - outside) ** 2 * (r > shade_px - 0.5) * 0.9
        lx, ly = math.cos(lit_angle), -math.sin(lit_angle)
        side = 0.35 + 0.65 * np.clip((u * lx + v * ly) * 0.5 + 0.5, 0, 1)
        halo = halo * side
        rgb = np.where((alpha > 0)[..., None], rgb, atm[None, None, :])
        alpha = np.maximum(alpha, halo)

    surface = to_surface(rgb, alpha)
    size = sphere_size(px, atmosphere)
    if surface.get_width() != size:
        surface = pygame.transform.smoothscale(surface, (size, size))
    return surface
```

- [ ] **Step 4: Run the tests**

Run: `./.venv/bin/python -m pytest -q tests/test_shade.py 2>&1 | tail -5`
Expected: all pass. If `test_the_atmosphere_tints_the_limb` fails by a small margin, raise `ATMOSPHERE_STRENGTH` to 0.65 and say so in the commit; do not weaken the test.

- [ ] **Step 5: Commit**

```bash
git add src/render/shade.py tests/test_shade.py
git commit -m "feat: per-pixel shaded spheres -- lighting, limb darkening, noise surfaces, atmosphere

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: `shade.star` -- a limb-darkened disc with granulation and a corona

**Files:**
- Modify: `src/render/shade.py` (append)
- Modify: `tests/test_shade.py` (append)

**Interfaces:**
- Produces: `shade.star(px, colour, seed=0) -> Surface` of side `shade.star_size(px)`; constants `GLOW_SCALE = 2.2`, `GLOW_STRENGTH = 0.55`, `STAR_LIMB = 0.4`, `GRANULE_CONTRAST = 0.06`. The surface is fully opaque with the falloff in colour, for `BLEND_ADD`.

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_shade.py`:

```python
# --- stars -----------------------------------------------------------------------

def test_a_star_is_opaque_everywhere_because_it_is_blitted_additively():
    surf = shade.star(20, (255, 244, 220))
    assert surf.get_size() == (shade.star_size(20), shade.star_size(20))
    corner = surf.get_at((0, 0))
    assert corner[3] == 255
    assert corner[:3] == (0, 0, 0), "outside the glow the colour is black, so adding it changes nothing"


def test_the_centre_is_brighter_than_the_limb_and_the_limb_brighter_than_the_corona():
    px = 30
    surf = shade.star(px, (255, 200, 120))
    centre = _lum(_at(surf, 0, 0))
    limb = _lum(_at(surf, px - 2, 0))
    corona = _lum(_at(surf, int(px * 1.5), 0))
    edge = _lum(_at(surf, int(px * shade.GLOW_SCALE), 0))
    assert centre > limb > corona > 0
    assert edge < corona


def test_a_cool_star_is_red_and_a_hot_one_is_blue():
    from src.render.palette import blackbody_rgb
    cool = shade.star(12, blackbody_rgb(3000.0))
    hot = shade.star(12, blackbody_rgb(20000.0))
    c = _at(cool, 9, 0)
    h = _at(hot, 9, 0)
    assert c[0] > c[2]
    assert h[2] > h[0]


def test_a_big_star_is_shaded_at_a_rung_and_scaled():
    surf = shade.star(600, (255, 244, 220))
    assert surf.get_size() == (shade.star_size(600), shade.star_size(600))
    assert _lum(_at(surf, 0, 0)) > 200
```

- [ ] **Step 2: Run to see them fail**

Run: `./.venv/bin/python -m pytest -q tests/test_shade.py -k star 2>&1 | tail -3`
Expected: `AttributeError: module 'src.render.shade' has no attribute 'star'`.

- [ ] **Step 3: Append the star to `src/render/shade.py`**

```python


# --- stars --------------------------------------------------------------------

#: Glow radius as a multiple of a star's disc radius, and the corona's peak
#: brightness just outside the disc as a fraction of the star's colour.
GLOW_SCALE = 2.2
GLOW_STRENGTH = 0.55
#: Limb darkening: the edge of the disc is this fraction of the centre.
STAR_LIMB = 0.4
GRANULE_CONTRAST = 0.06


def star_size(px):
    """Side of the square surface ``star`` returns: disc plus corona."""
    return 2 * (int(max(1, int(px)) * GLOW_SCALE) + 1) + 1


def star(px, colour, seed=0):
    """A star of radius ``px`` in ``colour``, corona included.

    Blit with ``BLEND_ADD``, which **ignores alpha** and adds raw channel
    values.  So the falloff lives in the colour itself and the surface is
    opaque everywhere; encoding the falloff as alpha would produce a solid
    disc that swallows everything within a couple of AU.
    """
    px = max(1, int(px))
    shade_px = shade_radius(px)
    half = int(shade_px * GLOW_SCALE) + 1
    coords = np.arange(2 * half + 1, dtype=np.float64) - half
    xs, ys = np.meshgrid(coords, coords)
    r = np.hypot(xs, ys)
    col = _rgb(colour)

    inside = np.clip(shade_px + 0.5 - r, 0.0, 1.0)
    z = np.sqrt(np.clip(1.0 - (r / shade_px) ** 2, 0.0, 1.0))
    limb = STAR_LIMB + (1.0 - STAR_LIMB) * z
    u = np.clip(xs / shade_px, -1, 1)
    v = np.clip(ys / shade_px, -1, 1)
    lon = (np.arctan2(u, np.maximum(z, 1e-6)) / math.pi + 1.0) * 0.5
    lat = (v + 1.0) * 0.5
    granules = 1.0 + GRANULE_CONTRAST * (fbm(lon, lat, seed, octaves=3, cells=12) * 2 - 1)
    disc = limb * granules * inside
    disc_rgb = col[None, None, :] * disc[..., None]
    # A white-hot core, so the centre saturates whatever the colour.
    core = np.clip((0.72 * shade_px - r) / (0.2 * shade_px), 0, 1) * inside
    disc_rgb = disc_rgb + (255.0 - disc_rgb) * (core * 0.8)[..., None]

    # The corona: an inverse-square falloff outside the disc, with a few soft
    # streamers at seeded angles.
    ratio = shade_px / np.maximum(r, shade_px)
    corona = GLOW_STRENGTH * ratio ** 2 * (r < shade_px * GLOW_SCALE)
    rng = np.random.default_rng(seed + 31)
    angle = np.arctan2(ys, xs)
    streamers = np.zeros_like(r)
    for _ in range(int(rng.integers(3, 6))):
        a0 = rng.uniform(-math.pi, math.pi)
        width = rng.uniform(0.25, 0.5)
        d = np.abs((angle - a0 + math.pi) % (2 * math.pi) - math.pi)
        streamers += np.exp(-(d / width) ** 2)
    corona = corona * (1.0 + 0.3 * streamers) * (1.0 - inside)

    rgb = disc_rgb + col[None, None, :] * corona[..., None]
    surface = to_surface(rgb, np.ones_like(r))
    size = star_size(px)
    if surface.get_width() != size:
        surface = pygame.transform.smoothscale(surface, (size, size))
    return surface
```

- [ ] **Step 4: Run the tests**

Run: `./.venv/bin/python -m pytest -q tests/test_shade.py 2>&1 | tail -3`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/render/shade.py tests/test_shade.py
git commit -m "feat: a star is a limb-darkened disc with granulation and a streamered corona

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: `Body.look` -- the field, the scenario, the save file, and the material rule

**Files:**
- Modify: `src/body.py:89-97`
- Modify: `src/world.py:52-77`
- Modify: `src/scenarios.py:19-49, 74-86`
- Modify: `src/save.py:83-87, 135-139, 208-211`
- Modify: `src/edits.py:257-275`
- Test: `tests/test_looks.py` (new)

**Interfaces:**
- Produces: `Body.look` (str or None, default None); `World.add(..., look=None)`; `PLANETS` / `MOONS` rows gain a trailing look column; save records carry `"look"`; `edits.set_material` and `cycle_material` set `look = None` unless the material is locked.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_looks.py`:

```python
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
    earth = next(b for b in data["bodies"] if b["name"] == "Earth")
    assert earth["look"] == "terrestrial"
    fresh = World()
    save.from_data(data, fresh)
    assert fresh.bodies[_index(fresh, "Earth")].look == "terrestrial"
    for b in data["bodies"]:
        del b["look"]
    older = World()
    save.from_data(data, older)
    assert all(b.look is None for b in older.bodies[:older.n])
```

- [ ] **Step 2: Run to see them fail**

Run: `./.venv/bin/python -m pytest -q tests/test_looks.py 2>&1 | tail -3`
Expected: `AttributeError: 'Body' object has no attribute 'look'` and a `TypeError` on `look=`.

- [ ] **Step 3: Add the field to `Body`**

In `src/body.py`, after the `self.disk = 0.0` line and its comment (line 97), add:

```python
        #: How this body wants to be drawn -- "terrestrial", "mars", ... --
        #: or None to draw from its material.  Pinned by the scenario on the
        #: named planets; ``edits.set_material`` drops it, because the
        #: material is then what the body is made of.  Appearance only, on
        #: the Body beside ``rings`` and ``spin`` because the force kernel
        #: never reads it.  Nothing inherits it: a fragment, a remnant made
        #: by ``world.add`` and a thrown comet all arrive with None.  A body
        #: that *survives* an accretion keeps its own ``Body`` and so keeps
        #: its look -- Earth swallowing a comet is still Earth.
        self.look = None
```

- [ ] **Step 4: Thread it through `World.add`**

In `src/world.py` change the signature to:

```python
    def add(self, name, material, mass, radius, pos, vel,
            kind=KIND_PLANET, seed=0, temperature=None, look=None):
```

and after `body = Body(...)` add `body.look = look`.

- [ ] **Step 5: Pin the looks in the scenario**

In `src/scenarios.py` change the `PLANETS` header comment and rows to:

```python
# name, a(AU), e, lon_peri(deg), mean_lon(deg), mass(Msun), radius(km), material, look
PLANETS = (
    ("Mercury", 0.38709927, 0.20563593, 77.45779628, 252.25032350,
     1.66013e-7, 2439.7, "iron", "mercury"),
    ("Venus", 0.72333566, 0.00677672, 131.60246718, 181.97909950,
     2.44784e-6, 6051.8, "rock", "venus"),
    ("Earth", 1.00000261, 0.01671123, 102.93768193, 100.46457166,
     3.0034896e-6, 6371.0, "rock", "terrestrial"),
    ("Mars", 1.52371034, 0.09339410, -23.94362959, -4.55343205,
     3.22716e-7, 3389.5, "rock", "mars"),
    ("Jupiter", 5.20288700, 0.04838624, 14.72847983, 34.39644051,
     9.5457937e-4, 69911.0, "gas", "jupiter"),
    ("Saturn", 9.53667594, 0.05386179, 92.59887831, 49.95424423,
     2.85885e-4, 58232.0, "gas", "saturn"),
    ("Uranus", 19.18916464, 0.04725744, 170.95427630, 313.23810451,
     4.36625e-5, 25362.0, "ice", "uranus"),
    ("Neptune", 30.06992276, 0.00859048, 44.96476227, -55.12002969,
     5.15139e-5, 24622.0, "ice", "neptune"),
)

# parent, name, a(AU), e, phase(deg), mass(Msun), radius(km), material, look
MOONS = (
    ("Earth", "Moon", 0.00256955, 0.0549, 210.0, 3.6923e-8, 1737.4, "rock", "moon"),
    ("Jupiter", "Io", 0.00281869, 0.0041, 30.0, 4.4919e-8, 1821.6, "rock", None),
    ("Jupiter", "Europa", 0.00448595, 0.0094, 150.0, 2.4139e-8, 1560.8, "rock_ice", None),
    ("Jupiter", "Ganymede", 0.00715525, 0.0013, 260.0, 7.4525e-8, 2634.1, "rock_ice", None),
    ("Jupiter", "Callisto", 0.01258520, 0.0074, 20.0, 5.4110e-8, 2410.3, "rock_ice", None),
    ("Saturn", "Titan", 0.00816766, 0.0288, 95.0, 6.7649e-8, 2574.7, "rock_ice", None),
)
```

In `solar_system`, change the two loops to unpack and pass the look:

```python
    for name, a, e, lon_peri, mean_lon, mass, radius_km, mat, look in PLANETS:
        ...
        world.add(name, materials.get(mat), mass, _km(radius_km), pos, vel, look=look)
```

```python
    for parent, name, a, e, phase, mass, radius_km, mat, look in (MOONS if moons else ()):
        ...
        world.add(name, materials.get(mat), mass, _km(radius_km),
                  world.pos[pi] + rel_pos, world.vel[pi] + rel_vel, look=look)
```

Grep `src/` and `tests/` for any other unpacking of `PLANETS` or `MOONS` (`grep -rn "in PLANETS\|in MOONS\|PLANETS\[\|MOONS\[" src tests`) and give each the extra column.

- [ ] **Step 6: Save and load it**

In `src/save.py` add `"look": b.look,` after `"disk": float(b.disk),` in the record builder; add `"look": None if rec.get("look") is None else str(rec["look"]),` after the `"disk"` line in the parser; add `body.look = p["look"]` after `body.disk = p["disk"]` in the loader.

- [ ] **Step 7: Drop it on a material change**

In `src/edits.py` `set_material`, after the lock check and before `body.material = material`, add:

```python
    # The material is now what the body is made of; a pinned look is a claim
    # about a body that no longer exists.
    body.look = None
```

(`cycle_material` calls `set_material`, so it needs nothing.)

- [ ] **Step 8: Run the tests**

Run: `./.venv/bin/python -m pytest -q tests/test_looks.py tests/test_save.py tests/test_edits.py tests/test_world.py 2>&1 | tail -3`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add src/body.py src/world.py src/scenarios.py src/save.py src/edits.py tests/test_looks.py
git commit -m "feat: a body has a look -- pinned by the scenario, saved, dropped on a material change

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: `sprites.py` draws bodies and stars through the shader; the cache learns the ladder and the look

**Files:**
- Modify: `src/render/sprites.py` (constants, `_make_body`, `SpriteCache`)
- Modify: `src/render/draw.py:73-125`
- Modify: `tests/test_sprites.py`
- Modify: `tests/test_draw.py`

**Interfaces:**
- Produces: `sprites.LOOKS` dict; `sprites.appearance(material, look) -> (tones, style, atmosphere)`; `SpriteCache.get(material, seed, px, lit_angle, is_star, temperature=CMB, look=None)`; `SpriteCache._scaled(key, px, make, size)` (the ladder). `GLOW_SCALE` and `GLOW_STRENGTH` now live in `shade` and are re-exported from `sprites` for the tests that read them.

- [ ] **Step 1: Update the sprite tests**

In `tests/test_sprites.py`:

Replace `test_lighting_makes_one_side_brighter` (lines 79-89) with:

```python
def test_lighting_makes_one_side_brighter(cache):
    from src import materials
    px = 30
    surf = cache.get(materials.get("rock"), seed=1, px=px, lit_angle=0.0, is_star=False)
    lit = surf.get_at((px + 18, px))
    dark = surf.get_at((px - 18, px))
    assert sum(lit[:3]) > sum(dark[:3]) * 1.5
```

Replace `test_tiny_bodies_still_produce_a_surface` with:

```python
def test_tiny_bodies_still_produce_a_surface(cache):
    from src import materials
    for px in (0, 1, 2):
        surf = cache.get(materials.get("rock"), seed=1, px=px,
                         lit_angle=0.0, is_star=False)
        assert surf.get_width() == max(1, px) * 2 + 1
        assert surf.get_at((max(1, px), max(1, px)))[3] == 255
```

Append these tests:

```python
# --- looks -----------------------------------------------------------------------

def test_a_look_changes_the_picture_and_the_cache_key(cache):
    from src import materials
    rock = materials.get("rock")
    plain = cache.get(rock, seed=1, px=20, lit_angle=0.0, is_star=False)
    earth = cache.get(rock, seed=1, px=20, lit_angle=0.0, is_star=False, look="terrestrial")
    assert earth is not plain
    assert pygame.image.tobytes(earth, "RGBA") != pygame.image.tobytes(plain, "RGBA")
    # Earth's disc is wider than a bare rock's: the atmosphere pads it.
    assert earth.get_width() > plain.get_width()


def test_an_unknown_look_draws_from_the_material(cache):
    from src import materials
    rock = materials.get("rock")
    a = cache.get(rock, seed=1, px=20, lit_angle=0.0, is_star=False, look="nonsense")
    b = cache.get(rock, seed=1, px=20, lit_angle=0.0, is_star=False)
    assert pygame.image.tobytes(a, "RGBA") == pygame.image.tobytes(b, "RGBA")


def test_every_look_renders():
    from src import materials
    from src.render.sprites import LOOKS, SpriteCache
    cache = SpriteCache()
    for look in LOOKS:
        surf = cache.get(materials.get("rock"), seed=3, px=16, lit_angle=0.5,
                         is_star=False, look=look)
        assert surf.get_at((surf.get_width() // 2, surf.get_height() // 2))[3] == 255


def test_earth_is_blue(cache):
    from src import materials
    px = 40
    surf = cache.get(materials.get("rock"), seed=3, px=px, lit_angle=0.0,
                     is_star=False, look="terrestrial")
    c = surf.get_width() // 2
    blues = sum(1 for dx in range(-30, 31, 4) for dy in range(-30, 31, 4)
                if (lambda p: p[2] > p[0] + 20)(surf.get_at((c + dx, c + dy))))
    assert blues > 60, "most of a shaded Earth should be ocean"


# --- the ladder ------------------------------------------------------------------

def test_a_big_sprite_is_the_size_it_was_asked_for(cache):
    from src import materials
    surf = cache.get(materials.get("gas"), seed=1, px=500, lit_angle=0.0, is_star=False)
    assert surf.get_size() == (1001, 1001)


def test_two_sizes_on_one_rung_share_the_shaded_base(cache):
    """A zoom notch must not re-shade a 400 px planet."""
    from src import materials
    from src.render import shade
    gas = materials.get("gas")
    cache.get(gas, seed=1, px=300, lit_angle=0.0, is_star=False)
    bases = len(cache._bases)
    cache.get(gas, seed=1, px=int(300 * 1.18), lit_angle=0.0, is_star=False)
    assert shade.shade_radius(300) == shade.shade_radius(int(300 * 1.18))
    assert len(cache._bases) == bases
```

Also in `test_stars_are_larger_than_their_disc_because_of_the_glow`, if it reads `sprites.GLOW_SCALE`, leave it: the constant is re-exported.

- [ ] **Step 2: Update the draw tests**

In `tests/test_draw.py`, find the test that relies on a body under 2 px being drawn with `set_at` (grep `set_at` and `1.5`; if there is none, skip this). Then append:

```python
def test_a_sub_pixel_body_is_a_sprite_not_a_set_at(parts):
    """The shader draws a soft dot at any size; the old single-pixel branch
    would leave the disc aliased at exactly the zoom where most bodies live."""
    renderer, camera, view = parts["renderer"], parts["camera"], parts["view"]
    renderer.cache.clear()
    camera.log_zoom = math.log(1e-3)      # everything is far under a pixel true size
    view.real_size = True                 # so the floor does not rescue them
    renderer.draw(parts["surface"], parts["sim"], camera, view)
    assert len(renderer.cache) > 0
```

(`parts` is the dict fixture at the top of the file; `import math` if the file does not already.)

- [ ] **Step 3: Run to see them fail**

Run: `./.venv/bin/python -m pytest -q tests/test_sprites.py tests/test_draw.py 2>&1 | tail -5`
Expected: failures on `look=`, `_bases`, size 1001, and the sub-pixel test.

- [ ] **Step 4: Rewrite the body path in `src/render/sprites.py`**

Replace the module docstring with:

```python
"""Procedurally drawn bodies.

Nothing here is authored art.  The child creates bodies at arbitrary radii out of
arbitrary materials, so there is no sprite sheet that could cover the space --
every disc is generated on demand and cached.  The pixels come from
``render/shade.py``; this module decides *what* to shade (tones, style,
atmosphere -- from the material, or from the body's look) and owns the cache.

The lighting direction is bucketed to 16 compass points.  A continuous angle
would miss the cache on every frame the camera moves, which would mean
regenerating every sprite, every frame.  Temperature, spin, disk and heading
are bucketed for the same reason.

Big sprites are shaded once per rung of ``shade``'s size ladder and scaled to
each size asked for (``SpriteCache._scaled``); the scaled sprites live in the
LRU and the shaded bases in a small side table.
"""
```

Delete the constants `MID_OFFSET`, `HIGHLIGHT_OFFSET`, `HIGHLIGHT_SCALE`, `DETAIL_THRESHOLD`, `GLOW_SCALE`, `GLOW_STRENGTH` and their comments, and add after the imports:

```python
from . import palette, shade
from .shade import GLOW_SCALE, GLOW_STRENGTH  # noqa: F401 -- read by tests and effects
```

(keep `from .. import units`).

Delete the functions `_shade`, `_draw_detail`, `_mask_to_circle` -- but **keep `_shade`** if `_make_black_hole` or `_make_craft` use it (grep); they do not, so delete it.

Add, after `_blend_toward`:

```python
#: How a named body wants to be drawn: tones, style and atmosphere colour.
#: For "terrestrial" the tones mean (ocean, land, cloud).  A body whose look
#: is not here draws from its material.
LOOKS = {
    "mercury": (((70, 70, 74), (128, 128, 132), (196, 196, 200)), "cratered", None),
    "venus": (((150, 120, 60), (214, 190, 120), (246, 232, 190)), "banded", (232, 214, 160)),
    "terrestrial": (((16, 52, 120), (86, 118, 62), (240, 244, 250)), "terrestrial", (120, 170, 255)),
    "mars": (((92, 40, 22), (176, 92, 54), (232, 190, 160)), "cratered", (220, 170, 130)),
    "moon": (((60, 60, 64), (118, 118, 122), (186, 186, 190)), "cratered", None),
    "jupiter": (((120, 70, 40), (214, 178, 132), (244, 232, 210)), "banded", (200, 180, 150)),
    "saturn": (((150, 120, 70), (214, 190, 130), (240, 226, 190)), "banded", (210, 190, 150)),
    "uranus": (((90, 170, 180), (150, 220, 230), (210, 240, 245)), "banded", (150, 220, 230)),
    "neptune": (((22, 40, 120), (56, 96, 200), (140, 180, 240)), "banded", (90, 130, 230)),
}


def appearance(material, look):
    """What to shade: ``(tones, style, atmosphere)`` for a material and a look."""
    if look in LOOKS:
        return LOOKS[look]
    return material.tones, material.style, None
```

Replace `_make_body` with:

```python
def _make_body(material, seed, px, lit_angle, is_star, glow_k=0.0, look=None):
    tones, style, atmosphere = appearance(material, look)
    dark, mid, light = tones
    if glow_k > 0.0:
        # Its own light, mixed into the light it reflects.  Applied before the
        # star branch as well as after it, so a red dwarf really is red and a
        # blue giant really is blue -- the same Planck locus, the same helper.
        target = palette.blackbody_rgb(glow_k)
        weight = glow_fraction(glow_k)
        dark = _blend_toward(dark, target, weight)
        mid = _blend_toward(mid, target, weight)
        light = _blend_toward(light, target, weight)
    if is_star:
        return shade.star(px, light, seed)
    return shade.sphere(px, lit_angle, (dark, mid, light), style, seed, atmosphere)
```

In `SpriteCache.__init__` add `self._bases = {}` and a class constant `BASE_LIMIT = 48`; in `clear` add `self._bases.clear()`. Add the ladder helper and rewrite `get`:

```python
    def _scaled(self, key, px, make, size):
        """A sprite at ``px`` from a base shaded at the ladder rung.

        ``make(radius)`` shades at that radius; ``size(radius)`` is the side of
        the surface it returns.  Below ``LADDER_START`` the base *is* the
        sprite.  Bases live in a small side table so a zoom notch re-uses the
        rung it already has instead of re-shading a screen-filling planet.
        """
        rung = shade.shade_radius(px)
        base = self._bases.get(key + (rung,))
        if base is None:
            base = make(rung)
            if len(self._bases) >= self.BASE_LIMIT:
                self._bases.pop(next(iter(self._bases)))
            self._bases[key + (rung,)] = base
        if rung == px:
            return base
        side = size(px)
        return pygame.transform.smoothscale(base, (side, side))

    def get(self, material, seed, px, lit_angle, is_star,
            temperature=units.CMB_TEMPERATURE_K, look=None):
        px = max(1, int(px))
        bucket = int(round(lit_angle / (2 * math.pi) * LIT_BUCKETS)) % LIT_BUCKETS
        heat = glow_key(temperature)
        look = look if look in LOOKS else None
        key = (material.key, seed, px, bucket, bool(is_star), heat, look)
        hit = self._hit(key)
        if hit is not None:
            return hit
        angle = bucket * 2 * math.pi / LIT_BUCKETS
        glow = heat * GLOW_TEMP_STEP
        atmosphere = appearance(material, look)[2]
        if is_star:
            size = shade.star_size
        else:
            def size(radius):
                return shade.sphere_size(radius, atmosphere)
        surface = self._scaled(
            (material.key, seed, bucket, bool(is_star), heat, look), px,
            lambda radius: _make_body(material, seed, radius, angle, is_star, glow, look),
            size)
        return self._store(key, surface)
```

- [ ] **Step 5: Update `src/render/draw.py`**

In `WorldRenderer.draw`, delete the branch

```python
            if px < 2.0:
                if 0 <= int(x) < width and 0 <= int(y) < height:
                    surface.set_at((int(x), int(y)), body.material.tones[2])
            elif px > MAX_SPRITE_HEIGHTS * height:
```

so that it reads `if px > MAX_SPRITE_HEIGHTS * height:`, and in the final `else` pass the look:

```python
                sprite = self.cache.get(body.material, body.seed, int(px),
                                        nearest_star_angle(world, i), is_star,
                                        float(world.temperature[i]),
                                        look=body.look)
```

Add `from ..body import KIND_BLACK_HOLE, KIND_CRAFT, KIND_STAR` is already there; nothing else changes yet (rings are Task 10).

- [ ] **Step 6: Run the tests**

Run: `./.venv/bin/python -m pytest -q tests/test_sprites.py tests/test_draw.py tests/test_effects.py tests/test_smoke.py 2>&1 | tail -5`
Expected: all pass. `test_sprite_silhouette_is_circular` (alpha 0 beyond `px + 1.5`) must still pass for a material with no atmosphere -- if it fails, the coverage alpha is wrong, not the test.

- [ ] **Step 7: Commit**

```bash
git add src/render/sprites.py src/render/draw.py tests/test_sprites.py tests/test_draw.py
git commit -m "feat: bodies and stars are shaded spheres; the cache learns looks and the size ladder

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Rings in two halves, with the planet's shadow

**Files:**
- Modify: `src/render/sprites.py` (`ring_surface` -> `ring_surfaces`, `SpriteCache.get_rings`)
- Modify: `src/render/draw.py` (the `body.rings` branch)
- Modify: `tests/test_sprites.py:123-129`

**Interfaces:**
- Produces: `sprites.ring_surfaces(px, tones, seed, lit_angle) -> (back, front)`, both `SRCALPHA`, side `sprites.ring_size(px)`; `SpriteCache.get_rings(px, tones, seed, lit_angle) -> (back, front)`. Constants `RING_INNER = 1.4`, `RING_OUTER = 2.3`, `RING_GAP = 1.95`, `RING_SQUASH = 0.30`.

- [ ] **Step 1: Replace the ring test**

Replace `test_ring_surface_is_wider_than_the_planet` in `tests/test_sprites.py` with:

```python
def test_rings_come_in_two_halves_that_bracket_the_planet():
    from src import materials
    from src.render.sprites import ring_size, ring_surfaces
    px = 20
    back, front = ring_surfaces(px, materials.get("gas").tones, seed=1, lit_angle=0.0)
    assert back.get_size() == front.get_size() == (ring_size(px), ring_size(px))
    assert back.get_width() > 2 * px + 1
    c = back.get_width() // 2
    # The back half is above the centre line and the front half below it.
    assert any(back.get_at((x, c - 4))[3] > 0 for x in range(back.get_width()))
    assert all(back.get_at((x, c + 4))[3] == 0 for x in range(back.get_width()))
    assert any(front.get_at((x, c + 4))[3] > 0 for x in range(front.get_width()))
    assert all(front.get_at((x, c - 4))[3] == 0 for x in range(front.get_width()))


def test_the_ring_has_a_gap():
    from src import materials
    from src.render.sprites import RING_GAP, ring_surfaces
    px = 40
    back, front = ring_surfaces(px, materials.get("gas").tones, seed=1, lit_angle=0.0)
    c = front.get_width() // 2
    # Along the horizontal axis the ring is unsquashed: the gap sits at RING_GAP radii.
    gap = front.get_at((c + int(px * RING_GAP), c + 1))[3]
    band = front.get_at((c + int(px * (RING_GAP + 0.15)), c + 1))[3]
    assert gap < band


def test_the_planet_shadows_the_far_half_of_the_ring():
    """With the light from +x, the anti-star direction is -x: the back half
    is darker there than where it is lit."""
    from src import materials
    from src.render.sprites import ring_surfaces
    px = 40
    back, _ = ring_surfaces(px, materials.get("gas").tones, seed=1, lit_angle=0.0)
    c = back.get_width() // 2
    y = c - int(px * 0.3 * 1.7)
    shadowed = sum(back.get_at((c - int(px * 0.3), y))[:3])
    lit = sum(back.get_at((c + int(px * 1.7), c - 2))[:3])
    assert shadowed < lit


def test_rings_are_cached(cache):
    from src import materials
    a = cache.get_rings(12, materials.get("gas").tones, seed=1, lit_angle=0.2)
    b = cache.get_rings(12, materials.get("gas").tones, seed=1, lit_angle=0.2)
    assert a is b
```

- [ ] **Step 2: Run to see them fail**

Run: `./.venv/bin/python -m pytest -q tests/test_sprites.py -k ring 2>&1 | tail -3`
Expected: `ImportError: cannot import name 'ring_surfaces'`.

- [ ] **Step 3: Replace `ring_surface` in `src/render/sprites.py`**

```python
# --- rings ----------------------------------------------------------------------

#: Inner and outer edge of the ring system, the gap, and the vertical squash,
#: all in planet radii.  Saturn's real rings run 1.24..2.27 radii with the
#: Cassini division at 1.95; the squash is the same near-edge-on view the
#: black hole's disk uses.
RING_INNER = 1.4
RING_OUTER = 2.3
RING_GAP = 1.95
RING_SQUASH = 0.30


def ring_size(px):
    """Side of the square surfaces ``ring_surfaces`` returns."""
    return 2 * (int(math.ceil(max(1, int(px)) * RING_OUTER)) + 2) + 1


def ring_surfaces(px, tones, seed, lit_angle):
    """A flat ring system as two surfaces, ``(back, front)``: the half drawn
    behind the planet and the half drawn in front of it, so the disc can sit
    between them.  The planet's shadow is cut across the back half along the
    anti-star direction.  Shaded at the ladder rung like a sphere."""
    px = max(1, int(px))
    rung = shade.shade_radius(px)
    half = int(math.ceil(rung * RING_OUTER)) + 2
    coords = np.arange(2 * half + 1, dtype=np.float64) - half
    xs, ys = np.meshgrid(coords, coords)
    # Unsquash: a point on the drawn ellipse maps to a radius in ring units.
    rr = np.hypot(xs, ys / RING_SQUASH) / rung
    band = (rr >= RING_INNER) & (rr <= RING_OUTER)
    t = np.clip((rr - RING_INNER) / (RING_OUTER - RING_INNER), 0, 1)
    density = shade.fbm(t, np.zeros_like(t), seed + 41, octaves=3, cells=8)
    alpha = (0.35 + 0.5 * density) * band
    alpha = alpha * (1.0 - shade._smoothstep(0.06, 0.02, np.abs(rr - RING_GAP)))
    # Soft edges, one ring-unit-pixel wide.
    edge = np.minimum(np.clip((rr - RING_INNER) * rung, 0, 1),
                      np.clip((RING_OUTER - rr) * rung, 0, 1))
    alpha = alpha * edge
    dark, mid, light = tones
    rgb = shade._lerp3(mid, light, density)

    # The planet's shadow: a band of the planet's width along the anti-star
    # direction, on the far side only.
    lx, ly = -math.cos(lit_angle), math.sin(lit_angle)
    along = xs * lx + ys * ly
    across = np.abs(xs * -ly + ys * lx)
    shadow = (along > 0) * shade._smoothstep(rung * 1.05, rung * 0.9, across)
    back_rgb = rgb * (1.0 - 0.75 * shadow)[..., None]

    back_alpha = alpha * (ys < 0)
    front_alpha = alpha * (ys >= 0)
    back = shade.to_surface(back_rgb, back_alpha)
    front = shade.to_surface(rgb, front_alpha)
    size = ring_size(px)
    if back.get_width() != size:
        back = pygame.transform.smoothscale(back, (size, size))
        front = pygame.transform.smoothscale(front, (size, size))
    return back, front
```

Add to `SpriteCache`:

```python
    def get_rings(self, px, tones, seed, lit_angle):
        """Both halves of a ring system, cached under one key."""
        px = max(1, int(px))
        bucket = int(round(lit_angle / (2 * math.pi) * LIT_BUCKETS)) % LIT_BUCKETS
        key = ("rings", tuple(tones), int(seed), px, bucket)
        hit = self._hit(key)
        if hit is not None:
            return hit
        angle = bucket * 2 * math.pi / LIT_BUCKETS
        return self._store(key, ring_surfaces(px, tones, seed, angle))
```

Add `import numpy as np` to the module imports if it was removed with `_draw_detail` (it is still needed here).

- [ ] **Step 4: Draw the two halves in `src/render/draw.py`**

Replace the ring branch inside the final `else`:

```python
            else:
                angle = nearest_star_angle(world, i)
                rings = None
                if body.rings and px >= 4:
                    rings = self.cache.get_rings(int(px), body.material.tones,
                                                 body.seed, angle)
                    back = rings[0]
                    surface.blit(back, back.get_rect(center=(int(x), int(y))))
                sprite = self.cache.get(body.material, body.seed, int(px),
                                        angle, is_star,
                                        float(world.temperature[i]),
                                        look=body.look)
                rect = sprite.get_rect(center=(int(x), int(y)))
                surface.blit(sprite, rect,
                             special_flags=pygame.BLEND_ADD if is_star else 0)
                if rings is not None:
                    front = rings[1]
                    surface.blit(front, front.get_rect(center=(int(x), int(y))))
```

and remove `from .sprites import ring_surface`.

- [ ] **Step 5: Run the tests**

Run: `./.venv/bin/python -m pytest -q tests/test_sprites.py tests/test_draw.py tests/test_smoke.py 2>&1 | tail -3`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/render/sprites.py src/render/draw.py tests/test_sprites.py
git commit -m "feat: Saturn's rings in two halves, with a gap and the planet's shadow

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: Supersample the black hole and the craft

**Files:**
- Modify: `src/render/sprites.py` (`_make_black_hole`, `_make_craft`, `get_black_hole`, `get_craft`)
- Modify: `tests/test_sprites.py`, `tests/test_black_hole_render.py` (only if a pixel probe fails -- see Step 4)

**Interfaces:**
- Produces: `sprites.SUPERSAMPLE = 2`; `_make_black_hole(px, spin, disk, seed, ss=1)` and `_make_craft(px, heading, ss=1)` take the factor for their one-pixel strokes; `sprites._downsample(surface, ss)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sprites.py`:

```python
# --- supersampling ----------------------------------------------------------------

def test_the_black_hole_is_supersampled_and_lands_at_its_own_size(cache):
    from src.render import sprites
    px = 12
    big = sprites._make_black_hole(px * sprites.SUPERSAMPLE, 0.5, 0.8, 1,
                                   ss=sprites.SUPERSAMPLE)
    small = cache.get_black_hole(seed=1, px=px, spin=0.5, disk=0.8)
    assert small.get_width() < big.get_width()
    assert small.get_width() % 2 == 1, "an odd side keeps the centre on a pixel"
    c = small.get_width() // 2
    assert small.get_at((c, c))[:3] == (0, 0, 0)


def test_a_supersampled_rim_is_antialiased(cache):
    """Somewhere on the photon ring there is a pixel that is neither black nor
    the full ring colour -- the whole point of shading at 2x."""
    px = 14
    surf = cache.get_black_hole(seed=1, px=px, spin=0.0, disk=0.0)
    c = surf.get_width() // 2
    shades = {sum(surf.get_at((c + dx, c + dy))[:3]) for dx in range(-px - 2, px + 3)
              for dy in range(-px - 2, px + 3)}
    partial = [s for s in shades if 40 < s < 700]
    assert partial, "every rim pixel is either black or full: no antialiasing happened"


def test_the_craft_is_supersampled_but_keeps_its_hull_colours(cache):
    from src.render.sprites import _CRAFT_HULL, _CRAFT_TILE
    sprite = cache.get_craft(14, 0.0)
    seen = {sprite.get_at((x, y))[:3] for x in range(sprite.get_width())
            for y in range(sprite.get_height()) if sprite.get_at((x, y))[3] > 200}
    assert _CRAFT_HULL in seen
    assert _CRAFT_TILE in seen
```

- [ ] **Step 2: Run to see them fail**

Run: `./.venv/bin/python -m pytest -q tests/test_sprites.py -k supersampled 2>&1 | tail -3`
Expected: `AttributeError: module ... has no attribute 'SUPERSAMPLE'`.

- [ ] **Step 3: Implement**

In `src/render/sprites.py` add near the top, after `LIT_BUCKETS`:

```python
#: The black hole and the craft are drawn from ellipses and polygons, which
#: pygame cannot antialias; they are drawn at this multiple of their size and
#: scaled down instead.
SUPERSAMPLE = 2


def _downsample(surface, ss):
    """Scale a supersampled sprite back down, keeping an odd side so the
    centre stays on a pixel."""
    w, h = surface.get_size()
    half_w = -(-(w // 2) // ss)
    half_h = -(-(h // 2) // ss)
    return pygame.transform.smoothscale(surface, (2 * half_w + 1, 2 * half_h + 1))
```

Change `_make_black_hole(px, spin, disk, seed)` to `_make_black_hole(px, spin, disk, seed, ss=1)`, and inside it make the two one-pixel strokes scale: the far-side arc's width `max(1, px // 5)` stays; the photon ring `pygame.draw.circle(surface, _PHOTON_RING + (255,), centre, px, 1 if detailed else 0)` becomes `..., ss if detailed else 0)`. Change `detailed = px >= BH_DETAIL_PX` to `detailed = px >= BH_DETAIL_PX * ss`.

Change `_make_craft(px, heading)` to `_make_craft(px, heading, ss=1)`; `if px < CRAFT_DETAIL_PX:` becomes `if px < CRAFT_DETAIL_PX * ss:`, `if px >= CRAFT_DETAIL_PX + 4:` becomes `if px >= (CRAFT_DETAIL_PX + 4) * ss:`, and the window line's width `1` becomes `ss`.

In `get_black_hole` replace the `surface = _make_black_hole(...)` line with:

```python
        surface = _downsample(
            _make_black_hole(px * SUPERSAMPLE, spin_bucket / 8.0 * units.MAX_SPIN,
                             disk_bucket / 8.0, seed, ss=SUPERSAMPLE),
            SUPERSAMPLE)
```

In `get_craft` replace `surface = _make_craft(px, bucket * 2 * math.pi / LIT_BUCKETS)` with:

```python
        surface = _downsample(
            _make_craft(px * SUPERSAMPLE, bucket * 2 * math.pi / LIT_BUCKETS,
                        ss=SUPERSAMPLE),
            SUPERSAMPLE)
```

- [ ] **Step 4: Run the sprite, black-hole and draw tests**

Run: `./.venv/bin/python -m pytest -q tests/test_sprites.py tests/test_black_hole_render.py tests/test_draw.py tests/test_craft.py 2>&1 | tail -8`
Expected: all pass. If a black-hole test that probes an exact pixel (`cy - px + 3`, `cx + int(px * 2.2)`) fails because the downscale spread the stroke by one pixel, widen **that probe** by one pixel (e.g. `d in (-2, -1, 0, 1, 2)`), and say which test and why in the commit body. Do not touch a test that fails for any other reason -- report it instead.

- [ ] **Step 5: Commit**

```bash
git add src/render/sprites.py tests/test_sprites.py tests/test_black_hole_render.py
git commit -m "feat: the black hole and the craft are drawn at 2x and scaled down

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 12: Phase-2 screenshots and CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Screenshots**

Run the `shots.py` script from Task 5 again (same command). Look at all four PNGs. Expected: Earth blue with continents and a blue rim, Jupiter banded with a storm and a limb, the Sun a limb-darkened disc with a corona, the system view with soft shaded dots. If a look reads wrong (e.g. Earth mostly green), adjust that look's tones in `LOOKS` and re-run -- tones only, nothing structural.

- [ ] **Step 2: Update CLAUDE.md**

In `CLAUDE.md` "### Design decisions worth knowing before changing them", replace the bullet beginning `- **A hot body's colour comes from the Planck locus, not the palette**` with:

```
- **Every planet, moon and star is a per-pixel shaded sphere** (`render/shade.py`, pure
  numpy). Lambert lighting from the nearest star with the light **tilted 40° toward the
  viewer** -- a true top-down view lit in-plane is a permanent half-moon -- a soft
  terminator, an 8 % ambient night side, limb darkening, and a noise albedo per style
  (`cratered`, `banded`, `cracked`, `turbulent`, `terrestrial`). Surfaces come from
  `frombuffer(...).copy()`, never `convert_alpha` (there is no display surface).
- **Big sprites are shaded at a rung of a 25 % size ladder and scaled** (`shade.shade_radius`,
  `SpriteCache._scaled`). A zoom notch is 18 %, so a notch re-uses the rung it has instead
  of re-shading a screen-filling planet; `SHADE_MAX_PX` (384) caps the rung.
- **A body's look is appearance, not substance** (`Body.look`, `sprites.LOOKS`). The
  scenario pins one on each named planet so Earth is blue although its material is rock;
  `edits.set_material` drops it, because the material is then what the body is made of.
  Nothing inherits it; a body that survives an accretion keeps its own `Body` and so its look.
- **A hot body's colour comes from the Planck locus, not the palette**
  (`render/palette.blackbody_rgb`). Temperature is quantised to 100 K before it
  reaches the sprite cache key, for the same reason the lighting angle is bucketed
  to 16 points. The blend runs before the star branch too, so a red dwarf is red and a
  blue giant is blue off the same helper -- `shade.star` takes the blended colour.
- **The black hole and the craft are supersampled** (`sprites.SUPERSAMPLE`): drawn from
  ellipses and polygons pygame cannot antialias, at 2x, and `smoothscale`d down to an odd
  side so the centre stays on a pixel. Still blitted normally, never `BLEND_ADD`.
- **Rings are two surfaces** (`sprites.ring_surfaces`): the far half behind the disc with
  the planet's shadow cut across it, the near half in front.
```

- [ ] **Step 3: Full suite, then commit**

Run (600000 ms timeout): `./.venv/bin/python -m pytest -q tests 2>&1 | tail -5`
Expected: all pass, count >= 1253.

```bash
git add CLAUDE.md
git commit -m "docs: the shading rules -- the light tilt, the size ladder, looks, supersampling, rings

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 3 -- Sky, trails, text, effects, chrome

### Task 13: A pre-rendered, coloured starfield with a Milky Way haze

**Files:**
- Modify: `src/render/starfield.py` (rewrite)
- Modify: `tests/test_draw.py:153-190` (starfield tests) and append

**Interfaces:**
- Produces: `Starfield(width, height, count=None, seed=7)` with `.width`, `.height`, `.count`, `.sky` (a `Surface` of `(width, height)`), `.draw(surface, camera)`; module function `star_count(width, height)`; constants `PARALLAX`, `MIN_STARS = 100`, `MAX_STARS = 3000`.

- [ ] **Step 1: Update and extend the tests**

In `tests/test_draw.py` keep the three existing starfield tests as they are (they pass `count=200` and `seed=42`), and append:

```python
def test_the_sky_is_prerendered_at_the_buffer_size():
    from src.render.starfield import Starfield
    field = Starfield(640, 360)
    assert field.sky.get_size() == (640, 360)


def test_the_star_count_scales_with_the_area_between_a_floor_and_a_cap():
    from src.render.starfield import MAX_STARS, MIN_STARS, star_count
    assert star_count(640, 360) == 300
    assert star_count(1280, 720) == 1200
    assert star_count(64, 36) == MIN_STARS
    assert star_count(6000, 4000) == MAX_STARS


def test_stars_come_in_more_than_one_colour_and_brightness():
    from src.render import palette
    from src.render.camera import Camera
    from src.render.starfield import Starfield
    surface = pygame.Surface((320, 200))
    surface.fill(palette.SPACE)
    Starfield(320, 200, count=300).draw(surface, Camera(320, 200))
    colours = {surface.get_at((x, y))[:3] for x in range(320) for y in range(200)}
    bright = [c for c in colours if max(c) > 200]
    warm = [c for c in colours if c[0] > c[2] + 20 and max(c) > 60]
    cool = [c for c in colours if c[2] > c[0] + 20 and max(c) > 60]
    assert bright, "no bright star"
    assert warm and cool, "a real sky has both red and blue stars"


def test_the_haze_lifts_the_background_above_pure_space():
    """The Milky Way band: somewhere the sky is faintly lighter than SPACE
    without being a star."""
    from src.render import palette
    from src.render.starfield import Starfield
    sky = Starfield(320, 200).sky
    faint = sum(1 for x in range(0, 320, 2) for y in range(0, 200, 2)
                if palette.SPACE[0] < sky.get_at((x, y))[0] < palette.SPACE[0] + 30)
    assert faint > 500
```

- [ ] **Step 2: Run to see them fail**

Run: `./.venv/bin/python -m pytest -q tests/test_draw.py -k "sky or star_count or colour or haze" 2>&1 | tail -3`
Expected: attribute / import errors.

- [ ] **Step 3: Rewrite `src/render/starfield.py`**

```python
"""The background.

Pre-rendered once per buffer size into ``sky`` -- ``App._build_render_stack``
makes a new one whenever the buffer changes shape, so nothing here caches a
size across frames -- and blitted up to four times a frame at the wrapped
parallax offset.  That replaces three hundred ``set_at`` calls with four
blits, and lets the sky hold things a per-pixel loop never could: stars of
several brightnesses and colours, and a faint band of Milky Way.

Real stars are so far away they would not parallax at all across a solar
system, but a very small factor gives a depth cue when panning without
implying they are nearby.  Positions come from a fixed seed so the sky does
not reshuffle itself every time the window is resized.

Stars are laid out across exactly one screen and the offset wraps modulo the
screen, so the sky tiles seamlessly and every star is always on screen.
"""

import numpy as np
import pygame

from . import palette, shade

PARALLAX = 0.02
#: Stars per 640x360 of buffer, and the floor and cap on the total.
STARS_PER_REF = 300
MIN_STARS = 100
MAX_STARS = 3000
#: The haze is rendered at this fraction of the buffer and scaled up: at
#: 3456x2168 a full-resolution fbm is a quarter of a second per resize.
HAZE_SCALE = 4
HAZE_BRIGHTNESS = 22


def star_count(width, height):
    """How many stars a buffer of this size gets."""
    n = int(round(STARS_PER_REF * width * height / (640.0 * 360.0)))
    return max(MIN_STARS, min(MAX_STARS, n))


class Starfield:
    def __init__(self, width, height, count=None, seed=7):
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.count = star_count(self.width, self.height) if count is None else int(count)
        self.seed = seed
        self.sky = self._render(np.random.default_rng(seed))

    def _render(self, rng):
        w, h = self.width, self.height
        sky = pygame.Surface((w, h))
        sky.fill(palette.SPACE)

        # The Milky Way: a diagonal band of low-frequency noise.
        hw, hh = max(2, w // HAZE_SCALE), max(2, h // HAZE_SCALE)
        xs, ys = np.meshgrid(np.linspace(0, 1, hw), np.linspace(0, 1, hh))
        angle = rng.uniform(0.4, 1.1)
        across = (xs - 0.5) * np.sin(angle) - (ys - 0.5) * np.cos(angle)
        band = np.exp(-(across / 0.18) ** 2)
        noise = shade.fbm(xs, ys, self.seed + 3, octaves=3, cells=3)
        haze = band * (0.4 + 0.6 * noise) * HAZE_BRIGHTNESS
        rgb = np.empty((hh, hw, 3), dtype=np.float64)
        rgb[..., 0] = palette.SPACE[0] + haze * 1.0
        rgb[..., 1] = palette.SPACE[1] + haze * 0.95
        rgb[..., 2] = palette.SPACE[2] + haze * 0.9
        small = pygame.image.frombuffer(
            np.clip(rgb, 0, 255).astype(np.uint8).tobytes(), (hw, hh), "RGB").copy()
        sky.blit(pygame.transform.smoothscale(small, (w, h)), (0, 0))

        # Stars: a power law of brightness, blackbody colours half-desaturated.
        xy = rng.random((self.count, 2)) * (w, h)
        u = rng.random(self.count)
        brightness = 40 + 215 * u ** 3
        kelvin = np.exp(rng.uniform(np.log(3000.0), np.log(15000.0), self.count))
        for (x, y), value, temp in zip(xy, brightness, kelvin):
            r, g, b = palette.blackbody_rgb(float(temp))
            grey = (r + g + b) / 3.0
            r, g, b = (grey + (r - grey) * 0.5), (grey + (g - grey) * 0.5), (grey + (b - grey) * 0.5)
            k = float(value) / 255.0
            colour = (int(r * k), int(g * k), int(b * k))
            xi, yi = int(x) % w, int(y) % h
            if value > 215:
                radius = 1 if value < 240 else 2
                halo = tuple(int(c * 0.3) for c in colour)
                pygame.draw.aacircle(sky, halo, (xi, yi), radius + 1)
                pygame.draw.aacircle(sky, colour, (xi, yi), radius)
            else:
                sky.set_at((xi, yi), colour)
        return sky

    def draw(self, surface, camera):
        shift = camera.centre * camera.zoom * PARALLAX
        ox = int(shift[0]) % self.width
        oy = int(-shift[1]) % self.height
        w, h = self.width, self.height
        for dx in (-ox, w - ox):
            for dy in (-oy, h - oy):
                if dx < surface.get_width() and dy < surface.get_height() \
                        and dx + w > 0 and dy + h > 0:
                    surface.blit(self.sky, (dx, dy))
```

- [ ] **Step 4: Run the tests**

Run: `./.venv/bin/python -m pytest -q tests/test_draw.py tests/test_smoke.py 2>&1 | tail -3`
Expected: all pass, including the three older starfield tests. If `test_starfield_wraps_seamlessly_when_panning` fails, the `% self.width` sign convention is wrong -- a star at sky `(x, y)` must land at `((x - shift_x) % w, (y + shift_y) % h)` as the old code had it.

- [ ] **Step 5: Commit**

```bash
git add src/render/starfield.py tests/test_draw.py
git commit -m "feat: a pre-rendered sky -- coloured stars by brightness, a Milky Way haze, four blits a frame

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 14: Antialiased trails, previews and markers

**Files:**
- Modify: `src/render/draw.py:184-187` (`_draw_trails`)
- Modify: `src/app/tools.py` -- every `pygame.draw.lines`, `pygame.draw.line`, and `pygame.draw.circle(..., 1)` at lines 463, 465, 476, 477, 491, 1532, 1591, 1606-1608, 1813
- Test: `tests/test_draw.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_draw.py`:

```python
def test_trails_are_antialiased(parts):
    """A diagonal trail must contain intermediate shades, not only its own
    colour and the background."""
    from src.render import palette
    surface, sim, camera = parts["surface"], parts["sim"], parts["camera"]
    world = sim.world
    for _ in range(40):
        sim.advance(2.0)
        sim.update_trails()
    surface.fill(palette.SPACE)
    parts["renderer"]._draw_trails(surface, world, camera,
                                   surface.get_width(), surface.get_height())
    colours = {surface.get_at((x, y))[:3] for x in range(surface.get_width())
               for y in range(surface.get_height())}
    colours.discard(palette.SPACE)
    assert len(colours) > 20, "aalines blend the line into the background at every step"
```

- [ ] **Step 2: Run to see it fail**

Run: `./.venv/bin/python -m pytest -q tests/test_draw.py -k antialiased 2>&1 | tail -3`
Expected: fails with a small colour count (eight fade chunks only).

- [ ] **Step 3: Switch the draw calls**

In `src/render/draw.py` `_draw_trails`, change `pygame.draw.lines(surface, colour, False, projected[start:stop].astype(int).tolist(), 1)` to:

```python
                pygame.draw.aalines(surface, colour, False,
                                    projected[start:stop].astype(int).tolist())
```

In `src/app/tools.py`, for each listed line: `pygame.draw.lines(surface, colour, bool(bound), pts, 1)` -> `pygame.draw.aalines(surface, colour, bool(bound), pts)`; `pygame.draw.line(surface, c, a, b, 1)` -> `pygame.draw.aaline(surface, c, a, b)`; `pygame.draw.circle(surface, c, centre, r, 1)` -> `pygame.draw.aacircle(surface, c, centre, r, 1)`. Leave filled circles (no width argument) alone. Read each call before editing: the integer-overflow guards around the preview lines stay exactly as they are.

- [ ] **Step 4: Run the tool and draw tests**

Run: `./.venv/bin/python -m pytest -q tests/test_draw.py tests/test_tools_launch.py tests/test_tools_fly.py tests/test_tools_select.py tests/test_tools_edit.py tests/test_smoke.py 2>&1 | tail -3`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/render/draw.py src/app/tools.py tests/test_draw.py
git commit -m "feat: trails, orbit previews and markers are antialiased

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 15: A real font, antialiased

**Files:**
- Already present: `assets/fonts/AtkinsonHyperlegible-Regular.ttf`, `assets/fonts/OFL.txt` (added to the working tree but not yet committed)
- Modify: `src/render/pixelfont.py` (rewrite)
- Modify: `tests/test_pixelfont.py`

**Interfaces:**
- Produces: `pixelfont.FONT_PATH` (a `pathlib.Path`), `pixelfont.DEFAULT_FONT_RATIO = 1.9`; `PixelFont(px, path=FONT_PATH)` with the same `height`, `width`, `render`, `draw` interface.

- [ ] **Step 1: Confirm the font files are there**

Run: `ls -la assets/fonts/ && ./.venv/bin/python -c "import pygame; pygame.init(); f = pygame.font.Font('assets/fonts/AtkinsonHyperlegible-Regular.ttf', 11); print(f.get_height())"`
Expected: two files (54348 and 4352 bytes) and `14`.

- [ ] **Step 2: Update the font tests**

In `tests/test_pixelfont.py` replace `test_text_is_not_antialiased` with:

```python
def test_text_is_antialiased():
    """Glyph edges blend into the background: at least one drawn pixel is
    neither the background nor exactly the text colour."""
    from src.render.pixelfont import PixelFont
    font = PixelFont(14)
    surface = pygame.Surface((160, 34))
    surface.fill((0, 0, 0))
    font.draw(surface, 1, 1, "AVW", (255, 255, 255))
    colours = {surface.get_at((x, y))[:3]
               for x in range(160) for y in range(34)}
    assert colours - {(0, 0, 0), (255, 255, 255)}, sorted(colours)


def test_the_bundled_face_is_used_and_sized_in_points():
    from src.render.pixelfont import FONT_PATH, PixelFont
    assert FONT_PATH.exists()
    assert 13 <= PixelFont(11).height <= 16
    assert PixelFont(22).height >= 2 * PixelFont(11).height - 2


def test_a_missing_font_file_falls_back_to_the_built_in_face_at_the_same_height(tmp_path):
    from src.render.pixelfont import PixelFont
    real = PixelFont(11).height
    fallback = PixelFont(11, path=tmp_path / "nope.ttf").height
    assert abs(fallback - real) <= 2
```

- [ ] **Step 3: Run to see them fail**

Run: `./.venv/bin/python -m pytest -q tests/test_pixelfont.py 2>&1 | tail -3`
Expected: `ImportError: cannot import name 'FONT_PATH'` and the antialiasing test fails.

- [ ] **Step 4: Rewrite `src/render/pixelfont.py`**

```python
"""Text rendering.

Named for the pixel-art era, when antialiasing was deliberately off; the
class keeps its name because forty callers know it and the interface --
``height``, ``width``, ``render``, ``draw`` -- is all any of them use, which is
exactly what let the face and the antialiasing change without touching one.

The face is Atkinson Hyperlegible (OFL, ``assets/fonts``), chosen for a child
reading numbers at a glance.  If the file is missing the built-in face stands
in, scaled so the line height matches: pygame sizes its built-in face by an
undocumented 0.6875 and it is a shorter face besides, so a request for 11
points becomes 21 to land on the same 14 px line (measured: 11 -> 14 px
bundled, 21 -> 13 px built-in).
"""

import pathlib

import pygame

FONT_PATH = (pathlib.Path(__file__).resolve().parents[2]
             / "assets" / "fonts" / "AtkinsonHyperlegible-Regular.ttf")
#: The built-in face needs this many times the point size to match the
#: bundled face's line height.
DEFAULT_FONT_RATIO = 1.9


class PixelFont:
    def __init__(self, px, path=FONT_PATH):
        if not pygame.font.get_init():
            pygame.font.init()
        self.px = px
        try:
            self._font = pygame.font.Font(str(path), px)
        except (OSError, pygame.error, FileNotFoundError):
            self._font = pygame.font.Font(None, max(1, round(px * DEFAULT_FONT_RATIO)))
        self.height = self._font.get_height()
        self._cache = {}

    def width(self, text):
        if not text:
            return 0
        return self._font.size(text)[0]

    def render(self, text, colour):
        """Cached, antialiased surface for one string in one colour."""
        key = (text, colour)
        surface = self._cache.get(key)
        if surface is None:
            surface = self._font.render(text, True, colour)
            if len(self._cache) > 512:
                self._cache.clear()
            self._cache[key] = surface
        return surface

    def draw(self, target, x, y, text, colour):
        """Blit ``text`` at ``(x, y)``.  Returns the horizontal advance."""
        if not text:
            return 0
        surface = self.render(text, colour)
        target.blit(surface, (x, y))
        return surface.get_width()
```

- [ ] **Step 5: Run the font, layout, widget, hud and smoke tests**

Run (600000 ms timeout): `./.venv/bin/python -m pytest -q tests/test_pixelfont.py tests/test_layout.py tests/test_widgets.py tests/test_hud.py tests/test_ui.py tests/test_explain_panel.py tests/test_smoke.py 2>&1 | tail -8`
Expected: all pass. If `test_layout.py` reports a control off the edge at `MIN_WINDOW`, the new face is wider somewhere than the layout floor allows: raise `config.MIN_INTERNAL` width by the overflow (keep it under 576 -- see the comment there), re-run, and say so in the commit. Do not change the layout tests.

- [ ] **Step 6: Commit**

```bash
git add assets/fonts src/render/pixelfont.py tests/test_pixelfont.py
git commit -m "feat: Atkinson Hyperlegible, antialiased, with the built-in face as fallback

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 16: Soft impact particles

**Files:**
- Modify: `src/render/effects.py:238-292`
- Modify: `tests/test_effects.py` (append)

**Interfaces:**
- Produces: `effects._soft_disc(radius, colour) -> Surface` (cached, `SRCALPHA`, alpha `(1 - r/R) ** 1.5`), `effects.SOFT_DISC_LIMIT = 256`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_effects.py`:

```python
def test_a_soft_disc_fades_from_the_centre_to_nothing_and_is_cached():
    from src.render.effects import _soft_disc
    disc = _soft_disc(8, (255, 200, 100))
    assert disc.get_size() == (17, 17)
    assert disc.get_at((8, 8))[3] == 255
    assert disc.get_at((0, 0))[3] == 0
    assert 0 < disc.get_at((8, 3))[3] < 255
    assert _soft_disc(8, (255, 200, 100)) is disc


def test_a_flash_is_soft_not_a_hard_disc():
    from src.render import palette
    from src.render.effects import EffectField
    field = EffectField()
    field.spawn(_event(energy=1e-6))
    surface = pygame.Surface((640, 400))
    surface.fill(palette.SPACE)
    field.draw(surface, _camera())
    colours = {surface.get_at((x, y))[:3] for x in range(280, 360) for y in range(160, 240)}
    assert len(colours) > 12, "a soft disc has a gradient; a hard one has two colours"
```

- [ ] **Step 2: Run to see them fail**

Run: `./.venv/bin/python -m pytest -q tests/test_effects.py -k soft 2>&1 | tail -3`
Expected: `ImportError: cannot import name '_soft_disc'`.

- [ ] **Step 3: Implement**

In `src/render/effects.py`, replace `_blit_disc` with:

```python
#: Radial-gradient sprites by (radius, colour).  Bounded: a flash cools through
#: a few dozen colours over its life and the ramp is quantised by int().
SOFT_DISC_LIMIT = 256
_SOFT_DISCS = {}


def _soft_disc(radius, colour):
    """An additive blob: alpha ``(1 - r/R) ** 1.5`` from the centre out."""
    radius = max(1, int(radius))
    key = (radius, tuple(colour))
    disc = _SOFT_DISCS.get(key)
    if disc is None:
        side = 2 * radius + 1
        coords = np.arange(side, dtype=np.float64) - radius
        xs, ys = np.meshgrid(coords, coords)
        r = np.hypot(xs, ys) / radius
        alpha = np.clip(1.0 - r, 0.0, 1.0) ** 1.5
        rgba = np.empty((side, side, 4), dtype=np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = colour[0], colour[1], colour[2]
        rgba[..., 3] = (alpha * 255).astype(np.uint8)
        disc = pygame.image.frombuffer(rgba.tobytes(), (side, side), "RGBA").copy()
        if len(_SOFT_DISCS) >= SOFT_DISC_LIMIT:
            _SOFT_DISCS.clear()
        _SOFT_DISCS[key] = disc
    return disc


def _blit_disc(surface, colour, sx, sy, radius_px):
    """An additive soft disc, clamped to something pygame can draw."""
    r = max(1, min(int(radius_px), 4000))
    rect = pygame.Rect(sx - r, sy - r, r * 2 + 1, r * 2 + 1)
    if not surface.get_rect().colliderect(rect.inflate(4, 4)):
        return
    surface.blit(_soft_disc(r, colour), rect.topleft, special_flags=pygame.BLEND_ADD)
```

In `draw`:
- `"ring"`: `pygame.draw.circle(surface, colour, (sx, sy), max(1, int(px)), 1)` -> `pygame.draw.aacircle(surface, colour, (sx, sy), max(1, int(px)), 1)`.
- `"spark"`: replace the `set_at` and its `if` with `_blit_disc(surface, colour, sx, sy, max(1.0, height / 360.0))` (a two-pixel spark at Retina, one at standard).
- `"plume"`: `pygame.draw.circle(surface, colour, (sx, sy), max(1, int(px)))` -> `_blit_disc(surface, colour, sx, sy, px)`.
- `"flare"`: unchanged (there is no antialiased arc).

- [ ] **Step 4: Run the effects and smoke tests**

Run: `./.venv/bin/python -m pytest -q tests/test_effects.py tests/test_smoke.py 2>&1 | tail -3`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/render/effects.py tests/test_effects.py
git commit -m "feat: impact flashes, sparks and plumes are soft additive discs

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 17: Antialiased icon curves; phase-3 screenshots; final CLAUDE.md; full suite

**Files:**
- Modify: `src/render/widgets.py` -- the `pygame.draw.circle(..., 1)` calls at lines 509, 541, 551, 606, 614, 617, 618 (outline circles only)
- Modify: `CLAUDE.md`

- [ ] **Step 1: Switch the outline circles**

In `src/render/widgets.py`, each `pygame.draw.circle(surface, colour, centre, radius, 1)` (a trailing `, 1` width) becomes `pygame.draw.aacircle(surface, colour, centre, radius, 1)`. Filled circles, arcs and ellipses stay.

Run: `./.venv/bin/python -m pytest -q tests/test_widgets.py tests/test_ui.py tests/test_layout.py 2>&1 | tail -3`
Expected: all pass.

- [ ] **Step 2: Screenshots**

Run the `shots.py` script from Task 5 once more and look at all four. Expected: crisp antialiased text in the new face, a coloured sky with a faint band, smooth trails, shaded bodies.

- [ ] **Step 3: Update CLAUDE.md**

In `CLAUDE.md`, replace the bullet `- **Icons are drawn from lines, not typed as glyphs.** ...` with:

```
- **Icons are drawn from lines, not typed as glyphs.** The font is whatever SDL supplies
  and may not have a pause bar or a chevron; a missing glyph would render as a box. Curved
  strokes use `aacircle`; there is no antialiased arc or ellipse in pygame-ce, so those stay.
- **The sky is pre-rendered** (`render/starfield.py`): one surface per buffer size, rebuilt
  by `_build_render_stack`, blitted up to four times at the wrapped parallax offset. Star
  count scales with area (100..3000), brightness is a power law, colours are half-desaturated
  blackbody, and a Milky Way haze is rendered at a quarter resolution and scaled.
- **Text is Atkinson Hyperlegible, antialiased** (`render/pixelfont.py`, OFL, `assets/fonts`),
  sized in points; the built-in face stands in at the same line height if the file is
  missing. The class keeps its old name because the interface never changed.
- **Impact particles are soft additive discs** (`effects._soft_disc`), cached by radius and
  colour; the wall-clock rule and the velocities are untouched.
```

In the `src/render/` row of the architecture table, add `shade.py` (per-pixel sphere and star shading) to the list of modules.

In the Commands block, change `# whole suite (1253 tests, ~3:40)` to the actual count and time from Step 4.

- [ ] **Step 4: Full suite**

Run (600000 ms timeout): `./.venv/bin/python -m pytest -q tests 2>&1 | tail -5`
Expected: all pass, count >= 1253. Record the count and the time.

- [ ] **Step 5: Run the real app one last time**

Run: `./.venv/bin/python -m src.app.main & sleep 6; kill %1`
Expected: a window opens; no traceback.

- [ ] **Step 6: Commit**

```bash
git add src/render/widgets.py CLAUDE.md
git commit -m "feat: antialiased icon curves; the sky, font and effects rules in CLAUDE.md

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Self-review notes (kept for the executor)

- Spec §1 (display) -> Tasks 1-5. §2.1-2.2 (shader, star) -> Tasks 6-7. §2.3 (looks) -> Tasks 8-9. §2.4 (supersampling) -> Task 11. §2.5 (rings) -> Task 10. §3.1 (sky) -> Task 13. §3.2 (trails) -> Task 14. §3.3 (font) -> Task 15. §3.4 (effects) -> Task 16. §3.5 (chrome) -> Task 17. §4 tests are spread across the tasks that build what they test. §5 rollout is the phase order.
- The spec says "every merge, graze, fragment and tidal debris has `look = None`"; a body that *survives* an accretion keeps its `Body` and therefore its look. That is the intended reading (Earth eating a comet is still Earth) and Task 8's docstring says so.
- `shade.star` takes the glow-blended `light` tone rather than reading `world.temperature` itself, because `_make_body` already does that blend for both branches.
