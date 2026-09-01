# Interface redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the tool palette and the object inspector -- currently on opposite
screen edges -- into one right-hand context column and a bottom hotbar, and give the
flat, text-dense chrome a depth/hierarchy pass, without changing any physics, edit or
tool-contract code.

**Architecture:** `src/app/tools.py`'s `controls()`/`perform()` contract is untouched --
this only changes *where* `src/app/ui.py` lays those controls out and how `src/render/hud.py`
and `src/render/widgets.py` draw. The tool-picker rail merges into the transport bar
(`_build_bottom_hotbar`, always two bottom-anchored rows -- no left rail, no
narrow-window branch to maintain). The inspector's actions move above its stats
(`_build_inspector_actions`, now a fixed 2x2 grid anchored at the top of the column) and
gain a progressive-disclosure collapse (`state.inspector_expanded`, read by
`Hud.inspector_rect`/`visible_rows`). The live tool's own controls
(`_build_tool_context`) anchor below `Hud.inspector_rect(...)` instead of below the old
rail. A thin selection tether and four palette/widget primitives (elevation tier,
`TEXT_FAINT`, `draw_bar`, a 1px press-offset) carry the visual-language refresh.

**Tech Stack:** Python 3, pygame (`SDL_VIDEODRIVER=dummy` in tests), pytest.

**Spec:** `docs/superpowers/specs/2026-08-31-ui-redesign-design.md`

## Global Constraints

- No changes to `src/*.py` physics modules, `src/edits.py`, `src/body.py`, or the
  `Tool.controls()`/`Tool.perform()` contract in `src/app/tools.py`.
- Every existing control `id` stays exactly what it is today (`tool.look`, `body.follow`,
  `edit.step`, etc.) -- `tests/test_ui.py` finds controls by id, and the hotkey-parity
  test in that file reads `hud.KEY_HELP` and must keep passing with **no table changes**,
  since no hotkey moves.
- `tests/test_layout.py` is the floor-fit gate: every control and panel must stay inside
  `config.min_internal(s)` for every grid `s` the window supports. It is expected to be
  the thing that fails first if a task's spacing is too generous -- when it does, tighten
  spacing/collapse defaults in that same task rather than changing the floor constant.
- Micro-animation (the spec's ~120ms open/close interpolation) is **deferred**: during
  brainstorming the user did not flag "no motion/feedback" as one of the things that reads
  as dated (they picked text-density, chrome clutter and flatness), so this plan ships the
  mini-card collapse as an instant toggle and keeps only the cheap, static press-offset.
  This is a scope trim from the committed spec's "Micro-feedback" section, made here for
  YAGNI reasons -- flagged, not silent.
- Full three-tier value/label/**unit** text colouring (the spec's numeric-hierarchy
  paragraph) is **also trimmed**: every `format_mass`/`format_length`/`format_speed`/
  `format_body_temperature` helper returns value and unit fused into one string, and
  several are pinned by exact-string tests in `tests/test_hud.py` (e.g.
  `format_body_temperature(273.15) == "273 K  (0 C)"`). Splitting them would touch every
  one of those helpers and their tests for a cosmetic-only gain. `palette.TEXT_FAINT`
  still ships as a primitive (Task 1) and gets one real, low-risk consumer (Task 6
  recolours the material-blurb row to it). Likewise, the spec's "double-gap between stat
  clusters" is folded into a smaller, zero-cascading-risk stand-in: Task 6 gives
  `palette.PANEL_ROW` a real consumer as the collapsed mini-card's background band instead
  of restructuring every blank-row gap in `inspector_rect`'s height math, which would
  ripple into Tasks 8-10's floor-fit anchoring for a purely cosmetic gain. Widening every
  group gap is left for the same follow-up pass as the formatter rework above.
- Follow the codebase's existing default-no-comments convention; a comment only earns its
  place when it records a non-obvious *why* (see any existing module for the bar to clear).

---

## File structure

| File | Responsibility after this plan |
|---|---|
| `src/render/palette.py` | + one elevation tier, `TEXT_FAINT`, a raised-edge tone |
| `src/render/widgets.py` | + `draw_bar`; press-offset in `_draw_button_content`; `Metrics.transport_h` becomes a fixed two-row height |
| `src/app/state.py` | + `inspector_expanded` flag and its reset-on-select |
| `src/render/hud.py` | + `headline_stats`; `ACTIONS_ROWS`/`ACTIONS_BAND_H`; collapse-aware `visible_rows`/`inspector_rect`; two progress bars, a raised collapsed-card band, a dimmer blurb |
| `src/app/ui.py` | `_build_tool_rail` removed; `_build_bottom_hotbar` (new, absorbs the old rail's tool buttons and the old `_build_transport`); `_build_inspector_actions` repositioned to the top of the column, fixed 2x2; new `_build_inspector_expand`; `_build_tool_context` re-anchored below the inspector; `_build_body_list` re-anchored to the freed left edge; selection tether added to `draw()` |
| `tests/test_layout.py`, `tests/test_ui.py`, `tests/test_hud.py`, `tests/test_widgets.py`, `tests/test_state.py` | extended for the above; no existing test's *assertion intent* changes, only what it needs to keep passing |

No new files -- every change lands in a module that already owns that responsibility.

---

### Task 1: Palette -- elevation tier and numeric-hierarchy colours

**Files:**
- Modify: `src/render/palette.py:29-42`
- Test: `tests/test_widgets.py` (new cases in the existing file)

**Interfaces:**
- Produces: `palette.PANEL_ROW`, `palette.TEXT_FAINT`, `palette.RAISED_EDGE` -- plain
  `(r, g, b)` tuples, consumed by Tasks 2, 3 and 6.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_widgets.py, appended under a new "--- palette tiers ---" section
def test_the_elevation_ladder_only_ever_gets_lighter():
    from src.render import palette
    ladder = [palette.SPACE, palette.PANEL, palette.PANEL_ROW, palette.BUTTON]
    for dim, brighter in zip(ladder, ladder[1:]):
        assert sum(brighter) > sum(dim), "%s is not lighter than %s" % (brighter, dim)


def test_text_faint_is_dimmer_than_text_dim():
    from src.render import palette
    assert sum(palette.TEXT_FAINT) < sum(palette.TEXT_DIM)


def test_raised_edge_is_lighter_than_the_ordinary_button_edge():
    from src.render import palette
    assert sum(palette.RAISED_EDGE) > sum(palette.BUTTON_EDGE)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_widgets.py -k "elevation_ladder or text_faint or raised_edge" -v`
Expected: FAIL with `AttributeError: module 'src.render.palette' has no attribute 'PANEL_ROW'`

- [ ] **Step 3: Add the three colours**

In `src/render/palette.py`, insert between `PANEL_EDGE` (line 12) and `TEXT` (line 13):

```python
#: One elevation tier above PANEL, one below BUTTON -- the background a group
#: of related stat rows sits on, so grouping reads from a colour step instead
#: of only from a blank row.
PANEL_ROW = (19, 24, 39)
```

Insert after `TEXT_DIM` (line 14):

```python
#: A third, dimmer text tier for units ("K", "km/s") -- TEXT is the value,
#: TEXT_DIM is the label, this is what is left over.
TEXT_FAINT = (92, 103, 126)
```

Insert after `BUTTON_EDGE` (line 33), before the `BUTTON_EDGE_ACTIVE` comment:

```python
#: A raised element's own top edge, one step lighter than its resting border --
#: the one-pixel highlight elevation is built from, not a drop shadow.
RAISED_EDGE = (78, 96, 132)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_widgets.py -k "elevation_ladder or text_faint or raised_edge" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/render/palette.py tests/test_widgets.py
git commit -m "feat: an elevation tier and two more text/edge colours"
```

---

### Task 2: `draw_bar` -- the bounded-quantity primitive

**Files:**
- Modify: `src/render/widgets.py` (new function, near `draw_slider`, ~line 312)
- Test: `tests/test_widgets.py`

**Interfaces:**
- Consumes: `palette.BUTTON`, `palette.BUTTON_EDGE` (existing).
- Produces: `draw_bar(surface, rect, fraction, colour, bg=palette.BUTTON)` -- draws a
  clamped 0..1 filled bar in `rect`. Consumed by Task 6.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_widgets.py
def test_a_bar_fills_by_fraction():
    from src.render import palette, widgets
    surface = pygame.Surface((100, 10))
    surface.fill((0, 0, 0))
    widgets.draw_bar(surface, pygame.Rect(0, 0, 100, 10), 0.5, palette.ACCENT)
    lit = [x for x in range(100) if surface.get_at((x, 5))[:3] == palette.ACCENT]
    assert 40 <= len(lit) <= 55


def test_a_bar_clamps_outside_zero_and_one():
    from src.render import palette, widgets
    for fraction, expected_min in ((-1.0, 0), (5.0, 95)):
        surface = pygame.Surface((100, 10))
        surface.fill((0, 0, 0))
        widgets.draw_bar(surface, pygame.Rect(0, 0, 100, 10), fraction, palette.ACCENT)
        lit = sum(1 for x in range(100) if surface.get_at((x, 5))[:3] == palette.ACCENT)
        assert lit >= expected_min or lit == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_widgets.py -k "a_bar" -v`
Expected: FAIL with `AttributeError: module 'src.render.widgets' has no attribute 'draw_bar'`

- [ ] **Step 3: Implement `draw_bar`**

In `src/render/widgets.py`, add after `draw_slider` (after line 310):

```python
def draw_bar(surface, rect, fraction, colour, bg=None):
    """A horizontal filled bar, clamped to 0..1 -- a bounded quantity read at a
    glance (how close to ignition, how close to boiling) instead of only as a
    number nobody has an intuition for.
    """
    pygame.draw.rect(surface, bg if bg is not None else palette.BUTTON, rect)
    frac = min(1.0, max(0.0, fraction))
    if frac > 0.0:
        fill = pygame.Rect(rect.x, rect.y, max(1, int(round(rect.width * frac))),
                           rect.height)
        pygame.draw.rect(surface, colour, fill)
    pygame.draw.rect(surface, palette.BUTTON_EDGE, rect, 1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_widgets.py -k "a_bar" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/render/widgets.py tests/test_widgets.py
git commit -m "feat: draw_bar, a clamped fill primitive for bounded stats"
```

---

### Task 3: Press-down offset

**Files:**
- Modify: `src/render/widgets.py:187-256` (`_draw_button_content`)
- Test: `tests/test_widgets.py`

**Interfaces:**
- Consumes: `palette.RAISED_EDGE` (Task 1); reads `control.state(...)` as
  `_draw_button_content` already does.
- Produces: no new symbol; the visible effect is that a resting or hovered button gets a
  1px lighter top edge (the elevation highlight), and a pressed button renders its
  icon/label/hotkey one pixel down-right of its resting position, fill and border
  otherwise unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_widgets.py
def test_a_resting_button_has_a_raised_top_highlight(font):
    from src.render import palette
    from src.render.widgets import Control, draw_button
    surface = pygame.Surface((40, 40))
    surface.fill((0, 0, 0))
    control = Control("x", label="Follow")
    control.rect = pygame.Rect(4, 4, 32, 20)
    draw_button(surface, font, control)
    top_row = [surface.get_at((x, 4))[:3] for x in range(5, 35)]
    assert palette.RAISED_EDGE in top_row


def test_a_pressed_icon_button_nudges_its_icon_down_and_right(font):
    from src.render.widgets import Control, draw_button

    def icon_bbox(pressed):
        surface = pygame.Surface((40, 40))
        surface.fill((0, 0, 0))
        control = Control("x", icon="plus", compact=True)
        control.rect = pygame.Rect(4, 4, 32, 32)
        draw_button(surface, font, control, hovered=True, pressed=pressed)
        points = [(x, y) for x in range(40) for y in range(40)
                  if surface.get_at((x, y))[:3] != (0, 0, 0)]
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        return min(xs), min(ys)

    rest_x, rest_y = icon_bbox(False)
    down_x, down_y = icon_bbox(True)
    assert (down_x, down_y) == (rest_x + 1, rest_y + 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_widgets.py -k "nudges_its_icon or raised_top_highlight" -v`
Expected: FAIL on both -- `(down_x, down_y) != (rest_x + 1, rest_y + 1)` (icon does not
move today), and `palette.RAISED_EDGE not in top_row` (no highlight line today).

- [ ] **Step 3: Implement the offset and the raised-edge highlight**

In `src/render/widgets.py`, inside `_draw_button_content`, right after
`pygame.draw.rect(surface, edge, rect, 1)` (line 211) and before the `STATE_ACTIVE` stripe
block, insert the highlight:

```python
    if state not in (STATE_DISABLED, STATE_DOWN):
        # The elevation cue: a raised element's own top edge reads one step
        # lighter than the rest of its border.  Skipped when pressed -- a
        # button being pushed in is the one moment it should not look raised.
        pygame.draw.line(surface, palette.RAISED_EDGE,
                         (rect.x + 1, rect.y), (rect.right - 2, rect.y))
```

Then, after the `STATE_ACTIVE` stripe block (after line 217) and before the
disabled-colour block (line 219), insert the press offset:

```python
    # A 1px press-down offset on the drawn content only -- the fill and edge
    # stay exactly where the layout put them, so the button does not visibly
    # resize, but the icon/label/hotkey nudge the way a real key does when
    # struck.  This is the cheapest of the flat-vs-depth fixes: no shadow, no
    # new colour, one moved rectangle.
    content = rect.move(1, 1) if state == STATE_DOWN else rect
```

Then replace every remaining use of `rect` in the function body **from this point on**
(the compact-icon branch, the compact-label branch, the icon+label branch, and the
`draw_hotkey` call) with `content`. Concretely, lines 232-256 become:

```python
    if control.compact:
        if control.icon:
            side = min(icon_side, content.width - 4, content.height - 4)
            icon_rect = pygame.Rect(0, 0, side, side)
            icon_rect.center = content.center
            draw_icon(surface, control.icon, icon_rect, text_colour)
        elif control.label:
            x = content.x + (content.width - font.width(control.label)) // 2
            font.draw(surface, x, content.y + (content.height - font.height) // 2,
                      control.label, text_colour)
        return

    x = content.x + pad
    if control.icon:
        icon_rect = pygame.Rect(x, content.y + (content.height - icon_side) // 2,
                                icon_side, icon_side)
        draw_icon(surface, control.icon, icon_rect, text_colour)
        x += icon_side + gap

    font.draw(surface, x, content.y + (content.height - font.height) // 2,
              control.label, text_colour)

    if control.hotkey:
        draw_hotkey(surface, font, control.hotkey, content,
                    dim=state in (STATE_REST, STATE_DISABLED))
```

(`rect` itself is still used earlier for the fill/edge/active-stripe -- leave those three
blocks alone.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_widgets.py -v`
Expected: PASS, all of `test_widgets.py` (the offset must not break the existing
`test_a_hotkey_cap_is_dropped_rather_than_overflowing` or the icon-in-box tests -- both
already tolerate a 1px shift within `rect.inflate(2, 2)`).

- [ ] **Step 5: Commit**

```bash
git add src/render/widgets.py tests/test_widgets.py
git commit -m "feat: a raised top edge and a 1px press-down offset on every button"
```

---

### Task 4: `AppState.inspector_expanded`

**Files:**
- Modify: `src/app/state.py:41-53` (`__init__`), `:100-102` (`select`)
- Test: `tests/test_state.py`

**Interfaces:**
- Produces: `AppState.inspector_expanded` (bool, default `False`),
  `AppState.toggle_inspector_expanded()`. Consumed by Task 5 (Hud) and Task 9 (the expand
  button).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_state.py, appended
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
```

(`state` fixture already exists at the top of `tests/test_state.py`; if it selects a body
by default, adjust the "new body" assertion to pick a body id different from whatever the
fixture already selected.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_state.py -k "inspector" -v`
Expected: FAIL with `AttributeError: 'AppState' object has no attribute 'inspector_expanded'`

- [ ] **Step 3: Implement**

In `src/app/state.py`, `__init__` (after `self.selected_id = None`, line 44):

```python
        #: Collapsed by default: a headline number is what a fresh selection
        #: earns, not a page of stats.  Not per-body -- nothing asked for that,
        #: and it would need a place to live on Body for a feature nobody
        #: requested.
        self.inspector_expanded = False
```

Replace `select` (line 100-101):

```python
    def select(self, body_id):
        self.selected_id = body_id
        self.inspector_expanded = False
```

Add near it:

```python
    def toggle_inspector_expanded(self):
        self.inspector_expanded = not self.inspector_expanded
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_state.py -v`
Expected: PASS, all of `test_state.py`.

- [ ] **Step 5: Commit**

```bash
git add src/app/state.py tests/test_state.py
git commit -m "feat: the inspector's expand state, collapsed on every new selection"
```

---

### Task 5: `Hud.headline_stats` and collapse-aware `visible_rows`/`inspector_rect`

**Files:**
- Modify: `src/render/hud.py:176-179` (class top), `:303-332` (`visible_rows`,
  `inspector_rect`)
- Test: `tests/test_hud.py`

**Interfaces:**
- Consumes: `state.inspector_expanded` (Task 4).
- Produces: `Hud.headline_stats(world, index) -> list[(label, value)]`,
  `Hud.ACTIONS_ROWS` / `Hud.actions_band_height(m) -> int`. `inspector_rect` and
  `visible_rows` keep their existing signatures but their *output* now depends on
  `state.inspector_expanded`. Consumed by Task 6 (drawing), Task 8 (actions anchor), Task
  10 (tool-context anchor).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hud.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hud.py -k "headline_stats or collapsed" -v`
Expected: FAIL with `AttributeError: 'Hud' object has no attribute 'headline_stats'`

- [ ] **Step 3: Implement**

In `src/render/hud.py`, add near the top of the `Hud` class (after line 179, the `__init__`):

```python
    #: Follow/Fly-to/Stop/Delete now sit above the stats, always as a fixed
    #: 2x2 grid -- see the module-level "Right context column" note in the
    #: design spec for why fixed beats the old one-row-if-it-fits branch.
    ACTIONS_ROWS = 2

    def actions_band_height(self, m):
        """Pixels the fixed action grid reserves above the stats."""
        return (self.ACTIONS_ROWS * m.btn_h + (self.ACTIONS_ROWS - 1) * m.gap
                + 2 * m.gap)

    def headline_stats(self, world, index):
        """The one or two numbers worth seeing before the chevron is clicked.

        Kind-specific, the same split `_rows` already makes: a star has
        nothing more headline-worthy than its mass, a planet's temperature is
        the number a child asks about second, and a black hole's horizon is
        the number that makes it a black hole.
        """
        body = world.bodies[index]
        mass = float(world.mass[index])
        if body.kind == KIND_BLACK_HOLE:
            return [("Mass", format_mass(mass)),
                    ("Horizon", format_length(float(world.radius[index])))]
        if body.kind == KIND_STAR:
            return [("Mass", format_mass(mass))]
        return [("Mass", format_mass(mass)),
                ("Temp", format_body_temperature(float(world.temperature[index])))]
```

Replace `visible_rows` (lines 303-316):

```python
    def visible_rows(self, state, index, height):
        """The inspector rows that fit above the buttons underneath it.

        Collapsed, this is identity plus `headline_stats` -- a glance.
        Expanded, it is the old trimmed-to-fit full readout.  Either way the
        panel that grows under a short window is the *expanded* one, since
        collapsed is already short enough to never need trimming in practice.
        """
        world = state.sim.world
        body = world.bodies[index]
        if not state.inspector_expanded:
            rows = [(body.name, palette.SELECT),
                    ("%s  -  %s" % (body.kind, body.material.name)
                     if body.kind != KIND_BLACK_HOLE else "black hole",
                     palette.TEXT_DIM)]
            rows += [("%-10s %s" % (label, value), palette.TEXT)
                     for label, value in self.headline_stats(world, index)]
            return rows
        rows = self._rows(state, index)
        m = self.metrics(0, height)
        reserved = self.actions_band_height(m) + m.gap
        room = m.content_bottom - m.content_top - reserved - PAD * 2
        return rows[:max(1, room // self.line)]
```

Replace `inspector_rect` (lines 318-332):

```python
    def inspector_rect(self, state, width, height):
        """Where the selected-body panel sits, or None with no selection.

        Starts a fixed `actions_band_height` below the top of the column --
        actions sit above this rect, not below it, so this method no longer
        needs to know anything about the action buttons themselves, only how
        tall their fixed band is.
        """
        index = state.selected_index
        if index is None:
            return None
        m = self.metrics(width, height)
        rows = self.visible_rows(state, index, height)
        panel_w = self.font.width("X") * PANEL_WIDTH_CHARS + PAD * 2
        panel_h = len(rows) * self.line + PAD * 2
        top = m.content_top + self.actions_band_height(m) + m.gap
        return pygame.Rect(width - panel_w - PAD, top, panel_w, panel_h)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hud.py -v`
Expected: PASS, all of `test_hud.py` (existing tests like
`test_the_farthest_row_fits_the_inspector_panel` call `hud._rows(...)` directly, which is
untouched, so they are unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/render/hud.py tests/test_hud.py
git commit -m "feat: a collapsed headline card, and a fixed actions band above it"
```

---

### Task 6: Progress bars, a collapsed-card elevation band, and a dimmer blurb

**Files:**
- Modify: `src/render/hud.py:222-301` (`_rows`), `:334-345` (`_draw_inspector`)
- Test: `tests/test_hud.py`

**Interfaces:**
- Consumes: `widgets.draw_bar` (Task 2), `palette.PANEL_ROW`, `palette.TEXT_FAINT`
  (Task 1), `units.STAR_MASS_THRESHOLD`, `Material.vapour_temperature_k` (both exist
  today).
- Produces: `Hud.bar_specs(world, index) -> list[(row_index, fraction, colour)]`, read by
  `_draw_inspector` to paint a bar beside specific rows. `_rows`'s return shape (a list of
  `(text, colour)` pairs) is **unchanged** -- `tests/test_hud.py`'s many direct callers of
  `hud._rows(...)` must keep working with plain 2-tuples; only the material-blurb row's
  *colour* changes, from `TEXT_DIM` to `TEXT_FAINT`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hud.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hud.py -k "bar_specs or dimmest_row or raised_background" -v`
Expected: FAIL -- `bar_specs` with `AttributeError: 'Hud' object has no attribute
'bar_specs'`; the blurb-colour test with `colour == palette.TEXT_DIM`, not `TEXT_FAINT`;
the band test with `palette.PANEL_ROW not in band` (nothing draws it today).

- [ ] **Step 3: Implement**

In `src/render/hud.py`, add a method to `Hud`, after `_rows` (after line 301):

```python
    def bar_specs(self, world, index, rows):
        """(row_index, fraction, colour) for every row `_rows` gave a bar to.

        Matched by the same text prefix `_rows` already writes, so the two
        cannot drift apart without a row's own label changing too.
        """
        body = world.bodies[index]
        specs = []
        for i, (text, _colour) in enumerate(rows):
            if text.startswith("Temp") and body.kind not in (KIND_STAR, KIND_BLACK_HOLE):
                vapour = body.material.vapour_temperature_k
                frac = float(world.temperature[index]) / vapour if vapour > 0 else 0.0
                specs.append((i, frac, palette.WARN))
            elif "heavier to ignite" in text:
                mass = float(world.mass[index])
                frac = mass / units.STAR_MASS_THRESHOLD if mass > 0 else 0.0
                specs.append((i, frac, palette.ACCENT))
        return specs
```

In the same file, find the material-blurb line at the end of `_rows` (line 300):

```python
        rows += [("", palette.TEXT), (body.material.blurb, palette.TEXT_DIM)]
```

and recolour it -- least important line, dimmest tier:

```python
        rows += [("", palette.TEXT), (body.material.blurb, palette.TEXT_FAINT)]
```

Replace `_draw_inspector` (lines 334-345):

```python
    def _draw_inspector(self, surface, state):
        index = state.selected_index
        if index is None:
            return
        width, height = surface.get_size()
        rect = self.inspector_rect(state, width, height)
        rows = self.visible_rows(state, index, height)
        self._panel(surface, rect)
        if not state.inspector_expanded:
            # The collapsed card is its own, slightly raised elevation tier --
            # one step lighter than the panel it sits in, the same ladder
            # `palette.py` defines for buttons, applied to a group of rows
            # instead of a control.
            pygame.draw.rect(surface, palette.PANEL_ROW, rect.inflate(-4, -4))
        bars = dict((i, (frac, colour)) for i, frac, colour
                   in self.bar_specs(state.sim.world, index, rows)) \
               if state.inspector_expanded else {}
        for k, (text, colour) in enumerate(rows):
            y = rect.y + PAD + k * self.line
            self.font.draw(surface, rect.x + PAD, y, text, colour)
            if k in bars:
                frac, bar_colour = bars[k]
                bar_w = self.font.width("X") * 10
                bar_rect = pygame.Rect(rect.right - PAD - bar_w,
                                       y + (self.line - 4) // 2, bar_w, 4)
                widgets.draw_bar(surface, bar_rect, frac, bar_colour)
```

`bars` is keyed against `rows` (the collapsed-or-expanded list `visible_rows` returned),
but `bar_specs` is computed from the always-full `_rows(...)`. Guard with
`state.inspector_expanded` as shown -- collapsed rows are a different, shorter list whose
indices do not line up with `_rows`'s, and no bar-worthy row is in `headline_stats` anyway
(they show plain values there; the bar is an expanded-only embellishment). Add the import
at the top of `src/render/hud.py`, alongside the existing `from . import palette`:

```python
from . import palette, widgets
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hud.py -v`
Expected: PASS, all of `test_hud.py`.

- [ ] **Step 5: Commit**

```bash
git add src/render/hud.py tests/test_hud.py
git commit -m "feat: ignition/vapour bars, a raised collapsed-card band, a dimmer blurb"
```

---

### Task 7: The bottom hotbar -- tool picker merges into transport

**Files:**
- Modify: `src/render/widgets.py:102-120` (`Metrics.__init__`, `transport_h`)
- Modify: `src/app/ui.py:53-57` (`TOOL_BUTTONS` -- unchanged, kept), `:148-164` (`build`),
  `:188-238` (`_build_tool_rail`, `_build_tool_context` -- `_build_tool_rail` removed
  here; `_build_tool_context` is touched again in Task 10), `:328-361`
  (`_build_transport`)
- Test: `tests/test_layout.py`, `tests/test_ui.py`

**Interfaces:**
- Consumes: `TOOL_BUTTONS` (existing module constant in `ui.py`).
- Produces: `UiLayer._build_bottom_hotbar(state, m, ctx)`, called from `build`. Removes
  `_build_tool_rail`. `_build_transport` gains a required `top` parameter and no longer
  creates its own panel. Every control id this method places (`tool.look`, `tool.throw`,
  `tool.edit`, `panel.bodies`, `scenario.cycle`, `sim.reset`, `time.slower`,
  `time.readout`, `time.faster`, `time.pause`) is unchanged from today.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ui.py, appended
def test_the_tool_picker_and_transport_share_the_bottom_hotbar(ui):
    tool_bottoms = {c.rect.bottom for c in ui.controls
                    if c.id.startswith("tool.") or c.id == "panel.bodies"}
    transport_tops = {c.rect.top for c in ui.controls
                      if c.id in ("sim.reset", "time.pause")}
    # The tool row sits directly above the transport row: its bottom edge
    # (plus one gap) is the transport row's top edge.
    assert max(tool_bottoms) < min(transport_tops)


def test_no_control_sits_on_the_left_edge_any_more(ui):
    """The whole reason for this redesign: the left edge is open canvas."""
    left_band = ui.metrics.pad + ui.metrics.gap * 2
    assert not any(c.rect.left < left_band and c.rect.top > ui.metrics.top_h
                  for c in ui.controls)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ui.py -k "bottom_hotbar or left_edge" -v`
Expected: FAIL -- today's tool buttons sit at `x = m.pad` on the left, well inside
`left_band`, and the tool row is not adjacent to the transport row.

- [ ] **Step 3a: Widen `Metrics.transport_h` to a fixed two-row band**

In `src/render/widgets.py`, `Metrics.__init__` (line 119), replace:

```python
        self.transport_h = self.btn_h + 2 * self.gap
```

with:

```python
        #: Two rows now: the tool picker sits directly above the transport
        #: row, in the same bottom-anchored bar, one gap between them.  Fixed
        #: rather than width-dependent -- see the design spec's "Bottom
        #: hotbar" section for why a conditional one-row/two-row branch was
        #: rejected.
        self.transport_h = 2 * self.btn_h + 3 * self.gap
```

- [ ] **Step 3b: Rewrite `_build_transport` to take an explicit `top` and own no panel**

In `src/app/ui.py`, replace `_build_transport` (lines 328-361):

```python
    def _build_transport(self, state, m, top):
        """Scenario, restart, speed and play -- the lower row of the hotbar.

        Takes its own top rather than reading `m.transport_y` directly, because
        it is now the *second* of two rows in one bar; `_build_bottom_hotbar`
        is what owns the bar's single panel and the tool row above this one.
        """
        paused = state.sim.paused
        speed_w = max(m.font.width(label)
                      for label in state.time_labels) + 4 * m.pad

        row = [
            Control("scenario.cycle", label=state.scenario_label, icon="list",
                    tooltip="Pick what Restart loads"),
            Control("sim.reset", label="Restart", icon="reset", hotkey="F5",
                    tooltip="Put the solar system back how it started"),
            Control("time.slower", icon="slower", hotkey="[", compact=True,
                    repeat=True, tooltip="Slow time down",
                    enabled=state.time_index > 0),
            Control("time.readout", label=state.time_label, readout=True),
            Control("time.faster", icon="faster", hotkey="]", compact=True,
                    repeat=True, tooltip="Speed time up",
                    enabled=state.time_index < state.time_index_max),
            Control("time.pause", label="Play" if paused else "Pause",
                    icon="play" if paused else "pause", hotkey="Space",
                    tooltip="Let time run again" if paused else "Freeze everything",
                    active=paused),
        ]
        widths = [speed_w if c.readout else
                  (m.icon_btn_w if c.compact else m.button_width(c)) for c in row]
        total = sum(widths) + m.gap * (len(row) - 1)

        x = max(m.pad, (m.width - total) // 2)
        for control, w in zip(row, widths):
            self._place([(control, pygame.Rect(x, top, w, m.btn_h))])
            x += w + m.gap
```

- [ ] **Step 3c: Add `_build_bottom_hotbar`, remove `_build_tool_rail`**

In `src/app/ui.py`, delete `_build_tool_rail` entirely (lines 188-214). In its place, add:

```python
    def _build_bottom_hotbar(self, state, m, ctx=None):
        """Tool picker above transport, one bottom-anchored bar.

        Tools moved off the left edge so the canvas has no permanent chrome
        there at all -- the whole point of this redesign.  Bodies travels
        with them: it is a mode toggle (the list open or closed), the same
        kind of thing Look/Throw/Edit are.
        """
        top = m.transport_y
        self._panel(pygame.Rect(m.pad - m.gap, top - m.gap,
                                m.width - 2 * (m.pad - m.gap), m.transport_h))

        tool_row = [Control(cid, icon=icon, hotkey=key, compact=True,
                            tooltip=tip, active=state.tool_name == label)
                   for cid, label, icon, key, tip in TOOL_BUTTONS]
        tool_row.append(Control("panel.bodies", icon="list", hotkey="B",
                                compact=True,
                                tooltip="Every body in the system, by name",
                                active=state.show_body_list))
        tool_w = (len(tool_row) * m.icon_btn_w + (len(tool_row) - 1) * m.gap)
        x = max(m.pad, (m.width - tool_w) // 2)
        for control in tool_row:
            self._place([(control, pygame.Rect(x, top, m.icon_btn_w, m.btn_h))])
            x += m.icon_btn_w + m.gap

        self._build_transport(state, m, top + m.btn_h + m.gap)
```

- [ ] **Step 3d: Wire it into `build`**

In `src/app/ui.py`, `build` (lines 148-164), replace:

```python
        rail = self._build_tool_rail(state, m)
        context_w = self._build_tool_context(state, m, rail)
        self._build_view_buttons(state, m, ctx)
        self._build_zoom_cluster(state, m)
        self._build_transport(state, m)
        self._build_body_list(state, m, rail[0] + max(rail[1], context_w))
        self._build_inspector_actions(state, m)
        self._build_help(state, m)
```

with (leave `_build_tool_context`, `_build_inspector_actions` and `_build_body_list`
as-is for this task -- their signatures change in Tasks 8/9/10, this step only reorders
what already compiles):

```python
        self._build_bottom_hotbar(state, m, ctx)
        self._build_view_buttons(state, m, ctx)
        self._build_zoom_cluster(state, m)
        context_w = self._build_tool_context(state, m, (m.pad, 0, m.content_top))
        self._build_body_list(state, m, m.pad)
        self._build_inspector_actions(state, m)
        self._build_help(state, m)
```

(`(m.pad, 0, m.content_top)` is a placeholder `rail`-shaped tuple so this step alone still
runs; Task 10 replaces it with the real anchor. `_build_body_list`'s second positional
argument changes meaning here from "left edge of the rail" to "the left edge, full stop" --
its body does not need to change in this task since `m.pad` is already what it multiplies
against internally today.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ui.py tests/test_layout.py tests/test_widgets.py -v`
Expected: PASS. If `test_layout.py`'s
`test_the_edit_panel_stays_clear_of_the_transport_bar` now fails because the taller
`transport_h` leaves less room above it, that is the floor-fit gate doing its job --
Task 10 (which shortens the inspector via collapse-by-default) is what is expected to
bring it back under budget; if it is still failing after Task 10, come back and trim
`ACTIONS_ROWS`-adjacent spacing rather than the transport band.

- [ ] **Step 5: Commit**

```bash
git add src/render/widgets.py src/app/ui.py tests/test_ui.py
git commit -m "feat: merge the tool picker into a two-row bottom hotbar"
```

---

### Task 8: Actions move above the stats, as a fixed 2x2 grid

**Files:**
- Modify: `src/app/ui.py:411-446` (`_build_inspector_actions`)
- Test: `tests/test_ui.py`

**Interfaces:**
- Consumes: `Hud.actions_band_height` (Task 5), `hud.PAD`, `hud.PANEL_WIDTH_CHARS`
  (imported).
- Produces: same control ids as today (`body.follow`, `body.flyto`, `body.stop`,
  `body.delete`), now anchored at `m.content_top` instead of below
  `hud.inspector_rect(...)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ui.py, appended
def test_the_action_buttons_sit_above_the_inspector_stats(state, ui):
    body = state.sim.world.bodies[3]
    state.select(body.id)
    ui.build(state, SIZE)
    follow = find(ui, "body.follow")
    stats_top = ui.hud.inspector_rect(state, SIZE[0], SIZE[1]).top
    assert follow.rect.bottom <= stats_top
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ui.py -k "sit_above_the_inspector" -v`
Expected: FAIL -- today `follow.rect.top` is *below* `inspector_rect`, not above it.

- [ ] **Step 3: Implement**

At the top of `src/app/ui.py`, extend the existing hud import:

```python
from ..render.hud import PAD as HUD_PAD, PANEL_WIDTH_CHARS
```

Replace `_build_inspector_actions` (lines 411-446):

```python
    def _build_inspector_actions(self, state, m):
        """Follow/Fly to/Stop/Delete -- always a fixed 2x2 grid, at the very
        top of the right column.  Above the stats, not below them: these are
        what the panel is for, and the stats are reference.
        """
        if state.selected_index is None:
            return
        panel_w = self.font.width("X") * PANEL_WIDTH_CHARS + 2 * HUD_PAD
        left = m.width - panel_w - HUD_PAD
        top = m.content_top

        following = state.camera.follow_id == state.selected_id
        rows = [[
            Control("body.follow", label="Follow", icon="follow", hotkey="F",
                    tooltip="Keep the camera on it", active=following),
            Control("body.flyto", label="Fly to", icon="flyto",
                    tooltip="Zoom right in on it"),
        ], [
            Control("body.stop", label="Stop", icon="stop", hotkey="0",
                    tooltip="Take its speed away and watch it fall"),
            Control("body.delete", icon="trash", hotkey="Del", compact=True,
                    danger=True, tooltip="Remove it from the system"),
        ]]

        height = self.hud.actions_band_height(m) - 2 * m.gap
        self._panel(pygame.Rect(left - m.gap, top - m.gap,
                                panel_w + 2 * m.gap, height + 2 * m.gap))
        y = top
        for group in rows:
            self._place(layout_row(group, left, y, m, panel_w))
            y += m.btn_h + m.gap
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ui.py -v`
Expected: PASS, all of `test_ui.py` (existing action-button tests find controls by id and
click through `on_event`, unaffected by the new position).

- [ ] **Step 5: Commit**

```bash
git add src/app/ui.py tests/test_ui.py
git commit -m "feat: the selected body's actions move above its stats"
```

---

### Task 9: The expand chevron

**Files:**
- Modify: `src/app/ui.py` (`_build_help`-adjacent area for the new builder, `perform`
  around line 646-657 for the branch, `build` to call it)
- Test: `tests/test_ui.py`

**Interfaces:**
- Consumes: `state.inspector_expanded`, `state.toggle_inspector_expanded()` (Task 4),
  `hud.inspector_rect` (Task 5).
- Produces: control id `"inspector.expand"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ui.py, appended
def test_the_expand_chevron_opens_and_closes_the_full_stats(state, ui):
    body = state.sim.world.bodies[3]
    state.select(body.id)
    ui.build(state, SIZE)
    assert not state.inspector_expanded
    short = ui.hud.inspector_rect(state, *SIZE).height

    click_id(ui, state, "inspector.expand")
    assert state.inspector_expanded
    tall = ui.hud.inspector_rect(state, *SIZE).height
    assert tall > short


def test_there_is_no_expand_chevron_without_a_selection(state, ui):
    assert find(ui, "inspector.expand") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ui.py -k "expand_chevron or no_expand_chevron" -v`
Expected: FAIL -- `find(ui, "inspector.expand")` is `None` in both cases (the control does
not exist yet), so the first test fails on `click_id`'s `assert control is not None`.

- [ ] **Step 3: Implement**

In `src/app/ui.py`, add a new builder near `_build_inspector_actions`:

```python
    def _build_inspector_expand(self, state, m):
        """The chevron that opens the full stat groups.  Sits at the top-right
        corner of the inspector panel, the same corner the help panel's own
        close button already claims on *its* panel -- one established place
        for "the button that changes what this panel is showing"."""
        rect = self.hud.inspector_rect(state, m.width, m.height)
        if rect is None:
            return
        control = Control("inspector.expand",
                          icon="up" if state.inspector_expanded else "down",
                          compact=True,
                          active=state.inspector_expanded,
                          tooltip="Fewer details" if state.inspector_expanded
                          else "More detail")
        self._place([(control, pygame.Rect(
            rect.right - m.icon_btn_w - 2, rect.y + 2, m.icon_btn_w, m.icon_btn_w))])
```

In `build`, add the call right after `self._build_inspector_actions(state, m)`:

```python
        self._build_inspector_actions(state, m)
        self._build_inspector_expand(state, m)
```

In `perform`, add a branch beside the other `body.*`/`view.*` branches (near line 617,
next to `view.help`):

```python
        if cid == "inspector.expand":
            state.toggle_inspector_expanded()
            return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ui.py -v`
Expected: PASS, all of `test_ui.py`.

- [ ] **Step 5: Commit**

```bash
git add src/app/ui.py tests/test_ui.py
git commit -m "feat: a chevron button to expand the inspector's full stats"
```

---

### Task 10: Tool controls re-anchor below the inspector; the Bodies list moves to the freed left edge

**Files:**
- Modify: `src/app/ui.py:216-238` (`_build_tool_context`), `:363-409`
  (`_build_body_list`), `build` (the placeholder from Task 7, step 3d)
- Test: `tests/test_ui.py`, `tests/test_layout.py`

**Interfaces:**
- Consumes: `hud.inspector_rect` (Task 5).
- Produces: same control ids as today for both methods (`throw.preset`, `throw.circular`,
  `throw.target`, `edit.step`, `edit.field`, `edit.lock`, `edit.unlock_all`,
  `edit.collapse`, `edit.status`, `panel.bodies.close`, `list.scroll`, `select.body`).
  `_build_tool_context` drops its `rail` parameter in favour of computing its own anchor;
  `_build_body_list` drops its `left_edge` parameter (it now always anchors at `m.pad`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ui.py, appended
def test_the_throw_panel_sits_under_the_inspector_when_something_is_selected(state, ui):
    from src.app.tools import LaunchTool
    body = state.sim.world.bodies[3]
    state.select(body.id)
    state.set_tool(LaunchTool)
    ui.build(state, SIZE)
    preset = find(ui, "throw.preset", payload=0)
    inspector_bottom = ui.hud.inspector_rect(state, *SIZE).bottom
    assert preset.rect.top >= inspector_bottom


def test_the_throw_panel_sits_near_the_top_with_nothing_selected(state, ui):
    from src.app.tools import LaunchTool
    state.set_tool(LaunchTool)
    ui.build(state, SIZE)
    preset = find(ui, "throw.preset", payload=0)
    assert preset.rect.top < SIZE[1] // 3


def test_the_bodies_list_now_opens_on_the_left(state, ui):
    ui.build(state, SIZE)
    row = find(ui, "select.body")
    assert row.rect.centerx < SIZE[0] // 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ui.py -k "throw_panel or bodies_list_now_opens" -v`
Expected: FAIL -- with the Task 7 placeholder, the Throw panel is still anchored under a
hardcoded `(m.pad, 0, m.content_top)` rather than under the inspector, and the body list
was already left-anchored in Task 7 so that third test may already pass; keep it as a
regression guard regardless.

- [ ] **Step 3a: Re-anchor `_build_tool_context`**

Replace `_build_tool_context` (lines 216-238) -- same body-layout logic as today, only the
first four lines (how `x`/`top` are found) change:

```python
    def _build_tool_context(self, state, m):
        """Whatever the live tool offers: throw presets, edit steppers.

        Anchored below the inspector when something is selected (the reading
        path research kept coming back to: tool controls and the thing they
        act on, one column, top to bottom) -- or near the top of the same
        column when nothing is, since Throw needs no selection at all.
        """
        tool = state.tool
        rows = tool.controls(state, m) if tool is not None else []
        if not rows:
            return 0

        panel_w = self.font.width("X") * PANEL_WIDTH_CHARS + 2 * HUD_PAD
        width = max(panel_w, max(
            sum(m.icon_btn_w if c.compact else m.button_width(c) for c in row)
            + m.gap * (len(row) - 1) for row in rows))
        x = m.width - width - HUD_PAD

        inspector = self.hud.inspector_rect(state, m.width, m.height)
        bottom = inspector.bottom if inspector is not None else m.content_top

        top = bottom + m.gap * 4
        height = len(rows) * m.btn_h + (len(rows) - 1) * m.gap + m.gap * 3 + m.line
        self._panel(pygame.Rect(x - m.gap, top - m.gap, width + 2 * m.gap, height),
                    tool.context_title)

        y = top + m.line
        for row in rows:
            self._place(layout_row(row, x, y, m, width))
            y += m.btn_h + m.gap
        return width
```

- [ ] **Step 3b: Re-anchor `_build_body_list` to the left edge unconditionally**

In `_build_body_list` (lines 363-409), replace the signature and the two lines that use
`left_edge`:

```python
    def _build_body_list(self, state, m):
```

```python
        x = m.pad
```

(was `x = left_edge + m.gap * 3`; drop the old parameter entirely from the signature and
from this one call site -- nothing else in the method references `left_edge`.)

- [ ] **Step 3c: Fix the two call sites in `build`**

Replace the Task 7 placeholder line and the body-list call:

```python
        context_w = self._build_tool_context(state, m, (m.pad, 0, m.content_top))
        self._build_body_list(state, m, m.pad)
```

with:

```python
        self._build_tool_context(state, m)
        self._build_body_list(state, m)
```

(`context_w` is no longer returned or used anywhere -- `_build_tool_context` still returns
`width` for symmetry with its old signature, but nothing in `build` needs it now that the
Bodies list anchors at `m.pad` unconditionally rather than past the rail+context width.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ui.py tests/test_layout.py -v`
Expected: PASS. Pay particular attention to
`tests/test_layout.py::test_the_edit_panel_stays_clear_of_the_transport_bar` -- this is
the task where the collapsed-by-default inspector (Task 5) is expected to buy back enough
room under the taller hotbar (Task 7) for the Edit panel to fit above `metrics.transport_y`
again. If it still fails at some grid, reduce `Hud.ACTIONS_ROWS`-derived spacing (the
`m.gap * 4` before `top` in `_build_tool_context`, or the `m.gap` after `actions_band_height`
in `inspector_rect`) rather than changing `config.MIN_INTERNAL`.

- [ ] **Step 5: Commit**

```bash
git add src/app/ui.py tests/test_ui.py
git commit -m "feat: tool controls anchor below the inspector; Bodies opens on the left"
```

---

### Task 11: Selection tether

**Files:**
- Modify: `src/app/ui.py:666-696` (`draw`)
- Test: `tests/test_ui.py`

**Interfaces:**
- Consumes: `state.camera.world_to_screen`, `hud.inspector_rect`, `palette.SELECT`
  (existing selection colour, already used for the on-body ring in `src/render/draw.py:134`
  and for the inspector's name row).
- Produces: no new control -- a drawing-only addition to `UiLayer.draw`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ui.py, appended
def test_a_selected_bodys_tether_reaches_toward_the_inspector(state, ui):
    from src.render import palette
    body = state.sim.world.bodies[3]
    state.select(body.id)
    ui.build(state, SIZE)
    surface = pygame.Surface(SIZE)
    surface.fill(palette.SPACE)
    ui.draw(surface, state)

    inspector = ui.hud.inspector_rect(state, *SIZE)
    # A strip just left of the inspector's left edge, spanning its height:
    # the tether has to cross this strip on its way from the body to the
    # panel, wherever exactly the body sits.
    strip_x = inspector.left - 4
    hit = any(surface.get_at((strip_x, y))[:3] == palette.SELECT
             for y in range(inspector.top, inspector.bottom))
    assert hit


def test_no_tether_without_a_selection(state, ui):
    from src.render import palette
    ui.build(state, SIZE)
    surface = pygame.Surface(SIZE)
    surface.fill(palette.SPACE)
    ui.draw(surface, state)
    # Nothing asserts an absence of palette.SELECT pixels globally (buttons
    # may use it too) -- this just exercises the no-selection path without
    # raising, which is the failure mode a missing `is None` guard produces.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ui.py -k "tether" -v`
Expected: FAIL on `test_a_selected_bodys_tether_reaches_toward_the_inspector` (`hit` is
`False` -- nothing draws there today); the second test currently passes already (there is
nothing to raise), which is fine -- it becomes a real regression guard once Step 3 adds a
code path that touches `state.selected_index`.

- [ ] **Step 3: Implement**

In `src/app/ui.py`, `draw` (lines 666-696), add a call right before the final tooltip
block (after the `if self.modal is not None:` block, before `if hovered is not None:`):

```python
        self._draw_tether(surface, state)
```

Add the method itself, near `draw`:

```python
    def _draw_tether(self, surface, state):
        """A leader line from the selected body to the inspector's corner --
        the "this on screen is that panel" link opposite corners used to
        give away for free.  `palette.SELECT` throughout: it is already the
        selection ring's colour (`render/draw.py`) and the inspector name
        row's colour, so this does not introduce a fourth meaning for a new
        colour, just extends the one selection already has.
        """
        index = state.selected_index
        if index is None:
            return
        inspector = self.hud.inspector_rect(state, self.metrics.width,
                                            self.metrics.height)
        if inspector is None:
            return
        body_pos = state.camera.world_to_screen(state.sim.world.pos[index])
        anchor = (inspector.left, inspector.top + self.metrics.line)
        pygame.draw.line(surface, palette.SELECT,
                         (int(body_pos[0]), int(body_pos[1])), anchor, 1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ui.py -v`
Expected: PASS, all of `test_ui.py`.

- [ ] **Step 5: Commit**

```bash
git add src/app/ui.py tests/test_ui.py
git commit -m "feat: a selection tether from the body to its inspector panel"
```

---

### Task 12: Full-suite verification

**Files:**
- None modified beyond whatever the previous eleven tasks' `Run` steps surfaced.
- Test: the whole suite.

**Interfaces:** none -- this task is a gate, not a feature.

- [ ] **Step 1: Run the whole suite**

Run: `python selftest.py`
Expected: all 800+ tests pass, including the pre-existing hotkey-parity test
(`tests/test_ui.py::test_every_shortcut_is_also_a_button`) with **zero changes** to
`hud.KEY_HELP`, and the layering test
(`tests/test_layering.py`, which asserts `src/*.py` never imports pygame -- untouched by
this plan, since every change here is in `src/app/` or `src/render/`).

- [ ] **Step 2: Manual smoke check**

Run: `python -m src.app.main`

Confirm by eye: the left edge is empty canvas; the tool icons and transport sit as two
rows in one bottom bar; selecting a body shows Follow/Fly-to/Stop/Delete above a
collapsed Mass/Temp card with a chevron; clicking the chevron expands it and a thin line
reaches from the body to the panel; switching to Edit stacks its steppers directly below
the (now shorter, collapsed) inspector rather than on the left; the window still resizes
down to its floor without anything walking off screen.

- [ ] **Step 3: Commit (only if Steps 1-2 required fixes)**

```bash
git add -A
git commit -m "fix: floor-fit and smoke-test fixes found in full-suite verification"
```

(Skip this commit entirely if nothing needed changing -- Task 11's commit is then the last
one this plan produces.)

---

## Post-implementation

After all twelve tasks are green, this plan's work is done. Do not invoke `writing-plans`
again for the Opus evaluation pass mentioned in the original request -- that is a separate,
already-tracked follow-up (compare the finished `src/app/ui.py` / `src/render/hud.py` /
`src/render/widgets.py` / `src/render/palette.py` against
`docs/superpowers/specs/2026-08-31-ui-redesign-design.md` and this plan, not against a new
plan of its own).
