# Interface redesign

The tool palette lives on the left edge; the object inspector lives on the right, in the
opposite corner. Picking a tool and reading what it is about to act on means a full
side-to-side eye movement every time -- the two things that belong together are drawn
apart. On top of that, the chrome is flat-fill with hairline borders and no depth cues, and
the inspector is an undifferentiated stack of text rows with no grouping or hierarchy. This
redesign fixes the split, not just the paint: it consolidates four scattered chrome anchors
(left rail, top-right toggles, bottom-right zoom, bottom-center transport) into three regions
-- a top bar, a bottom hotbar, and a single right-hand context column -- and leaves the whole
left edge as open canvas.

## Why this shape

Research into how modern spatial/creative tools solve the same problem (Figma, Blender,
Photoshop, and directly, Universe Sandbox -- this app's own genre precedent) converged on one
finding that rules out the obvious fix: **panels stay fixed and docked; what changes is that
everything relevant to the current selection lives in one continuous column.** Figma floated
its panels next to the canvas content in 2024 (UI3) and reverted it within months -- floating
chrome occludes the thing it describes and reads as clutter, not polish. Photoshop's
Contextual Task Bar has the same complaint on record: it parks over the region just edited,
and Adobe shipped a "pin position" escape hatch for it. So the fix is not tethering the
inspector to the body's screen position; it is putting the tool's own controls and the
selected body's stats in the *same column*, so there is one reading path instead of two.

Universe Sandbox supplies three more concrete precedents this design follows directly: tools
live in a bottom hotbar, not a side rail; object info is a collapsed mini-card that expands
into full detail on click, rather than a permanent wall of text; and the object's action
buttons sit at the *top* of its panel, next to the actions the panel exists to enable, not
below a page of stats.

**Scope decided during design:** this is a layout and visual-language redesign within the
existing hand-drawn `pygame.Surface` widget system -- no native panels, no user-customizable
docking, no touch/gesture rework beyond what the new hit targets naturally give. `tools.py`'s
`controls()`/`perform()` contract, the hotkey set, and every physics/edit invariant in
`CLAUDE.md` are unchanged; this only moves and restyles how they are drawn.

## Layout regions

Three zones replace the current four-corner scatter:

- **Top bar** (unchanged): elapsed time, speed, running-ratio warning, body count, tool
  name.
- **Bottom hotbar**: the tool picker (Look / Throw / Edit, plus the Bodies-list toggle) and
  the transport controls (scenario cycle, restart, speed steppers, play/pause), merged into
  one bottom-anchored bar.
- **Right context column**: one panel replacing today's separate tool-context panel,
  inspector panel and inspector-actions panel. Holds, top to bottom: selection identity,
  primary actions, the stat mini-card (collapsed by default, expandable), then the active
  tool's own controls.

The zoom cluster keeps its existing bottom-right vertical placement -- that is a universal
map convention independent of the tool/inspector split, not part of the anti-pattern being
fixed. The left edge carries no permanent chrome.

## Right context column

Order, top to bottom, one panel:

1. **Identity** -- name, kind, material (or "black hole"), in the existing `SELECT` /
   `TEXT_DIM` colours.
2. **Actions** -- Follow / Fly-to / Stop / Delete, the same four controls
   `_build_inspector_actions` builds today, moved above the stats because they are what the
   panel is for.
3. **Stat mini-card** -- a headline stat per body kind (star: mass; planet: mass and
   temperature; black hole: mass and horizon) plus a chevron control that expands into the
   existing full row set (`Hud._rows`), now broken into 2-3 whitespace-separated groups
   (composition, orbit, thermal/other) instead of one flat list.
4. **Tool controls** -- whatever the live tool's `controls()` returns (Throw's presets,
   Edit's steppers), stacked directly below, in the same panel.

When Look is active and nothing is selected, the column is empty except a one-line hint.
Section 4 renders whenever a tool provides rows, independent of whether a body is selected
(Throw needs no selection); section 1-3 render whenever a body is selected, independent of
which tool is active. Both can be present at once (Edit tool with a body selected is the
common case).

Collapse state is a single `AppState` flag, not per-body: a fresh selection starts collapsed;
deselecting and selecting again starts collapsed again. No per-body memory -- nothing asked
for it, and it would need a place to live (the `Body` object, following the `locked_fields`
precedent) for a feature nobody requested.

## Bottom hotbar

Tool icons (compact, same hotkeys `1`/`2`/`3`/`B`) sit at the left end of the bar that today
holds only the transport row; scenario/restart/speed/play stay centered. Whether this is one
row or two stacked rows at `config.MIN_INTERNAL`'s floor width (480 internal px) is an
implementation-time measurement, not a spec-level decision -- the constraint is that whatever
the arrangement, it stays anchored to the bottom edge and never reintroduces a left rail. If
it needs a second row, that row still spans the bottom, above the first.

## Bodies list

The `B` toggle now opens the list as a drawer using the freed left side, rather than floating
it awkwardly next to where the rail used to be. It closes on selecting a body from it or on
toggling `B` again -- borrowed space, not a permanent left column.

## Selection tether

A thin leader line plus an accent ring around the selected body, drawn in the body's own
colour (`Material.colour`, or the black-hole palette entry), connecting it visually to the
context column's identity header. This recovers the "this on screen is that panel" link that
opposite corners lost, without the occlusion problems a floating panel would reintroduce (see
"Why this shape").

## Visual language

- **Elevation.** Formalize the existing canvas -> panel -> button progression
  (`palette.SPACE` (6,8,16) < `PANEL` (14,18,30) < `BUTTON` (26,33,52)) into four explicit
  tiers by inserting a row-group tier between panel and button, plus a 1px lighter top edge
  on raised elements (drawn as a single `pygame.draw.line` in the lightest tier's colour).
  New colours live in `palette.py` beside the existing ladder; no shadows, no textures, no
  alpha-blended blur.
- **Spacing.** The font-derived `Metrics` scale (`pad`, `gap`, `line`) is unchanged; a
  deliberate double-gap is inserted between the stat groups described above so grouping reads
  from whitespace, matching how `_build_tool_rail`'s panel already separates itself from
  neighbours by a gap rather than a rule.
- **Numeric hierarchy.** Value / label / unit get three brightness tiers: value in `TEXT`,
  label in `TEXT_DIM` (both exist today), unit in a new, dimmer `TEXT_FAINT`. Two bounded
  quantities gain a small filled bar instead of a bare number: mass as a fraction of
  `units.STAR_MASS_THRESHOLD` (replacing/augmenting the existing "%.0fx heavier to ignite"
  line), and temperature as a fraction of the body's `Material.vapour_temperature_k`.
- **Micro-feedback.** A 1px press-down offset on `STATE_DOWN` (draw the button content
  shifted one pixel down-right, same fill), and a ~120ms open/close height interpolation on
  the stat mini-card's expand and the Bodies drawer. This is transient interaction state on
  `UiLayer` (an animation-progress float per panel, alongside the `_pressed`/`_repeat_at`
  state it already keeps across frames) -- not cached layout geometry, so it does not
  conflict with "layout is rebuilt every frame" (`ui.py` module docstring): the *geometry* is
  still recomputed every frame from current state, only interpolated toward by the animation
  progress, the same way a slider's drawn thumb position is computed fresh from `control.value`
  every frame today.

## Architecture

### `src/app/ui.py`

Replace `_build_tool_rail`, `_build_tool_context`, `_build_inspector_actions` and
`_build_body_list` with:

- `_build_bottom_hotbar(state, m, ctx)` -- tool-picker controls plus the existing transport
  row, in one bottom-anchored layout call. Absorbs `TOOL_BUTTONS` handling from the old
  `_build_tool_rail`.
- `_build_context_column(state, m)` -- identity, actions, stat mini-card (with its expand
  control) and the live tool's `controls()` rows, in that order, as one panel. Absorbs
  `_build_tool_context`'s row-layout logic and `_build_inspector_actions`'s action-row
  layout.
- `_build_body_list(state, m)` -- kept as its own method (the drawer), re-anchored to the
  left edge instead of `rail[0] + max(rail[1], context_w)`.

`UiLayer` gains the small animation-progress state described above (e.g. `self._anim`, a
dict of panel-id -> progress-float, advanced in `update()` the same way held-stepper repeats
already are).

### `src/render/hud.py`

`_rows` splits into a headline-stat function (per body kind, one or two lines) and the
existing full-row builder, now grouped with blank-row separators at group boundaries instead
of ungrouped. `inspector_rect` and `visible_rows` account for the collapsed/expanded height
instead of always the full row count.

### `src/render/widgets.py`

New: a horizontal filled-bar primitive (`draw_bar(surface, rect, fraction, colour)`,
clamped 0..1), a chevron/expand icon case in `draw_icon` (reuses the existing `_chevron`
helper with `"up"`/`"down"`), the 1px press-offset in `_draw_button_content`, and the new
elevation-tier fill lookup used by `draw_panel`.

### `src/render/palette.py`

New tier colours: one row-group fill between `PANEL` and `BUTTON`, `TEXT_FAINT` (dimmer than
`TEXT_DIM`), and a raised-edge highlight tone.

### Unchanged

`src/app/tools.py` -- no changes to `Tool`, `SelectTool`, `LaunchTool`, `EditTool`, or the
`controls()`/`perform()` contract; the redesign only changes where `ui.py` draws what tools
already describe. No changes to `src/edits.py`, `src/body.py`, or any physics/simulation
module. Every existing hotkey stays bound to the same action; only the button's on-screen
position changes, so `tests/test_ui.py`'s `KEY_HELP` parity check needs no table changes.

## Testing

- `tests/test_layout.py` -- rewritten for the new regions: the bottom hotbar (both the
  single-row and, if needed, two-row arrangement) and the context column must both fit at
  `config.min_internal(3)`, the same floor the current test enforces.
- `tests/test_ui.py` -- hotkey/button parity table is unchanged; add cases confirming the
  tool-picker controls now build inside the bottom-hotbar method and the inspector actions
  build inside the context-column method, so a future refactor cannot silently drop one.
- New assertions for the stat mini-card: collapsed by default on a fresh selection, expands
  on clicking the chevron, collapses again on deselect-then-reselect.
- `tests/test_smoke.py` -- already drives `App.step_once()` headless; extend to click through
  a tool selection and a body selection to confirm the consolidated column builds without
  error at both the default and floor window sizes.

## Non-goals

- No user-customizable or draggable panel docking.
- No touch/gesture-specific rework beyond whatever the new hit targets give for free.
- No per-body persistent expand/collapse memory.
- No change to the physics, edit, or collision/destruction invariants in `CLAUDE.md` --
  this spec touches `src/app/` and `src/render/` only.

## Risks

- **Bottom-hotbar width at the floor.** Merging the tool picker into the transport row adds
  four compact buttons to an already-measured row; if it doesn't fit in one row at 480
  internal px, the fallback is a second bottom-anchored row, not a return to a side rail.
- **Context-column height at the floor.** Stacking identity, actions, stat mini-card and tool
  controls in one panel is taller than today's three separate panels combined were, before
  collapse is accounted for; the collapsed-by-default stat card is what is expected to make
  this fit, but it needs verifying against `tests/test_layout.py`'s floor case once built.
  If it does not fit, the tool-controls section should scroll or the stat groups should
  compress further -- not spill outside `content_bottom`.
- **Animation state surviving a resize or grid switch.** The buffer can change shape at any
  frame boundary (`config` module docstring); an in-flight expand/collapse animation must not
  read a stale panel height from before the resize. Simplest guard: clamp the animation's
  target height to the freshly-built panel's actual height every frame, the way the slider
  clamps its own fraction today.
