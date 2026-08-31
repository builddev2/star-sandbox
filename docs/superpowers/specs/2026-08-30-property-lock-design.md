# Property lock

Any editable field on a body -- mass, density, temperature, material, spin, disk -- can
be locked. A locked field ignores every attempt to change it, direct or accidental, until
it is explicitly unlocked, so a child can pin the Sun's mass while they scrub its density
without a stray Left/Right nudging it back off target.

## Why this shape

The Edit tool already lets you select a field and change it (`src/app/tools.py:475`); a
lock is a per-field, per-body switch on top of that, not a new editing mode. Two things
follow:

1. **The lock has to live on the body.** It is a standing preference for *this* Jupiter,
   not a session flag -- select away and back, or leave the Edit tool and return, and the
   lock is still there. `Body.spin` and `Body.disk` already establish the pattern: static,
   per-body state the physics kernel never reads.
2. **The lock has to be enforced where the value is set, not where the button is drawn.**
   `edits.set_temperature` already refuses to touch a star or a black hole's temperature
   "even though the panel already omits the row... the panel is not the only caller." A
   lock gets the same treatment: each setter in `edits.py` refuses its own field when that
   field is locked. The panel disabling its buttons is a second, cosmetic line of defence,
   not the enforcement.

**Scope, decided during design:** a lock only blocks *direct* edits to its own field. It
does not chase that value through other fields -- locking density does not stop a material
swap from moving density, because `set_material` is editing material, not density. This
matches the one worked example (lock mass, freely change density/material/temperature) and
avoids inventing cross-field bookkeeping nothing asked for.

## Architecture

### Data model

`src/body.py` -- add `"locked_fields"` to `Body.__slots__`, defaulting to an empty
`frozenset` in `__init__`. Values are the same strings `edits.edit_fields()` already
returns: `"mass"`, `"density"`, `"temp"`, `"material"`, `"spin"`, `"disk"`.

A fresh body, a collision fragment, and a tidal fragment all start with no locks -- nothing
carries a parent's locks onto its debris, the same as `spin`/`disk` defaults today. A kind
change (ignite, extinguish, collapse) does not touch `locked_fields`; a lock on a field the
current kind doesn't offer is simply inert until the kind changes back (a star that cools
back into a planet, for instance, finds its old temperature lock still in place).

### `src/edits.py`

New:

- `is_locked(body, field) -> bool`
- `toggle_lock(body, field)` -- flips membership in `locked_fields`
- `unlock_all(body)` -- empties it

Changed -- one guard clause added at the top of each setter, keyed to its own field:

| setter | guarded field |
|---|---|
| `set_mass` | `"mass"` |
| `set_density` | `"density"` |
| `set_temperature` | `"temp"` |
| `set_spin` | `"spin"` |
| `set_disk` | `"disk"` |
| `set_material` / `cycle_material` | `"material"` |

`set_temperature` already has an early-return shape for stars/black holes; the lock check
joins it. Nothing else in `edits.py` changes -- `_refresh_kind`, `collapse`, and the
density-follows-mass paths inside `set_mass`/`set_material` are unaffected, since they are
edits to a *different* field than the one a lock on "mass" or "material" guards.

### `src/app/tools.py` (`EditTool`)

- `controls()`: each field row's two step buttons (`edit.step`) get `enabled=not
  edits.is_locked(body, field)`. A new third control per row, `edit.lock` (payload=field),
  is always enabled and shows icon `"lock"` when locked, `"unlock"` when not -- the same
  icon-swap-on-state already used for play/pause.
- A new panel-level control, `edit.unlock_all`, placed next to `Collapse`, `enabled=bool(
  body.locked_fields)`.
- `perform()`: `edit.lock` sets `self.field_index` to that row (matching how `edit.step`
  already does) and calls `edits.toggle_lock`. `edit.unlock_all` calls `edits.unlock_all`.
- `on_event()`: `pygame.K_k` toggles the lock on the currently selected field (mirrors
  Left/Right acting on `self.field_index`); `pygame.K_u` calls unlock-all. Both guarded the
  same way `K_c` (Collapse) already is -- guard on `state.selected_index is not None`, no
  `enabled` gate to duplicate since these two are never disabled themselves (`edit.lock` is
  always enabled; unlock-all's key path re-checks `body.locked_fields` before acting so it
  is a no-op with nothing to undo, matching the disabled button's behaviour).
- Tool hint text (`EditTool.hint`) gains a short clause for K.

### `src/render/hud.py`

Two rows added to `KEY_HELP`:

```
("K", "lock/unlock the selected field (Edit)")
("U", "unlock every field on the selected body (Edit)")
```

`tests/test_ui.py` already fails if a documented key has no button behind it, so these two
rows are what makes the parity test aware of the new controls.

### `src/render/widgets.py`

Two new cases in `draw_icon`, drawn from lines like every other icon: `"lock"` (a small
rect body under a closed shackle arc) and `"unlock"` (the same body under a shackle drawn
open/offset, so the two read as a pair rather than unrelated glyphs).

## Interface

Each row in the Edit panel goes from three controls to four:

```
[ - ]  Mass   1.00 Suns  [ + ]  [lock]
```

Clicking the lock icon, or pressing K while that row is selected, disables the row's `-`/`+`
(they draw greyed, matching every other `enabled=False` control) and swaps the icon to the
closed padlock. The row stays selectable and its value still reads out normally -- only the
two step controls and the paired hotkeys/drag are inert.

A new "Unlock all" row appears under Collapse (or in the black hole panel's equivalent
spot), greyed out when nothing on the body is locked.

No change to which fields a body offers -- `edit_fields()` is untouched. A locked field
that a kind change removes from the panel (temperature, on an ignited star) simply isn't
shown; its lock is inert, not lost, exactly as described under Data model.

## Testing

`tests/test_edits.py` -- for each of the six setters: locking the field and calling the
setter leaves the value (and, where relevant, the derived radius) unchanged; unlocking and
calling it again changes the value as it does today. Plus direct tests of `is_locked`,
`toggle_lock`, `unlock_all`.

`tests/test_tools_edit.py` -- `controls()` reports `enabled=False` on both step buttons of
a locked row and `enabled=True` on an unlocked one; clicking `edit.lock` toggles the lock
and updates `field_index`; `edit.unlock_all` is disabled with no locks and clears all of
them when pressed; the `K` and `U` key paths produce the same effects as their buttons,
including on a currently-unselected-but-navigated-to row.

`tests/test_ui.py` -- existing hotkey/button parity test picks up `K` and `U` once they're
in `KEY_HELP`; no new test needed there beyond the table update.

`tests/test_layout.py` -- existing "every panel at the floor" test must still pass with the
fourth control added to each row and the new Unlock-all row; if it doesn't, the fix is
sizing/wrapping in that test's terms, not a change to this feature's scope.

## Risks

- **Row width at the smallest grid.** A fourth control per row is new horizontal or
  vertical pressure on the Edit panel. `tests/test_layout.py` is the existing enforcement;
  if the floor doesn't fit, the lock icon should shrink/compact rather than the feature
  growing a second, narrower panel layout.
- **Icon legibility at small sizes.** Lock/unlock need to read as a pair at compact-control
  size (the same size the existing +/- glyphs already draw at), not just at preview size.
