# Save slots: keeping a wrecked solar system for tomorrow

## The problem

Nothing is saved.  The quit button says so in as many words, and the two-press
confirmation exists only because a stray click costs the child the whole
solar system he has spent an afternoon on.  He wants to keep up to three of
them, come back to them, and call them something.

What he asked for, in his words: three saved sessions; a choice of New, Load or
Quit when the program starts; in the game, save or load; saving into an empty
slot or over a full one; renaming a saved game.

## What this is not

Out of scope: more than three slots, deleting a slot (overwriting is how a
slot is reclaimed), an "unsaved changes" warning on quit (it is a sandbox, not
a document; the two-press quit stays as it is), autosave, saving orbit trails,
saving the window size or pixel grid, cloud anything, and any scenario picker
on the start screen -- New is the solar system, and the in-game Restart
control still switches to the black hole.

## The shape of it

### Starting

The program opens onto the solar system, frozen, with a menu over it:

```
          STAR SANDBOX

          [ New   N ]
          [ Load  L ]
          [ Quit  Q ]
```

While that menu is up nothing else is on screen -- no hotbar, no inspector --
and nothing else answers a key.  **New** closes the menu and time starts.
**Load** replaces the menu with the slot list below, in load-only form with a
Back button.  **Quit** exits at once; there is nothing to lose yet.

### In the game

One compact button, a disk icon, hotkey `S`, in the top-right cluster beside
Quit -- it is an application-level action like Quit, not a view toggle.  It
opens the slot list over a paused world:

```
  SAVED GAMES                                              [x]
  --------------------------------------------------------------
  1  Solar Year 3.2    Sat 12 Sep 14:03    9 bodies  [Load][Save][Rename]
  2  Empty                                            [Save]
  3  Mars trip         Fri 11 Sep 19:40   10 bodies  [Load][Save][Rename]
```

- **Save** into an empty slot happens at once and names the game
  `"<Scenario label> Year <world years, one decimal>"`, e.g. `Solar Year 3.2`.
  Save over a full slot is two-press: the button turns into **Overwrite?**
  until it is pressed again; Esc or a press anywhere else disarms it.
  Overwriting keeps the slot's name -- the name is his label for that game,
  and the game is what he is updating.
- **Load** replaces the world, closes the list, and time resumes in whatever
  pause state was saved.
- **Rename** turns the name into a text field with a cursor.  Printable ASCII
  edits it, up to 24 characters; Backspace deletes; Enter commits and writes
  the file; Esc cancels.  Committing an empty name keeps the old one.  While
  a field has focus every key goes to it -- `S` in the middle of a word must
  not close the dialog.
- **Close** or Esc puts the list away.  The world resumes in the pause state
  it was in before the list opened.

The sim is paused the whole time the list is open, so what is saved is what
is on screen.

A slot whose file cannot be read -- corrupt JSON, a version this build does
not know -- shows **Can't read** in place of its details, with Load and Rename
disabled.  Save stays enabled, because overwriting is the way out.

## What a save holds

Everything that makes the world his world, and nothing the machine owns.

**World.**  `time`, the next body id, and one record per body:
`id, name, kind, material` (the material's `key`), `mass, radius, pos, vel,
temperature, born_day, event_id, no_collide_until, seed, rings, spin, disk,
locked_fields` (a sorted list), `fuel_dv`, and `plan` as a list of
`{due_time, prograde, radial}`.  Ids are saved so the session's selection and
camera follow still name the same body after a load, and the id counter is
saved so a body added afterwards cannot collide with one that was.  Trails
are not saved: they refill within one orbit, which is what Restart already
accepts.

**Session.**  `scenario`, `time_index`, `paused`, the three view flags
(`show_trails, real_size, show_labels`), `explain_enabled`, `show_body_list`,
`selected_id`, and the camera as `centre` (two floats), `follow_id` and
**`au_across`** -- the view width in AU, not `log_zoom`.  `log_zoom` is pixels
per AU, so saving it would make the same save show a different stretch of sky
in a different window; `au_across` gives the same view everywhere and is what
`Camera.__init__` already takes.

**Header.**  `version` (an integer, `1`), `name`, `saved_at` (ISO 8601, local
time), and a `summary` of `scenario_label`, `days` and `bodies`, so the slot
list can be drawn from the header alone.

**Not saved.**  Window size, pixel grid, fullscreen, the help overlay, impact
effects, the explain report and any open burst, the inspector's expanded
state, the body-list scroll offset.

### Loading

`world.clear()`, then for each record `world.add(...)` with the saved
`temperature`, followed by setting the fields `add` does not take
(`id`, `born_day`, `event_id`, `no_collide_until`, `rings`, `spin`, `disk`,
`locked_fields`, `plan` as a tuple of `Maneuver`, `fuel_dv`) and re-keying
`_by_id`; then the id counter, then `world.time`.  `contact_radius` is set to
`radius` -- the app layer refreshes it from the camera on the next frame,
which is the same thing every other radius mutation relies on.  Then
`sim.refresh()`.  `thermal.settle` is **not** run: temperature is state and
was saved, and settling would erase a planet he had just heated.  The
session side restores the camera (`centre`, `follow_id`, `log_zoom` from
`au_across` against the *current* width), the view flags, the time index,
the scenario, the selection, and the explain toggle; `explain.clear()`,
`effects.clear()`, `body_list_offset = 0`, `quit_armed = False`.

A body's `kind` is loaded as saved and not re-derived from mass: a black
hole and a craft are one-way kinds that mass alone cannot reproduce.

## Files

`saves/slot1.json`, `saves/slot2.json`, `saves/slot3.json`, in a `saves/`
directory beside `run.command` (the repository root, found relative to the
package the way `run.command` finds it), created on first save and listed in
`.gitignore`.  A write goes to `slotN.json.tmp` and is moved into place with
`os.replace`, so a crash mid-write leaves the old save rather than half of a
new one.  Rename rewrites the whole file with only the header's `name`
changed, through the same path.

Listing reads only the header of each file.  It is done when a dialog opens
and after each save or rename, never every frame.

## Where the code goes

| | |
|---|---|
| `src/save.py` | `VERSION`, `to_data(world, session) -> dict` and `from_data(data, world) -> session`.  Pure: numpy in, plain lists and floats out, no files, no pygame -- `tests/test_layering.py` asserts it.  `session` is a small frozen dataclass, `Session`, defined here, so the physics layer never sees `AppState`.  Raises `SaveFormatError` on a version it does not know or a record it cannot read. |
| `src/app/saves.py` | `SlotStore(directory)`: `SLOTS = 3`, `path(slot)`, `list() -> [SlotInfo or None]`, `read(slot) -> dict`, `write(slot, data)`, `rename(slot, name)`; `SlotInfo(name, saved_at, summary, readable)`.  `default_directory()` resolves `saves/` beside the package. |
| `src/app/state.py` | `dialog` (`None`, `"start"`, `"load"`, `"saves"`), a `DialogState` holding the listed slots, the armed slot, the renaming slot and its text buffer, and the pause state to restore; the actions `open_saves`, `open_load`, `close_dialog`, `start_new`, `save_slot`, `load_slot`, `begin_rename`, `rename_type`, `rename_commit`, `rename_cancel`, `arm_overwrite`, `disarm`.  Every button and every key calls one of these and nothing else. |
| `src/app/ui.py` | `_build_start` and `_build_slots` (the latter for both `"load"` and `"saves"`, differing only in which buttons each row gets); while `state.dialog` is set `build` lays out **only** the dialog.  `perform` branches for the new control ids.  Key routing to the focused text field. |
| `src/app/input.py` | When `state.dialog` is set, every event goes to `ui.on_event` and stops there; `QUIT` and `VIDEORESIZE` are still honoured first. |
| `src/render/widgets.py` | `draw_text_field(surface, rect, text, focused, metrics)`: the text, a cursor when focused, a focus ring.  Sized from `Metrics`, no pixel constants. |
| `src/render/hud.py` | `KEY_HELP` gains `("S", "save or load a game")`; the start menu's `N`, `L`, `Q` and the dialog's Enter/Esc are documented on the buttons, not in the table. |
| `src/app/main.py` | `App.__init__` starts paused with `state.dialog = "start"`; `App` owns one `SlotStore` and hands it to `AppState`. |
| `.gitignore` | `saves/`. |

The dialog panel sizes itself from the font, like the help panel: as wide as
the longest row needs, capped to the window, rows clipped to the panel.  It
must fit at `config.min_internal(3)`, and `tests/test_layout.py` builds both
dialogs at the floor.

## Error handling

- A slot file that fails to parse, or whose `version` is unknown, lists as
  unreadable.  It is never deleted or repaired; a save over it replaces it.
- A material key that no longer exists loads as `materials.ROCK`.  A kind
  string that is unknown raises `SaveFormatError`, which lists the slot as
  unreadable -- the loader validates before touching the world, so a failed
  load leaves the running world untouched.
- More bodies than `World.MAX_BODIES` in a file is a `SaveFormatError`.
- A save directory that cannot be created (permissions) makes every Save
  button disabled with the tooltip saying why; nothing crashes.
- A `selected_id` or `follow_id` naming a body not in the file loads as
  `None`.

## Testing

**`tests/test_save.py`** -- round trip through `to_data` / `from_data` on a
world that exercises everything: the solar system after a merge and a shatter,
a collapsed black hole with spin and disk, a craft with fuel and a two-node
plan, a planet with `mass` and `temp` locked, Saturn's rings.  Every array
column equal to the last bit, every Body field equal, ids and the id counter
preserved, a body added after loading gets an id no saved body had.
`from_data` on a `version` of `99` raises; an unknown kind raises and leaves
the target world untouched; an unknown material falls back to rock; the JSON
produced is `json.dumps`-able with no numpy types in it.

**`tests/test_saves_store.py`** -- writes land in `slotN.json` and no `.tmp`
remains; a corrupt file lists as unreadable and does not raise; `rename`
changes only `name`; `list` on an empty directory is three `None`s; the
directory is created on first write.

**`tests/test_state.py`** additions -- opening the list pauses and closing
restores the prior pause state, both ways; saving into an empty slot needs
one press and into a full one needs two, and a press on anything else
disarms; rename commit writes the file, cancel does not, empty commit keeps
the old name; loading resolves the selection and the camera follow to the
right bodies; loading a save made at 800 px wide into a 400 px window shows
the same `au_across`.

**`tests/test_ui.py`** additions -- the parity test picks up `S` on its own;
with a dialog open no other control is built; with a text field focused the
`S` key edits the field and does not close the dialog; Esc from a field
cancels the rename and leaves the dialog open, a second Esc closes it.

**`tests/test_layout.py`** additions -- the start menu and the slot list fit
inside the window at the floor for every grid.

**`tests/test_smoke.py`** additions -- the real app driven start → New; and
start → Load → slot 1 after a file has been written; a Save, then a Load,
through `step_once`.

**`tests/test_layering.py`** -- unchanged, and still passing: `save.py`
imports numpy and the physics modules only.
