"""Everything the application knows that is not the world itself.

Selection and camera-follow store body **ids**, never indices, because
World.remove swap-removes: an index held from one frame to the next can quietly
start referring to a different planet.  Every read goes through index_of.
"""

import math
from datetime import datetime

import pygame

from .. import explain, save, units
from ..body import KIND_CRAFT
from ..render import scale
from ..render.camera import MAX_ZOOM, MIN_ZOOM
from ..render.effects import EffectField
from .saves import SLOTS, SlotInfo, UNREADABLE_NAME

#: Days of simulation per second of real time, coarsest last.
TIME_SCALES = (
    1.0 / 24.0, 0.25, 1.0, 3.0, 7.0, 30.0, 90.0,
    units.DAYS_PER_YEAR, 3.0 * units.DAYS_PER_YEAR, 10.0 * units.DAYS_PER_YEAR,
)
TIME_LABELS = (
    "1 hr/s", "6 hr/s", "1 day/s", "3 day/s", "1 week/s", "1 month/s",
    "3 month/s", "1 yr/s", "3 yr/s", "10 yr/s",
)

DEFAULT_TIME_INDEX = 5          # 1 month per second: fast enough to see motion

#: Starting arrangements the Restart button can load, and what to call them.
#: Short labels: the transport row is already tight at the minimum window
#: size, and "Solar system" / "Black hole" pushed it off screen.
SCENARIOS = ("solar_system", "black_hole")
SCENARIO_LABELS = {"solar_system": "Solar", "black_hole": "Hole"}

#: Click tolerance in pixels, so a 3px Mercury is still catchable.
PICK_MARGIN = 8

DEFAULT_HINT = "Click a body to inspect it  -  wheel zooms  -  ? for controls"

#: Milliseconds of quiet that end a burst of edits.  Wall clock, never
#: simulated time -- this is a property of how fast a child's hand moves, and
#: it must behave identically at one hour a second and at ten years a second,
#: the same argument render/effects.py makes about particle velocities.
SETTLE_MS = 350

#: Which modal is up.  ``"start"`` is the opening New / Load / Quit menu,
#: ``"load"`` the slot list reached from it (load only, Back), ``"saves"``
#: the in-game list with Load, Save and Rename per slot.
DIALOG_START = "start"
DIALOG_LOAD = "load"
DIALOG_SAVES = "saves"

#: Longest name a slot can have.  Wide enough for "Jupiter eats everything",
#: narrow enough that the name row still fits beside three buttons at the
#: smallest window (tests/test_layout.py).
NAME_MAX = 24


class DialogState:
    """What the open dialog knows: which one it is, the listed slots, and the
    two things a slot can be in the middle of -- an armed overwrite, or a
    name being typed.  Only one slot can be in either state at once, and
    starting one ends the other."""

    def __init__(self, kind, slots):
        self.kind = kind
        self.slots = slots
        #: Slot whose Save button reads "Overwrite?" and is waiting for the
        #: second press, or None.
        self.armed = None
        #: Slot whose name is a text field right now, or None.
        self.renaming = None
        self.text = ""
        #: Whether the sim was paused before the in-game dialog opened, so
        #: closing it puts time back how it was.
        self.resume_paused = False


class ExplainRecorder:
    """One report per burst of edits, not one per scrub notch.

    Mass scrubs by ratio at 1.06 a notch, three notches a press, repeating
    while held: doubling the Sun is a dozen-plus separate ``set_mass`` calls.
    A report per call would be a strobe, and the baseline would be the previous
    notch rather than the state before he started.

    So the baseline is taken once, on the first edit of a burst, and the report
    is computed once, when the burst has been quiet for ``SETTLE_MS``.

    The "after" side is *not* a snapshot taken at settle time, and not merely
    "on the next frame" either.  ``App.step_once`` runs ``sim.advance`` --
    which calls ``thermal.advance`` once, with the *whole* frame's worth of
    simulated time -- **before** it calls ``settle``.  So a snapshot deferred
    to the next ``settle`` call already has one frame of physics baked into
    it, and one frame is not small in simulated time: at ten years a second a
    60 fps frame is over 60 simulated days.  Temperature's relaxation time
    constant is under a thousandth of a day for anything glowing (see
    ``units.THERMAL_SKIN_DEPTH_M``'s use in ``thermal.py``), which is far
    shorter than that -- confirmed by direct measurement, not assumed: heating
    Earth to 3115 K and letting exactly one frame run at ten years a second
    already relaxes it back to 268 K, indistinguishable from having never
    written it.  A snapshot taken so much as one frame late would still blame
    the sim's own progress on the edit, just over a shorter window than the
    original bug.  Orbital elements do not have this problem -- they are
    invariant along an orbit -- which is why this class exists to treat
    temperature specially rather than to delay every snapshot.

    So the "after" snapshot is taken synchronously, in the same call stack as
    the write itself, before ``step_once`` ever reaches that frame's own
    ``sim.advance``: every write path calls ``note`` (opens or extends the
    burst) and then, once the edit is actually made, ``record`` (freezes
    ``_after``).  ``settle`` -- called once a frame regardless of whether
    anything was written -- only ever checks the quiet timer and, once it has
    elapsed, diffs the baseline against that already-frozen ``_after``; it
    never takes a snapshot of its own.
    """

    def __init__(self):
        #: The finished report on show, or None.
        self.report = None
        #: Off means off: ``note`` returns before taking a snapshot, so the
        #: feature costs nothing at all rather than recording invisibly.
        self.enabled = True
        self._baseline = None
        #: The world exactly as the last write left it -- see the class
        #: docstring.  None until the first ``record`` of a burst.
        self._after = None
        self._body_id = None
        self._field = None
        self._last_ms = 0

    def note(self, world, body_id, field, now_ms):
        """Call *before* each write.  Opens or extends a burst."""
        if not self.enabled or body_id is None:
            return
        if self._baseline is not None and (self._body_id != body_id
                                           or self._field != field):
            # A different body or a different property is a different lesson.
            self._close(world)
        if self._baseline is None:
            self._baseline = explain.snapshot(world)
            self._body_id, self._field = body_id, field
        self._last_ms = now_ms

    def record(self, world):
        """Call *after* each write, before anything else can move the world.

        This is what makes ``_after`` "the world at the write" rather than
        "the world one frame later": nothing runs between the edit and this
        call, in the same tool method, so no physics step -- and no thermal
        relaxation, whose time constant can be shorter than a single frame at
        real time-scales -- has a chance to happen first.
        """
        if not self.enabled or self._baseline is None:
            return
        self._after = explain.snapshot(world)

    def settle(self, world, now_ms):
        """Call once a frame.  Closes a burst that has gone quiet.

        Takes no snapshot itself -- ``record`` already froze ``_after`` at
        the write -- so this is one float comparison when nothing is pending.

        Off means off (``enabled``), but not by holding the report back: a
        burst still open when the toggle flips freezes right there, because
        ``note`` refuses to open or extend one while disabled.  The next call
        to this method after re-enabling closes that frozen burst and installs
        its report same as any other -- what disabling withholds is anything
        from the disabled interval showing up in it, not the report itself.
        """
        if not self.enabled or self._baseline is None:
            return
        if now_ms - self._last_ms >= SETTLE_MS:
            self._close(world)

    def dismiss(self):
        self.report = None

    def clear(self):
        """Drop everything.  A restart rebuilds every Body, so a standing
        report and an open burst both name bodies that no longer exist."""
        self.report = None
        self._baseline = None
        self._after = None
        self._body_id = self._field = None

    def _close(self, world):
        """Diff the baseline against ``_after`` -- the world exactly as the
        last write left it, never a fresh snapshot taken here.

        ``_after`` can still be ``None`` in principle -- a write path that
        calls ``note`` without ever calling ``record`` -- so this falls back
        to a fresh snapshot rather than diffing against nothing.  Every real
        write path (``EditTool._scrub``, ``EditTool._collapse``) calls both.
        """
        after = self._after if self._after is not None else explain.snapshot(world)
        found = explain.diff(self._baseline, after, self._body_id, self._field)
        if found is not None:
            self.report = found
        self._baseline = None
        self._after = None
        self._body_id = self._field = None


class AppState:
    def __init__(self, sim, camera, view, store=None):
        self.sim = sim
        self.camera = camera
        self.view = view
        self.effects = EffectField()
        #: The "what changed" report, and the burst that produces it.
        self.explain = ExplainRecorder()
        #: Where the explain panel was last drawn, so the Fly readout can start
        #: below it instead of underneath it.  None when no panel is up.
        self.explain_rect = None
        #: Whether the explain panel is showing its formulas.  Session-wide
        #: rather than per-report: someone who wants the maths wants it every
        #: time, and someone who does not never opens it.
        self.show_maths = False
        self.selected_id = None
        #: Collapsed by default: a headline number is what a fresh selection
        #: earns, not a page of stats.  Not per-body -- nothing asked for that,
        #: and it would need a place to live on Body for a feature nobody
        #: requested.
        self.inspector_expanded = False
        #: When the current selection was made, in ``pygame.time.get_ticks()``
        #: milliseconds.  The selection tether (``ui.py``) reads this to fade
        #: itself out a few seconds after a selection, and to restart that
        #: countdown when a selection happens again -- including re-picking
        #: the body that was already selected, which changes nothing else
        #: this state tracks.
        self.selected_at = 0
        self.time_index = DEFAULT_TIME_INDEX
        self.running = True
        self.help_visible = False
        self.tool = None
        self.tool_name = "Look"
        #: The list of bodies starts open.  It is how someone who has never
        #: pressed Tab discovers that the planets have names and are clickable.
        self.show_body_list = True
        self.body_list_offset = 0
        #: The quit button has been pressed once and is waiting to be confirmed.
        self.quit_armed = False
        #: Which arrangement Restart loads.  sim.reset() used to hard-code the
        #: solar system, so there was no way to reach any other.
        self.scenario = SCENARIOS[0]
        #: Where saves go, or None when there is nowhere (tests).  With no
        #: store every Save button is disabled and every slot lists empty.
        self.store = store
        #: The open modal, or None.  While one is up, ``ui.build`` lays out
        #: only it and ``input.handle_event`` routes everything to it.
        self.dialog = None

    # --- time -------------------------------------------------------------------

    @property
    def days_per_second(self):
        return TIME_SCALES[self.time_index]

    @property
    def time_label(self):
        return TIME_LABELS[self.time_index]

    @property
    def time_labels(self):
        """Every speed label, so the interface can size its readout once."""
        return TIME_LABELS

    @property
    def time_index_max(self):
        return len(TIME_SCALES) - 1

    def nudge_time(self, delta):
        self.time_index = max(0, min(len(TIME_SCALES) - 1, self.time_index + delta))

    def toggle_pause(self):
        self.sim.paused = not self.sim.paused

    @property
    def elapsed_label(self):
        days = self.sim.world.time
        years = days / units.DAYS_PER_YEAR
        return "Year %.1f  Day %d" % (years, int(days))

    # --- selection --------------------------------------------------------------

    @property
    def selected_index(self):
        if self.selected_id is None:
            return None
        return self.sim.world.index_of(self.selected_id)

    @property
    def selected_body(self):
        index = self.selected_index
        return None if index is None else self.sim.world.bodies[index]

    def select(self, body_id, now=None):
        self.selected_id = body_id
        self.inspector_expanded = False
        self.selected_at = pygame.time.get_ticks() if now is None else now

    def toggle_inspector_expanded(self):
        self.inspector_expanded = not self.inspector_expanded

    def cycle_selection(self, step=1):
        world = self.sim.world
        if world.n == 0:
            self.selected_id = None
            return
        current = self.selected_index
        nxt = 0 if current is None else (current + step) % world.n
        self.select(world.bodies[nxt].id)

    def pick_body_at(self, sx, sy, internal_height=None):
        """Index of the body under a screen point, or None.

        Among everything the click touches, the winner is whichever it landed
        *deepest inside*, measured as distance over that body's own drawn
        radius.  Neither simpler rule works: picking the nearest centre is a
        coin toss between Jupiter and Io at system zoom, where they are a third
        of a pixel apart, and picking the smallest match means clicking Jupiter
        always selects Io instead.
        """
        world = self.sim.world
        if world.n == 0:
            return None
        height = internal_height or self.camera.height
        radii = scale.display_radii_px(world.radii, self.camera.zoom, height,
                                       real_size=self.view.real_size,
                                       floor_radii=scale.visual_radii(world))
        screen = self.camera.world_to_screen_array(world.positions)
        craft_floor = scale.craft_floor_px(height)
        best = None
        best_score = float("inf")
        for i in range(world.n):
            radius = max(float(radii[i]), 2.0)
            if world.bodies[i].kind == KIND_CRAFT and not self.view.real_size:
                # Picking must agree with drawing (render/draw.py), or the
                # click target and the visible shuttle disagree, and the gap
                # widens with the window -- see scale.craft_floor_px.
                radius = max(radius, craft_floor)
            distance = math.hypot(screen[i][0] - sx, screen[i][1] - sy)
            if distance > radius + PICK_MARGIN:
                continue
            score = distance / radius
            if score < best_score:
                best_score, best = score, i
        return best

    def follow_selected(self):
        if self.selected_id is not None:
            self.camera.follow_id = self.selected_id

    def toggle_follow(self):
        """Follow the selection, or stop following if it is already followed.

        The button and the F key share this: a control that only ever turns
        something on gives no way back out except panning, which is not
        discoverable.
        """
        if self.selected_id is None:
            return
        if self.camera.follow_id == self.selected_id:
            self.camera.follow_id = None
        else:
            self.camera.follow_id = self.selected_id

    def fly_to_selected(self):
        """Zoom in on the selection and stay with it -- the double-click, as a
        button, for anyone who has not discovered that double-click works."""
        index = self.selected_index
        if index is None:
            return
        self.camera.frame_body(self.sim.world, index)
        self.camera.follow_id = self.selected_id

    # --- the body list ----------------------------------------------------------

    def scroll_body_list(self, delta):
        self.body_list_offset = max(0, self.body_list_offset + int(delta))

    def clamp_body_list_offset(self, visible):
        """Keep the scroll offset inside a list of ``visible`` rows.

        Bodies are destroyed while the list is open -- a merge shortens it under
        the cursor -- so the offset is clamped where it is read rather than
        where it is set.
        """
        highest = max(0, self.sim.world.n - int(visible))
        self.body_list_offset = max(0, min(self.body_list_offset, highest))
        return self.body_list_offset

    def delete_selected(self):
        index = self.selected_index
        if index is None:
            return
        self.sim.world.remove(index)
        self.selected_id = None
        self.sim.refresh()

    # --- tools ------------------------------------------------------------------

    def set_tool(self, tool_cls):
        if self.tool is not None:
            self.tool.on_exit(self)
        self.tool = tool_cls()
        self.tool_name = self.tool.name

    @property
    def hint(self):
        return self.tool.hint if self.tool is not None else DEFAULT_HINT

    # --- global commands --------------------------------------------------------

    def stop_selected(self):
        """Zero the selected body's velocity.  Stop the Earth; watch it fall."""
        index = self.selected_index
        if index is None:
            return
        self.sim.world.vel[index] = 0.0
        self.sim.refresh()

    # --- leaving ----------------------------------------------------------------

    def request_quit(self):
        """Press once to ask, twice to go.

        Nothing is saved, so a stray click on a quit button costs the child the
        whole solar system they have spent an afternoon wrecking.  The first
        press only arms it; the button then says so in as many words.
        """
        if self.quit_armed:
            self.running = False
        else:
            self.quit_armed = True

    def cancel_quit(self):
        self.quit_armed = False

    @property
    def scenario_label(self):
        return SCENARIO_LABELS[self.scenario]

    def cycle_scenario(self):
        """Step to the next arrangement.  Does not restart on its own -- the
        child picks, then presses Restart, so a stray click costs nothing."""
        here = SCENARIOS.index(self.scenario)
        self.scenario = SCENARIOS[(here + 1) % len(SCENARIOS)]

    def reset(self):
        self.sim.reset(self.scenario)
        self.explain.clear()
        self.explain_rect = None
        self.selected_id = None
        self.camera.follow_id = None
        self.body_list_offset = 0

    # --- the start menu and the saves dialog ------------------------------------

    def _list_slots(self):
        return [None] * SLOTS if self.store is None else self.store.list()

    def open_start(self):
        """The opening menu.  Time is frozen behind it until New or Load."""
        self.dialog = DialogState(DIALOG_START, [])
        self.sim.paused = True

    def start_new(self):
        self.dialog = None
        self.sim.paused = False

    def quit_now(self):
        """Quit from the start menu: nothing has happened yet, so no arming."""
        self.running = False

    def open_load(self):
        self.dialog = DialogState(DIALOG_LOAD, self._list_slots())

    def open_saves(self):
        """The in-game list.  Pauses, remembering whether it was, so what he
        saves is exactly what is on screen."""
        if self.dialog is not None:
            return
        d = DialogState(DIALOG_SAVES, self._list_slots())
        d.resume_paused = self.sim.paused
        self.sim.paused = True
        self.dialog = d
        self.cancel_quit()

    def toggle_saves(self):
        """The S key and the disk button: open the list, or put it away."""
        if self.dialog is None:
            self.open_saves()
        elif self.dialog.kind == DIALOG_SAVES:
            self.close_dialog()

    def close_dialog(self):
        """Esc and the close button.  The in-game list restores the pause
        state it found; the load list goes back to the start menu; the start
        menu has no way out but its own three buttons."""
        d = self.dialog
        if d is None:
            return
        self.rename_cancel()
        if d.kind == DIALOG_SAVES:
            self.sim.paused = d.resume_paused
            self.dialog = None
        elif d.kind == DIALOG_LOAD:
            self.open_start()

    def disarm(self):
        """Any press that is not the second press is an answer of "no"."""
        if self.dialog is not None:
            self.dialog.armed = None

    def default_save_name(self):
        years = self.sim.world.time / units.DAYS_PER_YEAR
        return "%s Year %.1f" % (self.scenario_label, years)

    def _session(self):
        camera = self.camera
        # The dialog pauses the sim mechanically the moment it opens
        # (open_saves), so sim.paused while it is up is not what he had
        # running -- it is always True.  What belongs in the save is the
        # pause state the dialog *found*, so a save made from a running
        # world loads running.
        paused = (self.dialog.resume_paused
                  if self.dialog is not None and self.dialog.kind == DIALOG_SAVES
                  else self.sim.paused)
        return save.Session(
            scenario=self.scenario, time_index=self.time_index,
            paused=paused, show_trails=self.view.show_trails,
            real_size=self.view.real_size, show_labels=self.view.show_labels,
            explain_enabled=self.explain.enabled,
            show_body_list=self.show_body_list, selected_id=self.selected_id,
            centre=(float(camera.centre[0]), float(camera.centre[1])),
            follow_id=camera.follow_id,
            au_across=camera.width / camera.zoom)

    def _apply_session(self, session):
        camera = self.camera
        if session.scenario in SCENARIOS:
            self.scenario = session.scenario
        self.time_index = max(0, min(len(TIME_SCALES) - 1, session.time_index))
        self.view.show_trails = session.show_trails
        self.view.real_size = session.real_size
        self.view.show_labels = session.show_labels
        self.explain.enabled = session.explain_enabled
        self.show_body_list = session.show_body_list
        self.selected_id = session.selected_id
        self.inspector_expanded = False
        camera.centre[:] = session.centre
        camera.follow_id = session.follow_id
        au_across = max(session.au_across, 1e-12)
        camera.log_zoom = min(max(math.log(camera.width / au_across),
                                  math.log(MIN_ZOOM)), math.log(MAX_ZOOM))

    def save_slot(self, slot):
        """Save into ``slot``.  Returns True only when a file was written.

        An empty slot takes one press.  A full one -- readable or not --
        needs a second press on the same slot while it is armed, and the
        name it had is kept: the name is his label for that game.
        """
        d = self.dialog
        if d is None or d.kind != DIALOG_SAVES or self.store is None:
            return False
        self.rename_cancel()
        info = d.slots[slot - 1]
        if info is not None and d.armed != slot:
            d.armed = slot
            return False
        d.armed = None
        name = info.name if (info is not None and info.readable) \
            else self.default_save_name()
        data = save.to_data(self.sim.world, self._session(), name, self.scenario_label)
        try:
            self.store.write(slot, data)
        except OSError:
            return False
        d.slots = self.store.list()
        return True

    def load_slot(self, slot):
        """Replace the running world with ``slot``.  Returns True on success.

        A file the loader refuses is marked unreadable in the list and the
        running world is left exactly as it was -- ``save.from_data``
        validates before it touches anything.
        """
        d = self.dialog
        if d is None or self.store is None:
            return False
        info = d.slots[slot - 1]
        if info is None or not info.readable:
            return False
        self.rename_cancel()
        d.armed = None
        try:
            data = self.store.read(slot)
            session = save.from_data(data, self.sim.world)
        except (OSError, ValueError):
            d.slots[slot - 1] = SlotInfo(slot, UNREADABLE_NAME, "", {}, False)
            return False
        self.sim.refresh()
        self._apply_session(session)
        # Loading is a restart as far as everything that remembers a body
        # goes: the report names bodies that no longer exist, the effects are
        # someone else's impacts, and a tool's target id could collide with
        # a loaded id by accident.
        self.explain.clear()
        self.explain_rect = None
        self.effects.clear()
        self.body_list_offset = 0
        self.quit_armed = False
        if self.tool is not None:
            self.set_tool(type(self.tool))
        self.dialog = None
        self.sim.paused = session.paused
        return True

    def begin_rename(self, slot):
        d = self.dialog
        if d is None or self.store is None:
            return
        info = d.slots[slot - 1]
        if info is None or not info.readable:
            return
        d.armed = None
        d.renaming = slot
        d.text = info.name

    def rename_type(self, text):
        """Append typed characters: printable ASCII, up to NAME_MAX."""
        d = self.dialog
        if d is None or d.renaming is None:
            return
        for ch in text:
            if 32 <= ord(ch) <= 126 and len(d.text) < NAME_MAX:
                d.text += ch

    def rename_backspace(self):
        d = self.dialog
        if d is not None and d.renaming is not None:
            d.text = d.text[:-1]

    def rename_commit(self):
        """Enter.  Writes the name; an empty one keeps the old name."""
        d = self.dialog
        if d is None or d.renaming is None:
            return
        slot, d.renaming = d.renaming, None
        info = d.slots[slot - 1]
        name = d.text.strip() or (info.name if info is not None else self.default_save_name())
        d.text = ""
        try:
            self.store.rename(slot, name[:NAME_MAX])
        except (OSError, ValueError):
            return
        d.slots = self.store.list()

    def rename_cancel(self):
        d = self.dialog
        if d is not None:
            d.renaming = None
            d.text = ""

    def describe_slot(self, info):
        """The dim second line under a slot's name."""
        if info is None:
            return "Nothing saved here yet"
        if not info.readable:
            return "This file can't be read -- saving over it will fix it"
        summary = info.summary
        try:
            when = datetime.fromisoformat(info.saved_at).strftime("%a %d %b %H:%M")
        except (TypeError, ValueError):
            when = info.saved_at
        years = float(summary.get("days", 0.0)) / units.DAYS_PER_YEAR
        return "%s  Year %.1f  %d bodies  %s" % (
            summary.get("scenario_label", "?"), years, int(summary.get("bodies", 0)), when)

    def cycle_grid(self, ctx):
        """Step the pixel grid.  Sprites are sized in pixels, so the cache dies.

        Cycles the grids this window is actually big enough for, which in a
        third-of-the-screen window is fewer than in a fullscreen one.
        """
        if ctx is None:
            return
        choices = ctx.usable_grids
        here = choices.index(ctx.s) if ctx.s in choices else -1
        ctx.set_grid(choices[(here + 1) % len(choices)])
        self.on_resize(ctx)

    def cycle_window_size(self, ctx):
        """Step the window through the size presets.  Leaves fullscreen behind.

        A cycle rather than four buttons because the toolbar is already eight
        icons wide, and three presses is a cheaper way through four choices
        than three more things to read.
        """
        if ctx is None:
            return
        ctx.cycle_window_size()
        self.on_resize(ctx)

    def on_resize(self, ctx):
        self.camera.width = ctx.width
        self.camera.height = ctx.height
