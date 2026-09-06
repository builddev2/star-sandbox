"""Everything the application knows that is not the world itself.

Selection and camera-follow store body **ids**, never indices, because
World.remove swap-removes: an index held from one frame to the next can quietly
start referring to a different planet.  Every read goes through index_of.
"""

import math

import pygame

from .. import explain, units
from ..body import KIND_CRAFT
from ..render import scale
from ..render.effects import EffectField

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


class ExplainRecorder:
    """One report per burst of edits, not one per scrub notch.

    Mass scrubs by ratio at 1.06 a notch, three notches a press, repeating
    while held: doubling the Sun is a dozen-plus separate ``set_mass`` calls.
    A report per call would be a strobe, and the baseline would be the previous
    notch rather than the state before he started.

    So the baseline is taken once, on the first edit of a burst, and the report
    is computed once, when the burst has been quiet for ``SETTLE_MS``.
    """

    def __init__(self):
        #: The finished report on show, or None.
        self.report = None
        #: Off means off: ``note`` returns before taking a snapshot, so the
        #: feature costs nothing at all rather than recording invisibly.
        self.enabled = True
        self._baseline = None
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

    def settle(self, world, now_ms):
        """Call once a frame.  Closes a burst that has gone quiet."""
        if self._baseline is None:
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
        self._body_id = self._field = None

    def _close(self, world):
        found = explain.diff(self._baseline, explain.snapshot(world),
                             self._body_id, self._field)
        if found is not None:
            self.report = found
        self._baseline = None
        self._body_id = self._field = None


class AppState:
    def __init__(self, sim, camera, view):
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
