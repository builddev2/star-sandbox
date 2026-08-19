"""Everything the application knows that is not the world itself.

Selection and camera-follow store body **ids**, never indices, because
World.remove swap-removes: an index held from one frame to the next can quietly
start referring to a different planet.  Every read goes through index_of.
"""

import math

from .. import units
from ..render import scale

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

#: Click tolerance in pixels, so a 3px Mercury is still catchable.
PICK_MARGIN = 8

DEFAULT_HINT = "Click a body to inspect it  -  wheel zooms  -  ? for controls"


class AppState:
    def __init__(self, sim, camera, view):
        self.sim = sim
        self.camera = camera
        self.view = view
        self.selected_id = None
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

    def select(self, body_id):
        self.selected_id = body_id

    def cycle_selection(self, step=1):
        world = self.sim.world
        if world.n == 0:
            self.selected_id = None
            return
        current = self.selected_index
        nxt = 0 if current is None else (current + step) % world.n
        self.selected_id = world.bodies[nxt].id

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
                                       real_size=self.view.real_size)
        screen = self.camera.world_to_screen_array(world.positions)
        best = None
        best_score = float("inf")
        for i in range(world.n):
            radius = max(float(radii[i]), 2.0)
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

    def reset(self):
        self.sim.reset()
        self.selected_id = None
        self.camera.follow_id = None
        self.body_list_offset = 0

    def cycle_grid(self, ctx):
        """Step the pixel grid.  Sprites are sized in pixels, so the cache dies."""
        if ctx is None:
            return
        from .config import GRID_CHOICES
        nxt = GRID_CHOICES[(GRID_CHOICES.index(ctx.s) + 1) % len(GRID_CHOICES)]
        ctx.set_grid(nxt)
        self.on_resize(ctx)

    def on_resize(self, ctx):
        self.camera.width = ctx.width
        self.camera.height = ctx.height
