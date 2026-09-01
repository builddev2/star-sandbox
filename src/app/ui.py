"""The clickable interface: layout, hit-testing, and what each control does.

**Nothing here is a second way to do anything.**  Every control calls the same
AppState or tool method the keyboard already calls, so the two input routes
cannot drift apart -- pressing ``T`` and clicking the trails button run the same
line, and anything added to one is automatically in the other.

**Every button carries its own hotkey.**  A child can play the whole sandbox
with the mouse and never read a key name; the little key cap drawn on each
button is what turns that child into someone who uses the keyboard six months
later.  That is the reason the interface is built from labelled buttons rather
than from menus.

**Layout is rebuilt every frame.**  Which controls exist depends on what is
selected, which tool is live and how big the window is, so caching them means
caching a stale answer.  It is a few dozen rectangles and costs nothing next to
the physics.

**The interface sees mouse events before the tools do.**  Otherwise clicking
"Delete" would also launch a comet through the button, because the throw tool
treats every press in the window as an aim.  A gesture that starts on a panel
belongs to that panel until the button comes back up.
"""

import pygame

from ..render import palette
from ..render.hud import PAD as HUD_PAD
from ..render.hud import PANEL_WIDTH_CHARS
from ..render.widgets import (Control, Metrics, draw_button, draw_panel,
                              draw_tooltip, slider_track)
from . import tools as tools_module
from .config import WINDOW_FRACTIONS

#: Milliseconds a stepper is held before it starts repeating, and the gap
#: between repeats after that.  A deliberate single click never repeats, while
#: holding the mass button turns Jupiter into a star in about two seconds --
#: at one notch per click that would be eighty clicks.
REPEAT_DELAY_MS = 320
REPEAT_INTERVAL_MS = 55
#: Never spend a whole frame catching up on repeats after a stall.
MAX_REPEATS_PER_FRAME = 8

#: Width of the body list, in characters.
BODY_LIST_CHARS = 16

#: Height of the zoom slider's track, in button-heights.  The whole log-space
#: zoom range (14 decades) has to fit in it, so a short slider is unusable --
#: every pixel dragged jumps the view by more than a button click would.  7 is
#: as tall as it can go: at the smallest window, coarsest usable grid, that
#: leaves 8px to spare against the panel's content band; any taller and the
#: zoom cluster is pushed off the bottom of the window there.
SLIDER_HEIGHT_NOTCHES = 7

TOOL_BUTTONS = (
    ("tool.look", "Look", "look", "1", "Select and inspect bodies"),
    ("tool.throw", "Throw", "throw", "2", "Throw a new body into the system"),
    ("tool.edit", "Edit", "edit", "3", "Change what a body is made of"),
)

#: (id, icon, label, hotkey, tooltip).  A button with no icon draws its label
#: instead, which is how the help button gets a question mark that is actually
#: legible at this size rather than a hand-drawn one.
VIEW_BUTTONS = (
    ("view.trails", "trails", "", "T", "Orbit trails"),
    ("view.sizes", "sizes", "", "R", "Real sizes -- everything becomes a dot"),
    ("view.labels", "labels", "", "L", "Name labels"),
    ("view.grid", "grid", "", "G", "Chunkier pixels"),
    ("view.window", "window", "", "W", "Window size"),
    ("view.fullscreen", "fullscreen", "", "Cmd+F", "Fullscreen"),
    ("view.help", "", "?", "?", "Every keyboard shortcut"),
)

#: Controls that need a DisplayContext to do anything, and so are drawn
#: disabled when there is not one (headless tests, and the smoke test).
NEEDS_DISPLAY = ("view.grid", "view.window", "view.fullscreen")


def _slider_fraction(control, pos):
    """Where ``pos`` sits along a vertical slider's track, 0 at the bottom.

    Through ``slider_track``, which is the same span ``draw_slider`` paints the
    thumb on -- reading the raw rect here instead would put the thumb a few
    pixels from where clicking it lands, so grabbing the thumb would jump the
    value before the drag had even begun.
    """
    rect = control.rect
    if rect is None or rect.height <= 0:
        return control.value
    top, height = slider_track(rect)
    if height <= 0:
        return control.value
    frac = 1.0 - (pos[1] - top) / height
    return min(1.0, max(0.0, frac))


def layout_row(controls, x, y, metrics, width):
    """Place one row of controls left to right inside ``width``.

    Compact controls keep their square footprint; everything else shares what
    is left over, so a row of one button fills the panel and a row of three
    splits it evenly.  Returns the list of (control, rect).
    """
    gap = metrics.gap
    fixed = sum(metrics.icon_btn_w for c in controls if c.compact)
    flexible = [c for c in controls if not c.compact]
    spare = width - fixed - gap * (len(controls) - 1)
    each = max(metrics.icon_btn_w, spare // len(flexible)) if flexible else 0

    placed = []
    cursor = x
    for k, control in enumerate(controls):
        if control.compact:
            w = metrics.icon_btn_w
        elif k == len(controls) - 1:
            w = max(metrics.icon_btn_w, x + width - cursor)
        else:
            w = each
        placed.append((control, pygame.Rect(cursor, y, w, metrics.btn_h)))
        cursor += w + gap
    return placed


class UiLayer:
    """Owns the controls for the current frame and the mouse state over them."""

    def __init__(self, font, hud=None):
        self.font = font
        self.hud = hud
        self.controls = []
        #: (rect, title) slabs drawn under the controls.  They also swallow
        #: clicks where there is no button, so a press on a panel's background
        #: cannot pan the camera behind it.
        self.panels = []
        self.metrics = None
        self.mouse = (-1, -1)
        self.hover_id = None
        #: Rectangle of the help panel while it is open.  Its presence is what
        #: makes the help modal: a click anywhere else puts it away, the way
        #: every dialog a child has ever met behaves.
        self.modal = None
        self._modal_control = None
        self._pressed = None
        self._press_on_ui = False
        self._repeat_at = 0
        self._cursor_is_hand = None

    # --- layout ------------------------------------------------------------------

    def build(self, state, size, ctx=None):
        """Rebuild every control for the current frame."""
        width, height = size
        m = Metrics(self.font, width, height)
        self.metrics = m
        self.controls = []
        self.panels = []

        self._build_bottom_hotbar(state, m, ctx)
        self._build_view_buttons(state, m, ctx)
        self._build_zoom_cluster(state, m)
        self._build_inspector_actions(state, m)
        self._build_inspector_expand(state, m)
        self._build_tool_context(state, m)
        self._build_body_list(state, m)
        self._build_help(state, m)
        return self.controls

    def _build_help(self, state, m):
        """The close button on the help panel, and the panel's own footprint."""
        self.modal = None
        self._modal_control = None
        if not state.help_visible or self.hud is None:
            return
        self.modal = self.hud.help_rect(m.width, m.height)
        self._modal_control = Control("view.help", icon="close", compact=True,
                                      tooltip="Close  [Esc]")
        self._place([(self._modal_control,
                      pygame.Rect(self.modal.right - m.icon_btn_w - m.gap,
                                  self.modal.y + m.gap, m.icon_btn_w, m.btn_h))])

    def _panel(self, rect, title=None):
        self.panels.append((rect, title))
        return rect

    def _place(self, placed):
        for control, rect in placed:
            control.rect = rect
            self.controls.append(control)

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

        top = bottom
        height = len(rows) * m.btn_h + (len(rows) - 1) * m.gap + m.line
        self._panel(pygame.Rect(x - m.gap, top - m.gap, width + 2 * m.gap, height),
                    tool.context_title)

        y = top + m.line
        for row in rows:
            self._place(layout_row(row, x, y, m, width))
            y += m.btn_h + m.gap
        return width

    def _build_view_buttons(self, state, m, ctx):
        """Icon toggles along the right of the top bar, then the way out.

        The quit button stays even now the sandbox opens in an ordinary window
        with a close box of its own: fullscreen has no window chrome at all, and
        a mouse that cannot leave the program is not a state to be one keystroke
        away from.
        """
        y = (m.top_h - m.btn_h) // 2
        if state.quit_armed:
            quit_control = Control("app.quit", label="Really quit?", icon="power",
                                   hotkey="Cmd+Q", danger=True, active=True,
                                   tooltip="Click again to close the sandbox -- "
                                           "nothing is saved")
            quit_w = m.button_width(quit_control)
        else:
            quit_control = Control("app.quit", icon="power", hotkey="Cmd+Q",
                                   compact=True, danger=True,
                                   tooltip="Close the sandbox")
            quit_w = m.icon_btn_w
        x = m.width - m.gap - quit_w
        self._place([(quit_control, pygame.Rect(x, y, quit_w, m.btn_h))])
        x -= m.gap * 3

        flags = {
            "view.trails": state.view.show_trails,
            "view.sizes": state.view.real_size,
            "view.labels": state.view.show_labels,
            "view.grid": ctx is not None and ctx.s != 1,
            "view.window": ctx is not None and not ctx.fullscreen
            and ctx.fraction_index != 0,
            "view.fullscreen": ctx is not None and ctx.fullscreen,
            "view.help": state.help_visible,
        }
        x -= m.icon_btn_w
        for cid, icon, label, key, tip in reversed(VIEW_BUTTONS):
            enabled = ctx is not None or cid not in NEEDS_DISPLAY
            if cid == "view.grid" and ctx is not None:
                # Nothing to step through in a window only the finest grid
                # fits in, and a button that does nothing is worse than one
                # that says so.
                enabled = len(ctx.usable_grids) > 1
            if cid == "view.window" and ctx is not None:
                # The one tooltip that has to say what pressing again will do:
                # a cycle button is otherwise a button whose effect you can only
                # find out by pressing it.
                nxt = WINDOW_FRACTIONS[(ctx.fraction_index + 1)
                                       % len(WINDOW_FRACTIONS)][1]
                tip = "Window size: %s -- click for %s" % (
                    ctx.window_size_label, nxt.lower())
            control = Control(cid, label=label, icon=icon, hotkey=key, tooltip=tip,
                              compact=True, active=flags.get(cid, False),
                              enabled=enabled)
            self._place([(control, pygame.Rect(x, y, m.icon_btn_w, m.btn_h))])
            x -= m.icon_btn_w + m.gap

    def _build_zoom_cluster(self, state, m):
        """Zoom and go-home, bottom right, where every map in the world puts them.

        Not everyone has a wheel.  Without these, a trackpad with no scroll
        gesture cannot zoom at all, which would make most of the sandbox
        unreachable by mouse -- the one thing this interface exists to prevent.

        The slider sits between the two buttons, in the same order a map's own
        zoom control uses: drag toward + to zoom in, toward - to zoom out.  It
        is a second route to the same ``Camera``, not a third zoom range of its
        own -- a fast, coarse jump the buttons then refine one notch at a time.
        """
        frac = state.camera.zoom_fraction
        items = [
            (Control("view.zoom", icon="plus", payload=1, hotkey="+", compact=True,
                     repeat=True, tooltip="Zoom in"), m.btn_h),
            (Control("view.zoom_slider", slider=True, value=frac, payload=frac,
                     tooltip="Drag to zoom"), SLIDER_HEIGHT_NOTCHES * m.btn_h),
            (Control("view.zoom", icon="minus", payload=-1, hotkey="-", compact=True,
                    repeat=True, tooltip="Zoom out"), m.btn_h),
            (Control("view.home", icon="home", hotkey="H", compact=True,
                    tooltip="Back to the whole solar system"), m.btn_h),
        ]
        total = sum(h for _, h in items) + m.gap * (len(items) - 1)
        x = m.width - m.pad - m.icon_btn_w
        y = m.content_bottom - total
        self._panel(pygame.Rect(x - m.gap, y - m.gap, m.icon_btn_w + 2 * m.gap,
                                total + m.gap))
        for control, h in items:
            self._place([(control, pygame.Rect(x, y, m.icon_btn_w, h))])
            y += h + m.gap

    def _build_transport(self, state, m, top):
        """Scenario, restart, speed and play -- the lower row of the hotbar.

        Takes its own ``top`` rather than reading ``m.transport_y`` directly,
        because it is now the *second* of two rows in one bar;
        ``_build_bottom_hotbar`` is what owns the bar's single panel and the
        tool row above this one.
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

    def _build_body_list(self, state, m):
        """A scrollable list of everything in the system, selectable by name.

        This is what replaces Tab for someone who has never pressed Tab: the
        planets have names, the names are on screen, and they are clickable.
        """
        if not state.show_body_list:
            return
        world = state.sim.world
        width = m.font.width("X") * BODY_LIST_CHARS + 2 * m.pad
        x = m.pad
        top = m.content_top
        header = m.btn_h + m.gap
        row_h = m.line + 2
        available = m.content_bottom - top - header - m.gap
        visible = max(1, min(max(world.n, 1), max(1, available // row_h)))
        offset = state.clamp_body_list_offset(visible)

        panel = pygame.Rect(x, top, width, header + visible * row_h + m.gap)
        self._panel(panel, "Bodies")

        buttons = [Control("panel.bodies.close", icon="close", compact=True,
                           tooltip="Close the list")]
        if world.n > visible:
            buttons = [
                Control("list.scroll", icon="up", payload=-1, compact=True,
                        repeat=True, tooltip="Scroll up", enabled=offset > 0),
                Control("list.scroll", icon="down", payload=1, compact=True,
                        repeat=True, tooltip="Scroll down",
                        enabled=offset + visible < world.n),
            ] + buttons
        bx = panel.right - m.gap
        for control in reversed(buttons):
            bx -= m.icon_btn_w
            self._place([(control, pygame.Rect(bx, top + m.gap // 2,
                                               m.icon_btn_w, m.btn_h))])
            bx -= m.gap

        y = top + header
        for i in range(offset, min(offset + visible, world.n)):
            body = world.bodies[i]
            control = Control("select.body", label=body.name, payload=body.id,
                              tooltip="Select %s" % body.name,
                              active=body.id == state.selected_id)
            self._place([(control, pygame.Rect(panel.x + m.gap, y,
                                               width - 2 * m.gap, row_h - 2))])
            y += row_h

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

    # --- hit testing -------------------------------------------------------------

    def control_at(self, pos):
        """Topmost clickable control under a point, or None.

        Reversed because later controls are drawn over earlier ones, and what is
        on top is what a click should reach.
        """
        for control in reversed(self.controls):
            if control.clickable and control.rect is not None \
                    and control.rect.collidepoint(pos):
                return control
        return None

    def over_panel(self, pos):
        if self.modal is not None and self.modal.collidepoint(pos):
            return True
        return any(rect.collidepoint(pos) for rect, _ in self.panels)

    def blocks(self, pos):
        """True if the interface owns this point and the world must not see it."""
        return self.control_at(pos) is not None or self.over_panel(pos)

    # --- events ------------------------------------------------------------------

    def on_event(self, event, state, ctx=None):
        """Handle one event.  Returns True when the world must not also see it."""
        if event.type == pygame.MOUSEMOTION:
            self.mouse = event.pos
            if self._pressed is not None and self._pressed.slider:
                # A slider tracks the cursor for as long as the button is down,
                # on or off its own rect -- the same "drag stays live" rule a
                # click-and-hold panel drag already gets below.
                self._pressed.payload = _slider_fraction(self._pressed, event.pos)
                self.perform(self._pressed, state, ctx)
                return True
            control = self.control_at(event.pos)
            self.hover_id = control.id if control is not None else None
            return self._press_on_ui

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.mouse = event.pos
            control = self.control_at(event.pos)
            if self.modal is not None and control is not self._modal_control:
                # Anything outside the help panel puts the help away and is
                # spent doing so, rather than also doing whatever it hit.
                state.help_visible = False
                self._press_on_ui = True
                return True
            if control is not None:
                self._pressed = control
                self._press_on_ui = True
                if control.slider:
                    # Clicking anywhere on the track jumps the thumb there, the
                    # way every slider does -- not just the notch a stepper
                    # click would give.
                    control.payload = _slider_fraction(control, event.pos)
                    self.perform(control, state, ctx)
                elif control.repeat:
                    # Steppers act on the way down and keep going while held.
                    # Waiting for the release would make them feel broken.
                    self.perform(control, state, ctx)
                    self._repeat_at = pygame.time.get_ticks() + REPEAT_DELAY_MS
                return True
            # A press that misses the quit button is an answer of "no" too.
            state.cancel_quit()
            if self.over_panel(event.pos):
                self._press_on_ui = True
                return True
            self._press_on_ui = False
            return False

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            pressed, self._pressed = self._pressed, None
            on_ui, self._press_on_ui = self._press_on_ui, False
            if pressed is not None:
                # Releasing away from the button cancels it, the way every other
                # interface behaves: a mis-click can be dragged off and undone.
                # A slider already acted continuously through the drag, so its
                # release needs no catch-up click of its own.
                if not pressed.repeat and not pressed.slider \
                        and pressed.rect.collidepoint(event.pos):
                    self.perform(pressed, state, ctx)
                return True
            return on_ui

        if event.type == pygame.MOUSEWHEEL and self.over_panel(self.mouse):
            state.scroll_body_list(-event.y)
            return True

        return False

    def update(self, state, ctx=None, now=None):
        """Fire held steppers.  Call once per frame, after events."""
        if self._pressed is None or not self._pressed.repeat:
            return 0
        if not self._pressed.enabled:
            return 0
        now = pygame.time.get_ticks() if now is None else now
        fired = 0
        while now >= self._repeat_at and fired < MAX_REPEATS_PER_FRAME:
            self.perform(self._pressed, state, ctx)
            self._repeat_at += REPEAT_INTERVAL_MS
            fired += 1
        if fired >= MAX_REPEATS_PER_FRAME:
            self._repeat_at = now + REPEAT_INTERVAL_MS
        return fired

    # --- actions -----------------------------------------------------------------

    def perform(self, control, state, ctx=None):
        """Run the action behind a control.  Returns True if anything happened.

        A control that is drawn disabled does nothing here either.  Hit-testing
        already skips them, but this is the door every other caller comes
        through -- the held-stepper repeat, the tests -- and "greyed out" has to
        mean the same thing at all of them.
        """
        if not control.clickable:
            return False
        cid, payload = control.id, control.payload

        # Doing anything else at all is an answer of "no" to a waiting quit.
        if cid != "app.quit":
            state.cancel_quit()

        if cid == "app.quit":
            state.request_quit()
            return True

        if cid.startswith("tool."):
            state.set_tool(tools_module.TOOL_BY_NAME[cid.split(".", 1)[1]])
            return True
        if cid == "panel.bodies":
            state.show_body_list = not state.show_body_list
            return True
        if cid == "panel.bodies.close":
            state.show_body_list = False
            return True
        if cid == "list.scroll":
            state.scroll_body_list(payload)
            return True
        if cid == "select.body":
            state.select(payload)
            return True

        if cid == "view.trails":
            state.view.show_trails = not state.view.show_trails
            return True
        if cid == "view.sizes":
            state.view.real_size = not state.view.real_size
            return True
        if cid == "view.labels":
            state.view.show_labels = not state.view.show_labels
            return True
        if cid == "view.grid":
            state.cycle_grid(ctx)
            return True
        if cid == "view.window":
            if ctx is None:
                return False
            state.cycle_window_size(ctx)
            return True
        if cid == "view.fullscreen":
            if ctx is None:
                return False
            ctx.toggle_fullscreen()
            state.on_resize(ctx)
            return True
        if cid == "inspector.expand":
            state.toggle_inspector_expanded()
            return True
        if cid == "view.help":
            state.help_visible = not state.help_visible
            return True
        if cid == "view.zoom":
            state.camera.zoom_by(payload)
            return True
        if cid == "view.zoom_slider":
            state.camera.set_zoom_fraction(payload)
            return True
        if cid == "view.home":
            state.camera.reset_view()
            return True

        if cid == "time.pause":
            state.toggle_pause()
            return True
        if cid == "time.slower":
            state.nudge_time(-1)
            return True
        if cid == "time.faster":
            state.nudge_time(1)
            return True
        if cid == "scenario.cycle":
            state.cycle_scenario()
            return True
        if cid == "sim.reset":
            state.reset()
            return True

        if cid == "body.follow":
            state.toggle_follow()
            return True
        if cid == "body.flyto":
            state.fly_to_selected()
            return True
        if cid == "body.stop":
            state.stop_selected()
            return True
        if cid == "body.delete":
            state.delete_selected()
            return True

        tool = state.tool
        if tool is not None:
            return bool(tool.perform(control, state))
        return False

    # --- drawing -----------------------------------------------------------------

    def draw(self, surface, state):
        """Panels first, then every control, then the help, then one tooltip."""
        for rect, title in self.panels:
            draw_panel(surface, rect)
            if title:
                self.font.draw(surface, rect.x + self.metrics.pad,
                               rect.y + self.metrics.gap, title, palette.TEXT_DIM)

        hovered = self.control_at(self.mouse)
        self._set_cursor(hovered is not None)
        for control in self.controls:
            if control is self._modal_control:
                continue
            draw_button(surface, self.font, control,
                        hovered=control is hovered,
                        pressed=control is self._pressed)

        if self.modal is not None:
            # Dim everything else so the help reads as one thing to deal with,
            # then put the panel and its close button on top of the dimming.
            veil = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            veil.fill((0, 0, 0, 128))
            surface.blit(veil, (0, 0))
            self.hud.draw_help(surface)
            draw_button(surface, self.font, self._modal_control,
                        hovered=self._modal_control is hovered,
                        pressed=self._modal_control is self._pressed)

        if hovered is not None:
            draw_tooltip(surface, self.font, hovered.hint(), hovered.rect,
                         prefer_above=hovered.rect.centery > surface.get_height() // 2)

    def _set_cursor(self, over_control):
        """Hand over a button, arrow everywhere else.

        Only on the frame it changes, because setting the cursor is a trip into
        SDL.  Wrapped because some drivers -- the headless one the tests run on
        among them -- have no cursors at all, and that must not end the frame.
        """
        if over_control == self._cursor_is_hand:
            return
        self._cursor_is_hand = over_control
        try:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND if over_control
                                    else pygame.SYSTEM_CURSOR_ARROW)
        except (pygame.error, AttributeError):
            pass
