"""Interaction tools.

Each tool owns its own event handling and its own overlay drawing, so adding one
is a new class rather than another branch in a growing event handler.  Exactly
one tool is active; the HUD's hint line is whatever it says it is.
"""

import pygame

#: A press-and-release within this many pixels counts as a click, not a drag.
CLICK_SLOP = 4
#: Two clicks within this many milliseconds on the same body zoom to it.
DOUBLE_CLICK_MS = 400


class Tool:
    name = "Tool"
    hint = ""

    def on_event(self, event, state):
        """Return True if the event was consumed."""
        return False

    def draw_overlay(self, surface, state, font):
        pass

    def on_exit(self, state):
        pass


class SelectTool(Tool):
    name = "Look"
    hint = ("Click a body to inspect it  -  drag to pan  -  wheel to zoom  -  "
            "double-click to fly to it  -  ? for controls")

    def __init__(self):
        self._press = None
        self._dragged = False
        self._last_click = (None, 0)

    def on_event(self, event, state):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._press = event.pos
            self._dragged = False
            return True

        if event.type == pygame.MOUSEMOTION and self._press is not None:
            if event.buttons[0]:
                if (abs(event.pos[0] - self._press[0]) > CLICK_SLOP
                        or abs(event.pos[1] - self._press[1]) > CLICK_SLOP):
                    self._dragged = True
                if self._dragged:
                    state.camera.pan_pixels(event.rel[0], event.rel[1])
                return True

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            pressed, self._press = self._press, None
            if pressed is None or self._dragged:
                return True
            self._click(event.pos, state)
            return True

        return False

    def _click(self, pos, state):
        index = state.pick_body_at(pos[0], pos[1])
        if index is None:
            state.selected_id = None
            self._last_click = (None, 0)
            return

        body_id = state.sim.world.bodies[index].id
        now = pygame.time.get_ticks()
        previous_id, previous_time = self._last_click
        if previous_id == body_id and now - previous_time < DOUBLE_CLICK_MS:
            state.camera.frame_body(state.sim.world, index)
            state.camera.follow_id = body_id
        state.selected_id = body_id
        self._last_click = (body_id, now)


TOOLS = {
    pygame.K_1: SelectTool,
}
