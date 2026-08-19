"""Keyboard and mouse routing.

Three layers, in this order: the buttons, then the active tool, then the global
keys.  Each may consume an event and stop it reaching the next.

The interface has to come first.  The throw tool reads every press in the window
as an aim, so if the tool went first, clicking "Delete" would fire a comet
through the button on the way past.  Tool before globals is the same rule one
level down: a tool can capture a drag without the camera panning underneath it.
"""

import pygame

from .tools import TOOLS


def handle_event(event, state, ctx, ui=None):
    """Route one pygame event.  ``ctx`` and ``ui`` may be None in tests."""
    if event.type == pygame.QUIT:
        state.running = False
        return

    if ui is not None and ui.on_event(event, state, ctx):
        return

    if state.tool is not None and state.tool.on_event(event, state):
        return

    if event.type == pygame.MOUSEWHEEL:
        mx, my = pygame.mouse.get_pos()
        state.camera.zoom_by(event.y, mx, my)
        return

    if event.type != pygame.KEYDOWN:
        return

    key = event.key
    command = event.mod & (pygame.KMOD_META | pygame.KMOD_CTRL)

    if command and key == pygame.K_q:
        state.running = False
    elif command and key == pygame.K_f:
        if ctx is not None:
            ctx.toggle_fullscreen()
            state.on_resize(ctx)
    elif key in TOOLS:
        state.set_tool(TOOLS[key])
    elif key == pygame.K_SPACE:
        state.toggle_pause()
    elif key == pygame.K_RIGHTBRACKET:
        state.nudge_time(1)
    elif key == pygame.K_LEFTBRACKET:
        state.nudge_time(-1)
    elif key == pygame.K_TAB:
        state.cycle_selection(-1 if event.mod & pygame.KMOD_SHIFT else 1)
    elif key == pygame.K_f:
        state.follow_selected()
    elif key == pygame.K_t:
        state.view.show_trails = not state.view.show_trails
    elif key == pygame.K_r:
        state.view.real_size = not state.view.real_size
    elif key == pygame.K_l:
        state.view.show_labels = not state.view.show_labels
    elif key == pygame.K_g:
        state.cycle_grid(ctx)
    elif key == pygame.K_b:
        state.show_body_list = not state.show_body_list
    elif key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
        state.camera.zoom_by(1)
    elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
        state.camera.zoom_by(-1)
    elif key == pygame.K_h:
        state.camera.reset_view()
    elif key == pygame.K_0:
        state.stop_selected()
    elif key in (pygame.K_DELETE, pygame.K_BACKSPACE):
        state.delete_selected()
    elif key == pygame.K_F5:
        state.reset()
    elif key in (pygame.K_SLASH, pygame.K_QUESTION):
        state.help_visible = not state.help_visible
    elif key == pygame.K_ESCAPE:
        if state.quit_armed:
            state.cancel_quit()
        elif state.help_visible:
            state.help_visible = False
        else:
            state.selected_id = None
