"""Keyboard and mouse routing.

Three layers, in this order: the buttons, then the active tool, then the global
keys.  Each may consume an event and stop it reaching the next.

The interface has to come first.  The throw tool reads every press in the window
as an aim, so if the tool went first, clicking "Delete" would fire a comet
through the button on the way past.  Tool before globals is the same rule one
level down: a tool can capture a drag without the camera panning underneath it.
"""

import pygame

from .state import DIALOG_SAVES, DIALOG_START
from .tools import TOOLS


def handle_event(event, state, ctx, ui=None):
    """Route one pygame event.  ``ctx`` and ``ui`` may be None in tests."""
    if event.type == pygame.QUIT:
        state.running = False
        return

    # Before the interface, not after: every control was laid out against the
    # old size, and one of them would otherwise get a click meant for a rectangle
    # that has already moved.  A pygame.Window reports its new size as
    # WINDOWSIZECHANGED (``x``, ``y``); VIDEORESIZE never fires for it.
    if event.type == pygame.WINDOWSIZECHANGED:
        if ctx is not None:
            ctx.resize((event.x, event.y))
            state.on_resize(ctx)
        return

    # A dialog owns every event.  The interface still gets the mouse, so its
    # buttons work as buttons; the keys go to the dialog and stop there --
    # the throw tool must not fire through the slot list, and a name being
    # typed must not toggle trails with every T.
    if state.dialog is not None:
        if event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN,
                          pygame.MOUSEBUTTONUP, pygame.MOUSEWHEEL):
            if ui is not None:
                ui.on_event(event, state, ctx)
        elif event.type == pygame.KEYDOWN:
            dialog_key(event, state, ctx)
        return

    if ui is not None and ui.on_event(event, state, ctx):
        return

    if state.tool is not None and state.tool.on_event(event, state):
        return

    if event.type == pygame.MOUSEWHEEL:
        # A wheel event carries no position of its own, so it has to be asked
        # for -- and asked for through ctx, which is what puts it in the same
        # internal pixels every other coordinate in the program is in.
        mx, my = ctx.mouse_pos() if ctx is not None else pygame.mouse.get_pos()
        state.camera.zoom_by(event.y, mx, my)
        return

    if event.type != pygame.KEYDOWN:
        return

    key = event.key
    command = event.mod & (pygame.KMOD_META | pygame.KMOD_CTRL)

    if command and key == pygame.K_q:
        state.running = False
    elif command and key == pygame.K_w:
        # Cmd+W means close the window everywhere else on this platform, and
        # would otherwise fall through to the plain W below and resize it --
        # the one keystroke where guessing wrong is unrecoverable, since
        # nothing here is saved.  Same two-press confirmation as the quit
        # button, for the same reason.
        state.request_quit()
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
    elif key == pygame.K_y:
        state.explain.enabled = not state.explain.enabled
    elif key == pygame.K_g:
        state.cycle_grid(ctx)
    elif key == pygame.K_w:
        state.cycle_window_size(ctx)
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
    elif key == pygame.K_s:
        state.toggle_saves()
    elif key in (pygame.K_SLASH, pygame.K_QUESTION):
        state.help_visible = not state.help_visible
    elif key == pygame.K_ESCAPE:
        if state.quit_armed:
            state.cancel_quit()
        elif state.help_visible:
            state.help_visible = False
        else:
            state.selected_id = None


def dialog_key(event, state, ctx):
    """Keys while a dialog is up.

    Cmd+Q and Cmd+F still mean what they mean everywhere.  A text field
    with focus takes everything else: Enter keeps the name, Esc drops the
    edit (and only the edit -- a second Esc closes the dialog), Backspace
    deletes, and any printable character is typed.  Otherwise Esc disarms
    a waiting Overwrite? before it closes anything, the start menu answers
    to its three letters, and S puts the in-game list away.
    """
    key = event.key
    command = event.mod & (pygame.KMOD_META | pygame.KMOD_CTRL)
    d = state.dialog
    if command and key == pygame.K_q:
        state.running = False
        return
    if command and key == pygame.K_f:
        if ctx is not None:
            ctx.toggle_fullscreen()
            state.on_resize(ctx)
        return
    if d.renaming is not None:
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            state.rename_commit()
        elif key == pygame.K_ESCAPE:
            state.rename_cancel()
        elif key == pygame.K_BACKSPACE:
            state.rename_backspace()
        elif not command and getattr(event, "unicode", ""):
            state.rename_type(event.unicode)
        return
    if key == pygame.K_ESCAPE:
        if d.armed is not None:
            state.disarm()
        else:
            state.close_dialog()
    elif d.kind == DIALOG_START:
        if key == pygame.K_n:
            state.start_new()
        elif key == pygame.K_l:
            state.open_load()
        elif key == pygame.K_q:
            state.quit_now()
    elif d.kind == DIALOG_SAVES and key == pygame.K_s:
        state.toggle_saves()
