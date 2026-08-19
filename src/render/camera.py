"""The mapping between AU and pixels.

Zoom is stored as its logarithm so that a wheel notch is a constant *ratio*
rather than a constant amount -- the only way a single control can span from
Neptune's orbit down to the surface of a moon.

Following stores a body **id**, never an index: World.remove swap-removes, so an
index held across a frame can silently start pointing at a different body.
"""

import math

import numpy as np

#: Ratio per wheel notch.
ZOOM_STEP = 1.18
#: Widest and closest views, in pixels per AU, applied as log bounds.
MIN_ZOOM = 1e-2
MAX_ZOOM = 1e12


class Camera:
    def __init__(self, width, height, au_across=70.0):
        self.width = width
        self.height = height
        self.centre = np.zeros(2, dtype=np.float64)
        self.log_zoom = math.log(width / au_across)
        self.follow_id = None

    # --- zoom -------------------------------------------------------------------

    @property
    def zoom(self):
        """Pixels per AU."""
        return math.exp(self.log_zoom)

    def zoom_by(self, steps, sx=None, sy=None):
        """Zoom by ``steps`` notches, keeping the world point under (sx, sy) fixed."""
        anchor = None
        if sx is not None and sy is not None:
            anchor = self.screen_to_world(sx, sy)

        self.log_zoom += steps * math.log(ZOOM_STEP)
        self.log_zoom = min(max(self.log_zoom, math.log(MIN_ZOOM)), math.log(MAX_ZOOM))

        if anchor is not None:
            # Re-centre so the anchored world point lands back under the cursor.
            moved = self.screen_to_world(sx, sy)
            self.centre += anchor - moved

    # --- panning and following ---------------------------------------------------

    def pan_pixels(self, dx, dy):
        """Drag the view by a screen delta.  Dragging right moves the world right."""
        self.centre[0] -= dx / self.zoom
        self.centre[1] += dy / self.zoom
        self.follow_id = None

    def update_follow(self, world):
        """Snap the centre onto the followed body, or forget it if it is gone."""
        if self.follow_id is None:
            return
        index = world.index_of(self.follow_id)
        if index is None:
            self.follow_id = None
            return
        self.centre[:] = world.pos[index]

    def frame_body(self, world, index, radii=8.0):
        """Centre on a body and zoom so it fills a sensible part of the view."""
        self.centre[:] = world.pos[index]
        span = max(world.radius[index] * radii, 1e-9)
        self.log_zoom = math.log(min(self.height, self.width) * 0.5 / span)
        self.log_zoom = min(max(self.log_zoom, math.log(MIN_ZOOM)),
                            math.log(MAX_ZOOM))

    # --- projection --------------------------------------------------------------

    def world_to_screen(self, point):
        zoom = self.zoom
        return (self.width * 0.5 + (float(point[0]) - self.centre[0]) * zoom,
                self.height * 0.5 - (float(point[1]) - self.centre[1]) * zoom)

    def world_to_screen_array(self, points):
        """Vectorised projection for trails and predicted orbits."""
        zoom = self.zoom
        points = np.asarray(points, dtype=np.float64)
        out = np.empty((len(points), 2), dtype=np.float64)
        out[:, 0] = self.width * 0.5 + (points[:, 0] - self.centre[0]) * zoom
        out[:, 1] = self.height * 0.5 - (points[:, 1] - self.centre[1]) * zoom
        return out

    def screen_to_world(self, sx, sy):
        zoom = self.zoom
        return np.array([
            self.centre[0] + (sx - self.width * 0.5) / zoom,
            self.centre[1] - (sy - self.height * 0.5) / zoom,
        ])
