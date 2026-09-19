"""Test-wide environment: no window and no audio device.

Every render/app test module already sets SDL_VIDEODRIVER=dummy at its top.
The audio driver is set here for the fifteen modules that call
``pygame.init()``, which initialises the mixer: on a machine whose audio
device is in a bad state CoreAudio takes fifteen seconds to fail, and the
sandbox never plays a sound.
"""

import os

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
