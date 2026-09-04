"""The physics layer must stay headless.

If a physics module ever imports pygame, the orbital mechanics stop being
testable without a display and the whole test suite becomes hostage to SDL.
This test is the enforcement.
"""

import importlib
import subprocess
import sys

PHYSICS_MODULES = [
    "src.units", "src.body", "src.materials", "src.world",
    "src.integrator", "src.orbits", "src.scenarios",
    "src.collisions", "src.destruction", "src.thermal", "src.sim", "src.edits",
    "src.maneuver",
]


def test_physics_layer_never_imports_pygame():
    # Run in a subprocess so an earlier test importing pygame cannot mask this.
    code = (
        "import sys;"
        "mods=%r;"
        "[__import__(m) for m in mods];"
        "assert 'pygame' not in sys.modules, sorted(sys.modules)"
    ) % (PHYSICS_MODULES,)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_every_physics_module_imports():
    for name in PHYSICS_MODULES:
        importlib.import_module(name)
