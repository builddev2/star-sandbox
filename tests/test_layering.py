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
    "src.maneuver", "src.save", "src.measure",
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


def test_nothing_calls_from_display_module():
    """``pygame.Window.from_display_module()`` hands back an object that
    destroys the real window when it is garbage-collected; the program then
    segfaults some frames later, far from the call that did it.

    Checked as an actual call, not a text search: ``src/app/config.py``'s own
    module docstring names the method to document why it is forbidden, which
    a plain substring search would misreport as an offender.
    """
    import ast
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "src"
    offenders = []
    for path in root.rglob("*.py"):
        if "from_display_module" not in path.read_text():
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "from_display_module"):
                offenders.append(path)
    assert offenders == []
