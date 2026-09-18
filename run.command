#!/bin/bash
# Double-click me in Finder.
cd "$(dirname "$0")" || exit 1

if [ ! -d .venv ]; then
    echo "First run: setting up..."
    python3 -m venv .venv || exit 1
    ./.venv/bin/pip install --quiet --upgrade pip
fi

# pygame and pygame-ce unpack into the same "pygame" directory and cannot
# coexist.  A venv built before the switch to pygame-ce has the old one; it
# must go before the requirements are installed, or pip leaves a broken mix.
if ./.venv/bin/pip show pygame >/dev/null 2>&1; then
    echo "Replacing pygame with pygame-ce..."
    ./.venv/bin/pip uninstall --quiet -y pygame
fi
./.venv/bin/pip install --quiet -r requirements.txt || exit 1

exec ./.venv/bin/python -m src.app.main
