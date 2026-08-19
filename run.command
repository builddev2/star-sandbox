#!/bin/bash
# Double-click me in Finder.
cd "$(dirname "$0")" || exit 1

if [ ! -d .venv ]; then
    echo "First run: setting up..."
    python3 -m venv .venv || exit 1
    ./.venv/bin/pip install --quiet --upgrade pip
    ./.venv/bin/pip install --quiet -r requirements.txt || exit 1
fi

exec ./.venv/bin/python -m src.app.main
