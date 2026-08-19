#!/usr/bin/env python3
"""Run the whole test suite in one command."""

import sys

import pytest

if __name__ == "__main__":
    sys.exit(pytest.main(["-q", "tests"]))
