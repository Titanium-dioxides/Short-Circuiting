"""Make the repo root importable in tests without installing the package
(the system Python is externally managed, so `pip install -e .` is not
available). `pyproject.toml` remains the canonical packaging definition."""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
