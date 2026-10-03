"""Thin public API. Config and chemistry live here; the C kernel only steps."""

from .api import Run, setup
from .config import full_schema_example, normalize, shock_tube

__all__ = [
    "setup",
    "Run",
    "shock_tube",
    "normalize",
    "full_schema_example",
]
__version__ = "0.1.0"
