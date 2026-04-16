"""OpenCAS: the Computational Autonomous System."""

from .sqlite_compat import patch_aiosqlite_for_python314

patch_aiosqlite_for_python314()

__version__ = "0.1.0"
