"""Bootstrap module for OpenCAS: staged startup pipeline."""

from .config import BootstrapConfig
from .context import BootstrapContext

__all__ = ["BootstrapConfig", "BootstrapContext", "BootstrapPipeline"]


def __getattr__(name: str):
    if name == "BootstrapPipeline":
        from .pipeline import BootstrapPipeline

        return BootstrapPipeline
    raise AttributeError(name)
