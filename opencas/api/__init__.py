"""API module for OpenCAS: external interfaces and LLM gateway adapter."""

from .llm import LLMClient
from .server import create_app

__all__ = ["LLMClient", "create_app"]
