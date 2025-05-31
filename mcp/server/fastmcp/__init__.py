"""FastMCP - A more ergonomic interface for MCP servers."""

from importlib.metadata import version

from .server import Context, FastMCP
from .utilities.types import Image

__version__ = "1.8.1"
__all__ = ["FastMCP", "Context", "Image"]
