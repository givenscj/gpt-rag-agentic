import sys
import os

# Add the parent directory to the system path - override any library imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import mcp
import autogen_ext

from telemetry import Telemetry

Telemetry.configure_basic()