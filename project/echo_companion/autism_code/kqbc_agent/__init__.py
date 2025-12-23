"""
KQBC Agent package for the ultimate unified system.

This package integrates the speech companion with an advanced cognitive
substrate.  It provides a configuration object, a speech loop tied to
an embedded AGI system and a Tkinter GUI for local operation.  The
agent leverages the unified system described in the accompanying
blueprints: a relational matrix, thought engines, emotional chemistry
and memory.  These components are orchestrated by ``AGISystem`` from
``unified_system_agi_core`` to provide a richer context for guiding
language practice and behavioural support.
"""

from .config import CompanionConfig, CONFIG
from .agent import KQBCAgent
from .speech_loop import SpeechLoop

# Note: The GUI and CLI modules are not imported here to avoid
# loading optional dependencies (e.g. tkinter) at import time.  To
# access the GUI or CLI, import ``kqbc_agent.tk_gui`` or
# ``kqbc_agent.cli`` directly.

__all__ = [
    "CompanionConfig",
    "CONFIG",
    "KQBCAgent",
    "SpeechLoop",
]