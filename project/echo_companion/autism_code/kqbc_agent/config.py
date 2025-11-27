"""
Configuration for the KQBC Agent.

This module defines a configuration class used throughout the KQBC agent
package.  It mirrors the structure of the companion configuration but
is defined separately so that the agent can evolve independently.  The
configuration specifies identifying information about the child and
caregiver and the file system locations of log files.  These values
can be overridden by environment variables or by constructing a new
``CompanionConfig`` instance at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class Paths:
    """Locations of the agent's CSV logs.

    ``metrics_csv`` records every speech attempt by the child, along
    with whether it needed correction and the corrected form.
    ``guidance_csv`` records behavioural events (e.g. anxiety,
    encouragement) and which strategy was used.  Both files are always
    appended to, never overwritten.
    """

    metrics_csv: Path = field(default_factory=lambda: Path(os.getenv("KQBC_METRICS_CSV", "kqbc_metrics.csv")))
    guidance_csv: Path = field(default_factory=lambda: Path(os.getenv("KQBC_GUIDANCE_CSV", "kqbc_guidance.csv")))


@dataclass
class CompanionConfig:
    """Top-level configuration for the KQBC agent.

    ``child_id`` is an internal identifier for the child.
    ``child_name`` is the name displayed in the UI.  ``caregiver_name``
    allows the dashboard and GUI to greet the caregiver personally.
    ``paths`` holds filesystem locations for metrics and guidance logs.
    """

    child_id: str = field(default="child_001")
    child_name: str = field(default="Jackson")
    caregiver_name: str = field(default="Molly")
    paths: Paths = field(default_factory=Paths)


# A default configuration instance.  Modules may import ``CONFIG`` to
# use sensible defaults.  Consumers requiring different settings can
# instantiate ``CompanionConfig`` directly and pass it into the agent.
CONFIG = CompanionConfig()