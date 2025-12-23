from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Mode = Literal["outer", "inner", "coach"]


@dataclass(slots=True)
class AgentDecision:
    """
    Represents a decision made by the agent.
    """

    target_text: str
    mode: Mode = "inner"
    metadata: dict | None = None
