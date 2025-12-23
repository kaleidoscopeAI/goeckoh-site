# ECHO_V4_UNIFIED/avatar/types.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Dict, Any
from events import AvatarFrame, now_ts

def default_avatar_frame(n_nodes: int = 18000) -> AvatarFrame:
    """
    Creates a default, initial state for the avatar.
    """
    rng = np.random.default_rng(42)
    positions = rng.uniform(-1.0, 1.0, size=(n_nodes, 3)).astype(np.float32)
    colors = np.ones((n_nodes, 3), dtype=np.float32) * 0.5
    sizes = np.ones((n_nodes,), dtype=np.float32) * 0.5
    return AvatarFrame(
        timestamp=now_ts(),
        positions=positions,
        colors=colors,
        sizes=sizes,
        meta={"state": "idle"},
    )
