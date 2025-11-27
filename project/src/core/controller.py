# ECHO_V4_UNIFIED/avatar/controller.py
from __future__ import annotations
import numpy as np
from events import AvatarFrame, now_ts, BrainMetrics, HeartMetrics
from .types import default_avatar_frame

class AvatarController:
    """
    Manages the state of the 18,000-node avatar, updating its visual
    properties based on the system's internal affective and cognitive state.
    """
    def __init__(self, n_nodes: int = 18000) -> None:
        self.n_nodes = n_nodes
        self._frame = default_avatar_frame(n_nodes)

    def update_from_state(
        self,
        heart: HeartMetrics,
        brain: BrainMetrics,
        caption: str,
    ) -> AvatarFrame:
        """
        Generates a new AvatarFrame based on the latest Heart and Brain metrics.
        """
        # Start with the previous frame's positions for smoother transitions
        positions = self._frame.positions.copy()
        colors = self._frame.colors.copy()
        sizes = self._frame.sizes.copy()

        # Map "temperature" and "energy" from the Heart onto radius and jitter
        energy = float(np.clip(heart.energy, 0.0, 2.0))

        # Radial expansion/contraction based on energy
        r = np.linalg.norm(positions, axis=1, keepdims=True) + 1e-6
        scaled_r = np.clip(r * (0.9 + 0.2 * energy), 0.2, 2.0)
        new_positions = positions / r * scaled_r
        
        # Color is driven by stress (red), harmony (green), and phi (blue)
        stress = float(np.clip(heart.stress, 0.0, 1.0))
        harmony = float(np.clip(heart.harmony, 0.0, 1.0))
        phi = float(np.clip(brain.phi, 0.0, 1.0))
        colors[:, 0] = 0.2 + 0.6 * stress  # Red channel
        colors[:, 1] = 0.2 + 0.6 * harmony # Green channel
        colors[:, 2] = 0.3 + 0.4 * phi     # Blue channel

        # Size of nodes is driven by confidence
        sizes[:] = 0.3 + 0.7 * float(np.clip(heart.confidence, 0.0, 1.0))
        
        meta = {
            "caption": caption,
            "stress": stress,
            "harmony": harmony,
            "energy": energy,
            "phi": phi,
        }
        
        self._frame = AvatarFrame(
            timestamp=now_ts(),
            positions=new_positions.astype(np.float32),
            colors=colors.astype(np.float32),
            sizes=sizes.astype(np.float32),
            meta=meta,
        )
        return self._frame

    @property
    def frame(self) -> AvatarFrame:
        """Returns the last computed avatar frame."""
        return self._frame
