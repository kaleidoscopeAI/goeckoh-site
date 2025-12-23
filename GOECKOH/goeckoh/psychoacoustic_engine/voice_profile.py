"""Voice fingerprint and speaker profile structures."""

from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class VoiceFingerprint:
    """
    Static Bubble Constraints Θ_u for one child.

    All values are deterministic statistics from enrollment.
    """

    mu_f0: float = 180.0
    sigma_f0: float = 25.0
    base_roughness: float = 0.2
    base_metalness: float = 0.5
    base_sharpness: float = 0.4
    rate: float = 3.8
    jitter_base: float = 0.08
    shimmer_base: float = 0.12
    base_radius: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpeakerProfile:
    """
    Full speaker description:
    - fingerprint: psychoacoustic bubble parameters
    - embedding: neural timbre embedding for the TTS backbone
    """

    user_id: str
    fingerprint: VoiceFingerprint
    embedding: np.ndarray  # 1D float32 array (e.g., 256–1024 dims)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "fingerprint": self.fingerprint.to_dict(),
            "embedding": self.embedding.tolist(),
        }
