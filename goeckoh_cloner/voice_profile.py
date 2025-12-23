from dataclasses import dataclass, asdict
from pathlib import Path
import json

from config import PROFILES_DIR


@dataclass(frozen=True)
class VoiceFingerprint:
    """Static Bubble Constraints (Theta_u) for one child."""
    mu_f0: float = 180.0        # Median pitch (Hz)
    sigma_f0: float = 25.0      # Pitch variability
    base_roughness: float = 0.2 # From HNR
    base_metalness: float = 0.5 # From spectral tilt
    base_sharpness: float = 0.4 # From ZCR
    rate: float = 3.0           # Syllables/sec
    jitter_base: float = 0.1    # Micro-tremor amplitude
    shimmer_base: float = 0.1   # Amplitude variation
    base_radius: float = 1.0    # Default bubble size

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def from_json(path: Path) -> "VoiceFingerprint":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return VoiceFingerprint(**data)


def get_profile_path(profile_name: str) -> Path:
    return PROFILES_DIR / f"{profile_name}_fingerprint.json"
