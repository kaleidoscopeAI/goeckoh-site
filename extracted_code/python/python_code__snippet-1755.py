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


