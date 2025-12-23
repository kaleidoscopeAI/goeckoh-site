"""Bubble shaping and material mapping for the Psychoacoustic Engine."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .attempt_analysis import AttemptFeatures
from .voice_profile import VoiceProfile


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _smoothstep(x: float, edge0: float, edge1: float) -> float:
    x_norm = (x - edge0) / (edge1 - edge0 + 1e-8)
    x_norm = np.clip(x_norm, 0.0, 1.0)
    return float(x_norm * x_norm * (3.0 - 2.0 * x_norm))


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(x, lo, hi))


def _color_from_pitch(mu_f0: float) -> np.ndarray:
    """F0 → Hue → RGB. Deterministic color per fingerprint."""
    f0 = np.clip(mu_f0, 80.0, 400.0)
    hue = (f0 - 80.0) / (400.0 - 80.0)  # 0..1
    h = hue * 6.0
    c = 1.0
    x = c * (1.0 - abs(h % 2.0 - 1.0))
    if 0.0 <= h < 1.0:
        r, g, b = c, x, 0.0
    elif 1.0 <= h < 2.0:
        r, g, b = x, c, 0.0
    elif 2.0 <= h < 3.0:
        r, g, b = 0.0, c, x
    elif 3.0 <= h < 4.0:
        r, g, b = 0.0, x, c
    elif 4.0 <= h < 5.0:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    return np.array([r, g, b], dtype=np.float32)


@dataclass
class BubbleState:
    radii: np.ndarray  # [N_vertices]
    colors: np.ndarray  # [N_vertices, 3]
    pbr_props: Dict[str, float]  # {"rough", "metal", "spike"}


def compute_bubble_state(
    vertices: np.ndarray,
    profile: VoiceProfile,
    attempt_feat: AttemptFeatures,
    t_idx: int,
    layout: Optional[Dict[str, Any]] = None,
    *,
    base_radius: Optional[float] = None,
) -> BubbleState:
    """
    Psychoacoustic Engine upgrade over the old sound physics.

    - ZCR → Bouba/Kiki spikes (pbr_props["spike"])
    - Idle heartbeat at user speaking rate
    - Spectral tilt / HNR → PBR material (metalness, roughness)
    - Deterministic; no RNGs
    """
    fp = profile.fingerprint
    N = vertices.shape[0]

    dt = attempt_feat.dt
    t = t_idx * dt

    # Idle heartbeat frequency: ω_idle = 2π * Rate_u (syllables/sec)
    idle_freq = 2.0 * np.pi * float(fp.rate)

    # Loudness for this frame
    energy = float(attempt_feat.energy_attempt[t_idx])
    volume_norm = energy
    # Steep sigmoid to decisively choose active vs idle
    G_active = float(_sigmoid((volume_norm - 0.05) * 10.0))

    R0 = base_radius if base_radius is not None else getattr(fp, "base_radius", 1.0)

    # Voice-driven radius (active)
    u_energy = float(np.clip(volume_norm, 0.0, 1.5))

    # Modal term: prefer per-vertex field if provided
    if layout is not None and "voice_field" in layout:
        u_modes_field = np.asarray(layout["voice_field"], dtype=np.float32)
        if u_modes_field.shape[0] != N:
            raise ValueError("voice_field must be shaped [N_vertices].")
    else:
        f0 = float(attempt_feat.f0_attempt[t_idx])
        f0_norm = np.clip((f0 - 80.0) / (400.0 - 80.0), 0.0, 1.0)
        u_modes_field = np.full(
            (N,),
            0.08 * np.sin(2.0 * np.pi * f0_norm * t),
            dtype=np.float32,
        )

    # Bouba/Kiki spike coefficient χ(t) from ZCR
    zcr = float(attempt_feat.zcr_attempt[t_idx])
    chi = _smoothstep(zcr, 0.10, 0.40)  # 0 = Bouba, 1 = Kiki
    gamma_spike = 0.12

    active_scalar = R0 * (1.0 + u_energy + chi * gamma_spike)
    active_r = active_scalar + (R0 * 0.1 * u_modes_field)

    # Idle heartbeat
    A_idle = 0.05
    idle_r = R0 * (0.85 + A_idle * np.sin(idle_freq * t))

    # Blend active vs idle
    final_r = G_active * active_r + (1.0 - G_active) * idle_r

    # PBR mapping
    tilt = float(attempt_feat.spectral_tilt[t_idx])
    hnr = float(attempt_feat.hnr_attempt[t_idx])
    roughness = _clamp(1.0 - hnr, 0.0, 1.0)
    metalness = _clamp(tilt * 1.5, 0.0, 1.0)
    spike_amt = _clamp(chi, 0.0, 1.0)

    base_color = _color_from_pitch(fp.mu_f0)
    colors = np.tile(base_color[None, :], (N, 1))

    return BubbleState(
        radii=final_r.astype(np.float32),
        colors=colors,
        pbr_props={
            "rough": roughness,
            "metal": metalness,
            "spike": spike_amt,
        },
    )
