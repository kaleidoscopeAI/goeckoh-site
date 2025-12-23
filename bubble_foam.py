"""
Bubble shaping and material mapping for the Psychoacoustic Engine.

Maps acoustic features to visual bubble properties:
- Radius (energy + modes + Bouba/Kiki spikes)
- Color (pitch-derived hue)
- PBR materials (roughness from HNR, metalness from tilt)

Provides both scalar (single bubble) and per-vertex APIs.

goeckoh/psychoacoustic_engine/bubble_foam.py
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .attempt_analysis import AttemptFeatures
from .voice_profile import VoiceProfile


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Smooth activation function."""
    return 1.0 / (1.0 + np.exp(-x))


def _smoothstep(x: float, edge0: float, edge1: float) -> float:
    """Smooth interpolation between two edges."""
    x_norm = (x - edge0) / (edge1 - edge0 + 1e-8)
    x_norm = np.clip(x_norm, 0.0, 1.0)
    return float(x_norm * x_norm * (3.0 - 2.0 * x_norm))


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value to range."""
    return float(np.clip(x, lo, hi))


def _color_from_pitch(mu_f0: float) -> np.ndarray:
    """
    F0 → Hue → RGB. Deterministic color per fingerprint.
    
    Maps fundamental frequency to a color on the spectrum.
    Lower pitches → warmer colors (red/orange)
    Higher pitches → cooler colors (blue/purple)
    """
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
    """
    Complete visual state of the bubble at one time frame.
    """
    radii: np.ndarray          # [N_vertices] - per-vertex displacement
    colors: np.ndarray         # [N_vertices, 3] - RGB colors
    pbr_props: Dict[str, float]  # {"rough", "metal", "spike"}


def compute_bubble_state_vertices(
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
    
    Transforms acoustic features into visual bubble properties:
    - ZCR → Bouba/Kiki spikes (pbr_props["spike"])
    - Idle heartbeat at user speaking rate
    - Spectral tilt / HNR → PBR material (metalness, roughness)
    - Deterministic; no RNGs
    
    Args:
        vertices: Vertex positions [N, 3]
        profile: Child's VoiceProfile (Θᵤ + embedding)
        attempt_feat: Acoustic features for this utterance
        t_idx: Current frame index
        layout: Optional pre-computed fields (e.g., "voice_field")
        base_radius: Override base bubble size
    
    Returns:
        BubbleState with radii, colors, and PBR properties
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
        # Fallback: simple sinusoidal based on F0
        f0 = float(attempt_feat.f0_attempt[t_idx])
        f0_norm = np.clip((f0 - 80.0) / (400.0 - 80.0), 0.0, 1.0)
        u_modes_field = np.full(
            (N,),
            0.08 * np.sin(2.0 * np.pi * f0_norm * t),
            dtype=np.float32,
        )
    
    # Bouba/Kiki spike coefficient χ(t) from ZCR
    zcr = float(attempt_feat.zcr_attempt[t_idx])
    chi = _smoothstep(zcr, 0.10, 0.40)  # 0 = Bouba (smooth), 1 = Kiki (spiky)
    gamma_spike = 0.12
    
    # Active radius: base + energy + modes + spikiness
    active_scalar = R0 * (1.0 + u_energy + chi * gamma_spike)
    active_r = active_scalar + (R0 * 0.1 * u_modes_field)
    
    # Idle heartbeat
    A_idle = 0.05
    idle_r = R0 * (0.85 + A_idle * np.sin(idle_freq * t))
    
    # Blend active vs idle based on activity gate
    final_r = G_active * active_r + (1.0 - G_active) * idle_r
    
    # PBR mapping from acoustic features
    tilt = float(attempt_feat.spectral_tilt[t_idx])
    hnr = float(attempt_feat.hnr_attempt[t_idx])
    
    roughness = _clamp(1.0 - hnr, 0.0, 1.0)  # High HNR = smooth surface
    metalness = _clamp(tilt * 1.5, 0.0, 1.0)  # Bright sound = metallic look
    spike_amt = _clamp(chi, 0.0, 1.0)         # High ZCR = spiky (Kiki)
    
    # Color from pitch (deterministic per child)
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


def compute_bubble_state(
    profile: VoiceProfile,
    attempt_feat: AttemptFeatures,
    t_time: float,
    t_idx: Optional[int] = None,
) -> Dict[str, float]:
    """
    Simplified scalar bubble state for single-bubble visualization.
    
    This version returns a dict with scalar values for simple use cases
    (testing, 2D visualization, single sphere).
    
    Args:
        profile: Child's VoiceProfile (Θᵤ + optional embedding)
        attempt_feat: Acoustic features for this utterance
        t_time: Current time in seconds
        t_idx: Frame index (if None, computed from t_time)
    
    Returns:
        Dict with keys: {radius, color_r, color_g, color_b, rough, metal, spike}
    """
    fp = profile.fingerprint
    
    # Determine frame index
    if t_idx is None:
        t_idx = int(t_time / attempt_feat.dt)
    t_idx = min(t_idx, len(attempt_feat.energy_attempt) - 1)
    
    dt = attempt_feat.dt
    t = t_idx * dt
    
    # Idle heartbeat frequency
    idle_freq = 2.0 * np.pi * float(fp.rate)
    
    # Current frame energy
    energy = float(attempt_feat.energy_attempt[t_idx])
    volume_norm = energy
    
    # Activity gate (steep sigmoid)
    G_active = float(_sigmoid((volume_norm - 0.05) * 10.0))
    
    R0 = getattr(fp, "base_radius", 1.0)
    
    # Active radius calculation
    u_energy = float(np.clip(volume_norm, 0.0, 1.5))
    
    # Simple modal term for scalar case
    f0 = float(attempt_feat.f0_attempt[t_idx])
    f0_norm = np.clip((f0 - 80.0) / (400.0 - 80.0), 0.0, 1.0)
    u_mode = 0.08 * np.sin(2.0 * np.pi * f0_norm * t)
    
    # Bouba/Kiki coefficient
    zcr = float(attempt_feat.zcr_attempt[t_idx])
    chi = _smoothstep(zcr, 0.10, 0.40)
    gamma_spike = 0.12
    
    # Active radius: energy expansion (factor of 2.0)
    active_scalar = R0 * (1.0 + u_energy * 2.0 + chi * gamma_spike)
    active_r = active_scalar + (R0 * 0.1 * u_mode)
    
    # Idle breathing
    A_idle = 0.1
    idle_r = R0 * (0.9 + A_idle * np.sin(idle_freq * t))
    
    # Blend
    final_r = G_active * active_r + (1.0 - G_active) * idle_r
    
    # PBR properties
    tilt = float(attempt_feat.spectral_tilt[t_idx])
    hnr = float(attempt_feat.hnr_attempt[t_idx])
    
    roughness = _clamp(1.0 - hnr, 0.0, 1.0)
    metalness = _clamp(tilt * 1.5, 0.0, 1.0)
    spike_amt = _clamp(chi, 0.0, 1.0)
    
    # Color from pitch
    base_color = _color_from_pitch(fp.mu_f0)
    
    return {
        "radius": float(final_r),
        "color_r": float(base_color[0]),
        "color_g": float(base_color[1]),
        "color_b": float(base_color[2]),
        "rough": roughness,
        "metal": metalness,
        "spike": spike_amt,
    }
