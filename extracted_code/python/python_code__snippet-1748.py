import numpy as np
from scipy.special import sph_harm


def procedural_phase(t: float, k: int, user_seed: int) -> float:
    """
    Deterministic pseudo-random phase Φ_proc(t, k).

    Same user_seed + same timeline + same mode index ⇒ identical sequence.
    """
    t_int = int(t * 1000.0)
    x = (k * 73856093) ^ (t_int * 19349663) ^ (user_seed * 83492791)
    x &= 0xFFFFFFFF
    return (x / 0xFFFFFFFF) * 2.0 * np.pi


def generate_sh_modes(vertices: np.ndarray, k_count: int) -> np.ndarray:
    """Generate deterministic spherical harmonics basis [K, N]."""
    vertices = np.asarray(vertices, dtype=np.float32)
    N = vertices.shape[0]
    modes = np.zeros((k_count, N), dtype=np.float32)
    theta = np.arccos(np.clip(vertices[:, 2], -1.0, 1.0))  # polar angle
    phi = np.arctan2(vertices[:, 1], vertices[:, 0])  # azimuth

    idx = 0
    l = 0
    while idx < k_count:
        for m in range(-l, l + 1):
            if idx >= k_count:
                break
            modes[idx] = np.real(sph_harm(m, l, phi, theta)).astype(np.float32)
            idx += 1
        l += 1

    # Normalize per-mode to keep amplitudes bounded
    max_per_mode = np.max(np.abs(modes), axis=1, keepdims=True) + 1e-6
    modes = modes / max_per_mode
    return modes


def generate_voice_field(
    vertices: np.ndarray,
    t: float,
    user_seed: int,
    K: int = 10,
