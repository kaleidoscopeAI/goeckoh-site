"""
Deterministic modal superposition for bubble vertex displacement.

Creates per-vertex "voice fields" using harmonic modes with
deterministic phases based on vertex positions.

goeckoh/psychoacoustic_engine/voice_field.py
"""

import numpy as np


def procedural_phase(vertex: np.ndarray, mode_idx: int) -> float:
    """
    Generate deterministic phase from vertex position and mode index.
    
    Uses a hash-like function to create reproducible but spatially-varying
    phases for each vertex and harmonic mode.
    
    Args:
        vertex: 3D position [x, y, z]
        mode_idx: Harmonic mode number (1, 2, 3, ...)
    
    Returns:
        Phase in radians [0, 2π)
    """
    x, y, z = vertex
    
    # Deterministic hash-like function
    seed = (
        np.sin(x * 12.9898 + mode_idx * 78.233) * 43758.5453 +
        np.sin(y * 93.9898 + mode_idx * 67.345) * 27183.1234 +
        np.sin(z * 45.1234 + mode_idx * 34.567) * 31415.9265
    )
    
    phase = (seed % 1.0) * 2.0 * np.pi
    return float(phase)


def generate_voice_field(
    vertices: np.ndarray,
    f0: float,
    t: float,
    num_modes: int = 3,
    decay_rate: float = 0.7
) -> np.ndarray:
    """
    Generate per-vertex modal displacement field.
    
    Creates a deterministic "voice field" where each vertex oscillates
    according to a superposition of harmonic modes. The phase of each
    mode at each vertex is determined by the vertex's position, ensuring
    spatial coherence and reproducibility.
    
    Args:
        vertices: Vertex positions [N, 3]
        f0: Fundamental frequency (Hz)
        t: Current time (seconds)
        num_modes: Number of harmonic overtones to include
        decay_rate: Amplitude decay for each successive mode
    
    Returns:
        Per-vertex displacement field [N]
    """
    N = vertices.shape[0]
    field = np.zeros(N, dtype=np.float32)
    
    # Normalize F0 to prevent extreme values
    f0_norm = np.clip(f0, 60.0, 500.0)
    omega0 = 2.0 * np.pi * f0_norm
    
    for n in range(N):
        vertex = vertices[n]
        displacement = 0.0
        
        for k in range(1, num_modes + 1):
            # Harmonic frequency
            omega_k = k * omega0
            
            # Deterministic phase based on vertex position and mode
            phi_k = procedural_phase(vertex, k)
            
            # Amplitude decays with mode number
            amplitude = (decay_rate ** (k - 1)) / k
            
            # Modal contribution
            displacement += amplitude * np.sin(omega_k * t + phi_k)
        
        field[n] = displacement
    
    return field


def generate_idle_field(
    vertices: np.ndarray,
    rate: float,
    t: float,
    amplitude: float = 0.05
) -> np.ndarray:
    """
    Generate idle heartbeat field for silent periods.
    
    When the bubble is not actively speaking, it breathes at the
    child's natural speaking rate.
    
    Args:
        vertices: Vertex positions [N, 3]
        rate: Speaking rate (syllables per second)
        t: Current time (seconds)
        amplitude: Breathing amplitude
    
    Returns:
        Per-vertex idle displacement [N]
    """
    N = vertices.shape[0]
    
    # Idle frequency from speaking rate
    omega_idle = 2.0 * np.pi * rate
    
    # Global heartbeat with slight spatial variation
    field = np.zeros(N, dtype=np.float32)
    for n in range(N):
        vertex = vertices[n]
        phase = procedural_phase(vertex, 0) * 0.1  # Small spatial variation
        field[n] = amplitude * np.sin(omega_idle * t + phase)
    
    return field


def blend_voice_and_idle(
    voice_field: np.ndarray,
    idle_field: np.ndarray,
    activity_level: float
) -> np.ndarray:
    """
    Blend between voice-driven and idle heartbeat fields.
    
    Args:
        voice_field: Active voice displacement [N]
        idle_field: Idle heartbeat displacement [N]
        activity_level: 0 (idle) to 1 (active speaking)
    
    Returns:
        Blended displacement field [N]
    """
    return activity_level * voice_field + (1.0 - activity_level) * idle_field
