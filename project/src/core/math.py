"""Core math helpers for CrystalBrain metrics (lightweight placeholders)."""

from __future__ import annotations

import numpy as np


def information_energy(bits: np.ndarray) -> float:
    """Placeholder: returns mean bit energy."""
    if bits.size == 0:
        return 0.0
    return float(np.mean(bits))


def field_stability(embs: np.ndarray) -> float:
    """Placeholder stability metric."""
    if embs.size == 0:
        return 0.0
    return float(1.0 / (1.0 + np.std(embs)))


def lyapunov_loss(H_bits: float, S_field: float) -> float:
    """Placeholder Lyapunov-like loss."""
    return float(abs(H_bits - S_field))


def coherence_metric(embs: np.ndarray) -> float:
    if embs.size == 0:
        return 0.0
    return float(1.0 / (1.0 + np.mean(np.var(embs, axis=0))))


def integrated_information(embs: np.ndarray) -> float:
    if embs.size == 0:
        return 0.0
    return float(np.clip(np.mean(np.abs(embs)), 0.0, 1.0))
