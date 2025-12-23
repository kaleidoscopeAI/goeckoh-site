def hash_embedding(text: str, dim: int) -> np.ndarray:
"""Deterministic, stable embedding from text hash (no external models)."""
