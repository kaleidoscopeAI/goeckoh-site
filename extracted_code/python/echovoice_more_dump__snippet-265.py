def thought_to_vector(thought: str, dim: int) -> np.ndarray:
    """Converts a thought string into a 'nudge' vector."""
    # This is a placeholder; a real implementation would use a proper text embedding model.
    vec = np.zeros(dim)
    for i, char in enumerate(thought):
        vec[i % dim] += ord(char)
    
    # Normalize and scale the vector to create a gentle nudge
    norm = np.linalg.norm(vec)
    if norm > 0:
        return (vec / norm) * 0.1 # Nudge scale factor of 0.1
    return vec

