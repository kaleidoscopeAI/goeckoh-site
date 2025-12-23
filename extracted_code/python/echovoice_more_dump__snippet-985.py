"""Orchestrates the main cognitive loop, including LLM reflection."""

def summarize_state(self, state: HybridState, hamiltonian: SemanticHamiltonian) -> str:
    """Creates a text summary of the current system state for the LLM."""
    current_energy = hamiltonian.energy(state)

    # Calculate bit coherence (1 is perfectly aligned, 0 is random)
    all_bits = np.stack([state.E[n] for n in sorted(state.E.keys())])
    avg_bit_pattern = np.mean(all_bits, axis=0)
    coherence = 1.0 - np.mean(4 * avg_bit_pattern * (1 - avg_bit_pattern))

    # Calculate vector cohesion
    all_vectors = np.stack([state.x[n] for n in sorted(state.x.keys())])
    center_of_mass = np.mean(all_vectors, axis=0)
    avg_dist = np.mean(np.linalg.norm(all_vectors - center_of_mass, axis=1))

    return (f"System state report: Current energy is {current_energy:.4f}. "
            f"Discrete state (bit) coherence is {coherence:.3f}. "
            f"Continuous state (vector) average distance from center is {avg_dist:.3f}. "
            f"Reflect on this state and describe the emergent cognitive structure.")

def reflect_with_ollama(self, prompt: str) -> str:
    """Simulated/mocked call to an Ollama LLM."""
    print(f"\n[Cognitive Engine] Sending prompt to Ollama: '{prompt[:100]}...'")
    # In a real system, this would be a network request.
    # Here, we generate a plausible response based on the prompt's data.

    # Simple logic to generate varied, deterministic responses for the demo
    energy_val = float(prompt.split('is ')[1].split('.')[0])
    coherence_val = float(prompt.split('coherence is ')[1].split('.')[0])

    if energy_val < 500:
        if coherence_val > 0.8:
            return "The system is reaching a highly stable, crystalline state of understanding. A core concept has been solidified."
        else:
            return "Energy is low but the structure is not yet fully coherent. It is settling into a relaxed but unfocused state."
    else:
        if coherence_val < 0.2:
            return "High energy and low coherence indicate a state of cognitive dissonance or confusion. The system is exploring disparate concepts."
        else:
            return "A state of high energy and high coherence suggests intense, focused computation. A complex pattern is being actively processed."

