class Visualizer:
    """Simulates the 3D thought-form visualization from the React component."""
    def __init__(self, num_nodes=8000):
        self.num_nodes = num_nodes
        self.target_positions = np.zeros((num_nodes, 3))
        print(f"Visualizer initialized for {num_nodes} particles.")

    def create_visual_embedding(self, thought: str) -> np.ndarray:
        """Direct Python port of the createVisualEmbedding logic from JS."""
        thought_hash = sum(ord(c) for c in thought)
        
        # Create vectors based on the thought hash
        indices = np.arange(self.num_nodes)
        t = (indices / self.num_nodes) * np.pi * 8
        r = 300 + 100 * np.sin(thought_hash * 0.01 + indices * 0.02)
        twist = np.sin(thought_hash * 0.005) * 2.0
        
        x = r * np.cos(t + twist)
        y = r * np.sin(t + twist)
        z = 200 * np.sin(indices * 0.01 + thought_hash * 0.03)
        
        return np.stack([x, y, z], axis=1)

    def update_from_thought(self, thought: str):
        """Updates the visual state and reports summary statistics."""
        self.target_positions = self.create_visual_embedding(thought)
        
        mins = self.target_positions.min(axis=0)
        maxs = self.target_positions.max(axis=0)
        avg_radius = np.mean(np.linalg.norm(self.target_positions, axis=1))
        
        print("\n--- Visualizer State Update ---")
        print(f"Thought-form embedded from text: '{thought[:60]}...'")
        print(f"Bounding Box (X, Y, Z):")
        print(f"  min: [{mins[0]:.1f}, {mins[1]:.1f}, {mins[2]:.1f}]")
        print(f"  max: [{maxs[0]:.1f}, {maxs[1]:.1f}, {maxs[2]:.1f}]")
        print(f"Average Particle Radius from Center: {avg_radius:.2f}")
        print("---------------------------------")


class CognitiveEngine:
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

