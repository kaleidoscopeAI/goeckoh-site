1.  **Bit-Level Crystallization (from crystal_ai_os_full_system.py):**
    - The system's core is a `HybridState` containing both continuous (vector) and
      discrete (bit-string) data for a network of nodes.
    - A `SemanticHamiltonian` defines the system's total "energy," quantifying
      its cognitive stress and incoherence.
    - A `MetropolisEngine` uses simulated annealing to optimize the discrete bit
      states, allowing the system to find globally optimal configurations.
    - A `GradientFlow` engine optimizes the continuous vector states via
      gradient descent.

2.  **Ollama LLM Integration (from cognitive_crystal_system.py):**
    - A `CognitiveEngine` orchestrates the system.
    - Periodically, it reflects on its own state, creating a summary prompt.
    - It calls a simulated `reflect_with_ollama` function to generate a
      high-level textual "thought" about its current condition.

3.  **Visual Crystallization (from UnifiedCognitiveSystem.js):**
    - A `Visualizer` class directly implements the 3D embedding logic from the
      React frontend.
    - It takes the AI's textual "thought" and transforms it into a set of 3D
      coordinates for a particle cloud, simulating the emergent "thought-form."
    - Instead of rendering, it reports descriptive statistics of the visual state.

