   * The file crystal_ai_os/dynamics/metropolis.py implements the Metropolis-Hastings algorithm. This is a core
     component that evolves a system of bit-vectors (E) state-by-state to find configurations that minimize a
     "semantic energy" function defined by the Hamiltonian.
   * The main script crystal_ai_os/cli/run.py uses this metropolis_anneal function to run the evolution over
     thousands of iterations, making it a complete system for evolving a network of interacting elements.

