         "semantic energy" function defined by the Hamiltonian.
       * The main script crystal_ai_os/cli/run.py uses this metropolis_anneal function to run the evolution over
         thousands of iterations, making it a complete system for evolving a network of interacting elements.

   2. `umaa` - Kaleidoscope Engine with Replica Exchange:
       * The umaa/kaleidoscope_engine.py and umaa/main.py files define a system of Unified Interface Nodes (UINs)
         that evolve based on an annealing process.
       * The system uses Parallel Tempering (Replica Exchange), where multiple copies ("replicas") of the node
         network evolve at different temperatures. The attempt_swap function periodically swaps states between these
         replicas. This is a sophisticated technique for evolving a system towards a global energy minimum.
       * The main.py file contains the main evolution loop, which repeatedly calls rep.step() for each replica,
         showing a clear time-based evolution of the entire system.

  Both projects contain core loops that iteratively update the state of a system according to a set of rules and
  energy functions, which aligns with your definition of letting a system "actually evolve."

  To implement the new UMAA v3.0 system without external dependencies as you require, I will now proceed by
  building the necessary components—such as vector math and graph data structures—from scratch. I will begin by
  implementing the core data structures and mathematical functions.

