- A `CognitiveEngine` orchestrates the system.
- Periodically, it reflects on its own state, creating a summary prompt.
- It calls a simulated `reflect_with_ollama` function to generate a
  high-level textual "thought" about its current condition.

