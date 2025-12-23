const sampleSize = Math.min(1000, this.nodeCount);
let totalCoherence = 0;

for (let i = 0; i < sampleSize; i++) {
  const idx = Math.floor((i * this.nodeCount) / sampleSize);
  const state = this.quantumStates[idx];

  // Enhanced emotional modulation
  const emotionalBoost = this.dynamicParameters.emotionalValenceBoost;
  state.phase += (emotionalField.valence * 0.1 + emotionalField.arousal * 0.05) * 
                (1 + emotionalBoost);

  // Update coherence with dynamic decay
  state.coherence *= this.dynamicParameters.coherenceDecayRate;

  // Calculate local coherence from spatial relationships
  const localCoherence = this.calculateLocalCoherence(idx, positions);
  state.coherence = 0.9 * state.coherence + 0.1 * localCoherence;

  totalCoherence += state.coherence;

  // Knowledge crystallization with embedded Ollama
  if (this.shouldCrystallizeKnowledge(idx, state)) {
    await this.crystallizeKnowledge(idx, emotionalField);
  }
}

this.globalCoherence = totalCoherence / sampleSize;
