const baseProbability = 0.001 * state.coherence;
const hypothesisBoost = this.getHypothesisConfidenceBoost();
return Math.random() < (baseProbability * (1 + hypothesisBoost));
