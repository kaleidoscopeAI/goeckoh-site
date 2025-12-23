const stateA = this.quantumStates[particleA];
const stateB = this.quantumStates[particleB];
const phaseCorrelation = Math.cos(stateA.phase - stateB.phase);
return phaseCorrelation * stateA.coherence * stateB.coherence;
