const pa = this.quantumStates[a];
const pb = this.quantumStates[b];
return Math.cos(pa.phase - pb.phase) * pa.coherence * pb.coherence;
