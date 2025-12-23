let localCoherence = 0;
let neighborCount = 0;
const i3 = nodeIdx * 3;

// Sample nearby nodes for coherence calculation
for (let j = 0; j < Math.min(20, this.nodeCount); j += 5) {
  const otherIdx = (nodeIdx + j) % this.nodeCount;
  if (otherIdx === nodeIdx) continue;

  const j3 = otherIdx * 3;
  const dx = positions[i3] - positions[j3];
  const dy = positions[i3 + 1] - positions[j3 + 1];
  const dz = positions[i3 + 2] - positions[j3 + 2];
  const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

  if (dist < 80) {
    const otherState = this.quantumStates[otherIdx];
    const phaseDiff = Math.abs(this.quantumStates[nodeIdx].phase - otherState.phase);
    const entanglement = this.dynamicParameters.quantumEntanglementStrength;
    localCoherence += Math.cos(phaseDiff) * entanglement;
    neighborCount++;
  }
}

return neighborCount > 0 ? localCoherence / neighborCount : 0.5;
