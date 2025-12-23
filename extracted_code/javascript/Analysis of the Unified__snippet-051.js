const i3 = particleIdx * 3;
const type = particleIdx % 2; // 0 = red, 1 = blue

let totalForce: [number, number, number] = [0, 0, 0];
let neighborCount = 0;

const neighbors = spatialGrid.getNeighbors(
  positions[i3], positions[i3+1], positions[i3+2], 60
);

for (const j of neighbors) {
  if (j === particleIdx) continue;

  const j3 = j * 3;
  const dx = positions[j3] - positions[i3];
  const dy = positions[j3+1] - positions[i3+1];
  const dz = positions[j3+2] - positions[i3+2];
  const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);

  if (dist > 0 && dist < 60) {
    const jType = j % 2;

    // Red-Blue mimicry dynamics
    let forceStrength = (type === jType) ? 0.15 : -0.08;

    // Emotional modulation
    forceStrength *= (1 + emotionalContext.valence * 0.5);
    forceStrength *= (1 + emotionalContext.arousal * 0.3);

    // Embedded Ollama parameter modulation
    forceStrength *= dynamicParams.mimicryForceModifier;

    // Quantum effects
    const quantumEffect = quantumEngine.getQuantumInfluence(particleIdx, j);
    forceStrength *= (1 + quantumEffect * 0.2 * dynamicParams.quantumEntanglementStrength);

    const force = forceStrength / (dist + 0.1);
    totalForce[0] += dx * force;
    totalForce[1] += dy * force;
    totalForce[2] += dz * force;
    neighborCount++;
  }
}

if (neighborCount > 0) {
  totalForce[0] /= neighborCount;
  totalForce[1] /= neighborCount;
  totalForce[2] /= neighborCount;
}

return totalForce;
