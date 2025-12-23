if (!quantumEngine) return;
if (availableBuffers.length === 0) {
  // skip this frame if no buffer available
  return;
}
const bufferIndex = availableBuffers.shift()!;
const workingBuffer = buffers[bufferIndex];
// compute emotional field sample
const emotionalField = computeEmotionalField(positions, velocities, count);
// update engine
quantumEngine.update(positions, emotionalField);
const dynamicParams = quantumEngine.getDynamicParameters();
// Update half particles (partial updates)
const half = Math.floor(count / 2);
const startIndex = (step % 2) * half;
const dt = 0.016;
for (let i = startIndex; i < Math.min(startIndex + half, count); i++) {
  const i3 = i * 3;
  // compute mimicry force
  const force = computeForce(i, positions, velocities, emotionalField, dynamicParams);
  velocities[i3] += force[0] * dt;
  velocities[i3 + 1] += force[1] * dt;
  velocities[i3 + 2] += force[2] * dt;
  const damping = 0.98 - emotionalField.arousal * 0.1;
  velocities[i3] *= damping;
  velocities[i3 + 1] *= damping;
  velocities[i3 + 2] *= damping;
  workingBuffer[i3] = positions[i3] + velocities[i3] * dt;
  workingBuffer[i3 + 1] = positions[i3 + 1] + velocities[i3 + 1] * dt;
  workingBuffer[i3 + 2] = positions[i3 + 2] + velocities[i3 + 2] * dt;
  const r = Math.hypot(workingBuffer[i3], workingBuffer[i3 + 1], workingBuffer[i3 + 2]);
  if (r > 400) {
    const scale = 0.95;
    workingBuffer[i3] *= scale;
    workingBuffer[i3 + 1] *= scale;
    workingBuffer[i3 + 2] *= scale;
    velocities[i3] *= scale;
    velocities[i3 + 1] *= scale;
    velocities[i3 + 2] *= scale;
  }
}
positions = workingBuffer;
// prepare system state
const state = {
  globalCoherence: quantumEngine.getGlobalCoherence(),
  emotionalField,
  knowledgeCrystals: quantumEngine.getKnowledgeCrystals(),
  hypotheses: quantumEngine.generateHypotheses(3),
  dynamicParameters: quantumEngine.getDynamicParameters()
};
self.postMessage({ cmd: 'positions', positions: workingBuffer.buffer, systemState: state }, [workingBuffer.buffer]);
step++;
