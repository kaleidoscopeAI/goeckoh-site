if (availableBuffers.length === 0) return;

const bufferIndex = availableBuffers.shift()!;
const workingBuffer = buffers[bufferIndex];

// Update spatial grid
spatialGrid.update(positions);

// Calculate emotional field
const emotionalField = calculateEmotionalField(positions, velocities);

// Update quantum engine with embedded Ollama
await quantumEngine.update(positions, emotionalField);

const dynamicParams = quantumEngine.getDynamicParameters();

// Partial updates for performance (50% per frame)
const halfCount = Math.floor(count / 2);
const startIndex = (step % 2) * halfCount;
const dt = 0.016;

for (let i = startIndex; i < startIndex + halfCount; i++) {
  if (i >= count) break;

  const i3 = i * 3;

  // Compute enhanced mimicry forces
  const mimicryForce = EmbeddedCognitiveDynamics.computeMimicryForce(
    i, positions, velocities, emotionalField, dynamicParams
  );

  // Update velocity
  velocities[i3] += mimicryForce[0] * dt;
  velocities[i3+1] += mimicryForce[1] * dt;
  velocities[i3+2] += mimicryForce[2] * dt;

  // Adaptive damping
  const damping = 0.98 - emotionalField.arousal * 0.1;
  velocities[i3] *= damping;
  velocities[i3+1] *= damping;
  velocities[i3+2] *= damping;

  // Update position
  workingBuffer[i3] = positions[i3] + velocities[i3] * dt;
  workingBuffer[i3+1] = positions[i3+1] + velocities[i3+1] * dt;
  workingBuffer[i3+2] = positions[i3+2] + velocities[i3+2] * dt;

  // Soft boundary conditions
  const r = Math.sqrt(
    workingBuffer[i3]**2 + workingBuffer[i3+1]**2 + workingBuffer[i3+2]**2
  );
  if (r > 400) {
    const scale = 0.95;
    workingBuffer[i3] *= scale;
    workingBuffer[i3+1] *= scale;
    workingBuffer[i3+2] *= scale;
    velocities[i3] *= scale;
    velocities[i3+1] *= scale;
    velocities[i3+2] *= scale;
  }
}

positions = workingBuffer;

// Prepare system state
const systemState = {
  globalCoherence: quantumEngine.getGlobalCoherence(),
  emotionalField,
  knowledgeCrystals: quantumEngine.getKnowledgeCrystals(),
  hypotheses: quantumEngine.generateHypotheses(3),
  dynamicParameters: dynamicParams,
  ollamaStatus: quantumEngine.getOllamaStatus(),
  particleState: {
    emotionalField,
    quantumCoherence: quantumEngine.getGlobalCoherence(),
    embeddedOllama: true
  }
};

self.postMessage({
  cmd: 'positions',
  positions: workingBuffer.buffer,
  systemState
}, [workingBuffer.buffer]);

step++;

