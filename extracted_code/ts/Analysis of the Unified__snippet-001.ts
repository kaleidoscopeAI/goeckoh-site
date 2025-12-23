// src/workers/particleWorker.ts
import type { EnhancedQuantumConsciousnessEngineType } from '../engines/types';
import EnhancedQuantumConsciousnessEngine from '../engines/EnhancedQuantumConsciousnessEngine';

const BUFFER_COUNT = 3;
let buffers: Float32Array[] = [];
let availableBuffers: number[] = [];

let positions: Float32Array;
let velocities: Float32Array;
let count = 0;
let step = 0;
let quantumEngine: EnhancedQuantumConsciousnessEngineType | null = null;

let globalRequestId = 0;
const outstandingRequests = new Map<number, any>();

self.onmessage = (e: MessageEvent) => {
  const { cmd, data } = e.data;
  if (cmd === 'init') {
    count = data.count;
    buffers = [];
    availableBuffers = [];
    const initialPositions = new Float32Array(data.positions);
    for (let i = 0; i < BUFFER_COUNT; i++) {
      const b = new Float32Array(count * 3);
      b.set(initialPositions);
      buffers.push(b);
      availableBuffers.push(i);
    }
    positions = buffers[0];
    velocities = new Float32Array(count * 3);
    for (let i = 0; i < velocities.length; i++) velocities[i] = (Math.random() - 0.5) * 2;
    quantumEngine = new EnhancedQuantumConsciousnessEngine(count);
    self.postMessage({ cmd: 'ready' });
  } else if (cmd === 'update') {
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
  } else if (cmd === 'returnBuffer') {
    // main thread returned buffer; push index back based on buffer reference
    // buffer arrives as ArrayBuffer — find matching index by comparing byteLength
    const buff = data.buffer as ArrayBuffer;
    const matchIndex = buffers.findIndex(b => b.buffer.byteLength === buff.byteLength && b.buffer !== buff);
    // best-effort: if no exact match, push the first free index
    if (matchIndex !== -1) availableBuffers.push(matchIndex);
    else availableBuffers.push(0);
  } else if (cmd === 'ollama_feedback') {
    // main thread feedback forwarded to worker
    const { requestId, analysis } = data;
    const req = outstandingRequests.get(requestId);
    if (req && quantumEngine) {
      quantumEngine.integrateOllamaFeedback(analysis);
      outstandingRequests.delete(requestId);
    }
  } else if (cmd === 'hypothesis_response') {
    // helper if you choose to return analysis in another shape
  }
};

// helper implementations (simplified)
function computeEmotionalField(positions: Float32Array, velocities: Float32Array, count: number) {
  let totalVal = 0;
  let totalArousal = 0;
  const sampleSize = Math.min(1000, count);
  for (let i = 0; i < sampleSize; i++) {
    const idx = Math.floor((i * count) / sampleSize) * 3;
    const vy = velocities[idx];
    const speed = Math.hypot(velocities[idx], velocities[idx + 1], velocities[idx + 2]);
    totalVal += (vy > 0 ? 1 : -1) * speed;
    totalArousal += speed;
  }
  return { valence: Math.tanh(totalVal / sampleSize), arousal: Math.tanh(totalArousal / sampleSize / 10) };
}

function computeForce(i: number, positions: Float32Array, velocities: Float32Array, emotionalContext: any, dynamicParams: any) {
  // VERY simple placeholder force using distance from origin and emotional modulation
  const i3 = i * 3;
  const dx = -positions[i3];
  const dy = -positions[i3 + 1];
  const dz = -positions[i3 + 2];
  const dist = Math.hypot(dx, dy, dz) + 1e-6;
  const base = (1 / dist) * 0.02;
  const emotionalMod = 1 + emotionalContext.valence * 0.5;
  const forceStrength = base * emotionalMod * (dynamicParams.mimicryForceModifier || 1);
  return [dx * forceStrength, dy * forceStrength, dz * forceStrength];
}
