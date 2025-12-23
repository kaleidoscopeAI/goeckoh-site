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
