count = data.count;

// Initialize triple buffering
buffers = [];
availableBuffers = [];
const initialPositions = new Float32Array(data.positions);

for (let i = 0; i < BUFFER_COUNT; i++) {
  const buffer = new Float32Array(count * 3);
  buffer.set(initialPositions);
  buffers.push(buffer);
  availableBuffers.push(i);
}

positions = buffers[0];
velocities = new Float32Array(count * 3);
for (let i = 0; i < velocities.length; i++) {
  velocities[i] = (Math.random() - 0.5) * 2;
}

// Initialize embedded quantum engine
quantumEngine = new EmbeddedQuantumEngine(count);

self.postMessage({ cmd: 'ready' });

