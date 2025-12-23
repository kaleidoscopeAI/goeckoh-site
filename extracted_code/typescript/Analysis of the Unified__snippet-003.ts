// src/workers/embeddedParticleWorker.ts
import { EmbeddedQuantumEngine } from '../engines/EmbeddedQuantumEngine';

const BUFFER_COUNT = 3;
let buffers: Float32Array[] = [];
let availableBuffers: number[] = [];
let positions: Float32Array;
let velocities: Float32Array;
let count = 0;
let step = 0;
let quantumEngine: EmbeddedQuantumEngine;

// Spatial partitioning for performance
class SpatialGrid {
  private cellSize: number;
  private grid: Map<string, number[]>;

  constructor(cellSize: number = 50) {
    this.cellSize = cellSize;
    this.grid = new Map();
  }

  hash(x: number, y: number, z: number): string {
    return `${Math.floor(x/this.cellSize)},${Math.floor(y/this.cellSize)},${Math.floor(z/this.cellSize)}`;
  }

  update(positions: Float32Array): void {
    this.grid.clear();
    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      const key = this.hash(positions[i3], positions[i3+1], positions[i3+2]);
      if (!this.grid.has(key)) this.grid.set(key, []);
      this.grid.get(key)!.push(i);
    }
  }

  getNeighbors(x: number, y: number, z: number, radius: number = this.cellSize): number[] {
    const neighbors: number[] = [];
    const cellRadius = Math.ceil(radius / this.cellSize);
    const baseX = Math.floor(x / this.cellSize);
    const baseY = Math.floor(y / this.cellSize);
    const baseZ = Math.floor(z / this.cellSize);
    
    for (let dx = -cellRadius; dx <= cellRadius; dx++) {
      for (let dy = -cellRadius; dy <= cellRadius; dy++) {
        for (let dz = -cellRadius; dz <= cellRadius; dz++) {
          const key = `${baseX+dx},${baseY+dy},${baseZ+dz}`;
          if (this.grid.has(key)) {
            neighbors.push(...this.grid.get(key)!);
          }
        }
      }
    }
    return neighbors;
  }
}

const spatialGrid = new SpatialGrid(40);

// Enhanced cognitive dynamics with embedded Ollama feedback
class EmbeddedCognitiveDynamics {
  static computeMimicryForce(
    particleIdx: number, 
    positions: Float32Array, 
    velocities: Float32Array, 
    emotionalContext: any,
    dynamicParams: any
  ): [number, number, number] {
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
  }
}

// Main worker execution
self.onmessage = async function(e: MessageEvent) {
  const { cmd, data } = e.data;
  
  if (cmd === 'init') {
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
    
  } else if (cmd === 'update') {
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
    
  } else if (cmd === 'returnBuffer') {
    // Return buffer to pool
    const bufferIndex = buffers.findIndex(buf => buf.buffer === data.buffer);
    if (bufferIndex !== -1) {
      availableBuffers.push(bufferIndex);
    }
  } else if (cmd === 'shutdown') {
    if (quantumEngine) {
      await quantumEngine.shutdown();
    }
  }
};

function calculateEmotionalField(positions: Float32Array, velocities: Float32Array): any {
  let totalValence = 0;
  let totalArousal = 0;
  const sampleSize = Math.min(1000, count);
  
  for (let i = 0; i < sampleSize; i++) {
    const idx = Math.floor((i * count) / sampleSize);
    const v = idx * 3;
    const speed = Math.sqrt(velocities[v]**2 + velocities[v+1]**2 + velocities[v+2]**2);
    totalValence += (velocities[v] > 0 ? 1 : -1) * speed;
    totalArousal += speed;
  }
  
  return {
    valence: Math.tanh(totalValence / sampleSize),
    arousal: Math.tanh(totalArousal / sampleSize / 10)
  };
}
