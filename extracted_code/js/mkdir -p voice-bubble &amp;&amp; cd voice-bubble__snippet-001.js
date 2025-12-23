// Current: energy → scale
bubble.scale.setScalar(0.92 + 0.45 * feat.energy);

// Goeckoh: lattice energy → scale, coherence → spikes
bubble.scale.setScalar(latticeEnergy * 1.2);
spikeUniform.value = 1.0 - latticeCoherence;  // Low coherence = spiky
