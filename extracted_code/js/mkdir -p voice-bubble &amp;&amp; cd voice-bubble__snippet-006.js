// ===== LATTICE HEART COMPUTE =====
let latticeEnergy = feat.energy * 1.8;
let latticeCoherence = 1.0;
let latticeValence = 0.5;
let latticeArousal = feat.energy;

if (wasmReady && feat.energy > 0.02) {
  try {
    const lattice = lattice_from_features(feat.energy, feat.f0, feat.zcr, feat.hnr, feat.tilt, feat.dt);
    latticeEnergy = lattice.lattice_energy;
    latticeCoherence = lattice.lattice_coherence;
    latticeValence = lattice.lattice_valence;
    latticeArousal = lattice.lattice_arousal;
  } catch(e) { /* fallback */ }
}

// ===== PHYSICS MAPPING =====
bubble.scale.setScalar(0.85 + 0.55 * latticeEnergy);
spikeUniform.value = 1.0 - latticeCoherence;
material.roughness = 0.06 + (1.0 - latticeValence) * 0.55 + latticeArousal * 0.25;
material.metalness = 0.05 + latticeValence * 0.65;

// ===== DSP FEEDBACK (worklet gets this) =====
if (workletNode) {
  workletNode.port.postMessage({
    type: "lattice",
    energy: latticeEnergy,
    coherence: latticeCoherence,
    valence: latticeValence,
    arousal: latticeArousal
  });
}
