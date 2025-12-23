// ===== GOECKOH LATTICE INTEGRATION =====
let lattice = null;
if (wasmReady && feat.energy > 0.02) {  // Only compute when voiced
  try {
    lattice = lattice_from_features(
      feat.energy, feat.f0, feat.zcr, feat.hnr, feat.tilt, feat.dt
    );
  } catch(e) {
    console.warn("Lattice compute:", e);
  }
}

// FALLBACK: original mapping if WASM cold/offline
const latticeEnergy = lattice?.lattice_energy ?? (0.65 + 1.35 * feat.energy);
const latticeCoherence = lattice?.lattice_coherence ?? (1.0 - clamp((feat.zcr - 0.02) / 0.18, 0, 1));
const latticeValence = lattice?.lattice_valence ?? 0.5;
const latticeArousal = lattice?.lattice_arousal ?? feat.energy;

// ===== HEART-DRIVEN PHYSICS =====
// Scale: lattice energy (total activation)
bubble.scale.setScalar(0.85 + 0.55 * latticeEnergy);

// Spikes: inverse coherence (chaotic = spiky)
spikeUniform.value = clamp(1.0 - latticeCoherence, 0, 1);

// Roughness: low valence + high arousal = rough surface
material.roughness = clamp(0.06 + (1.0 - latticeValence) * 0.55 + latticeArousal * 0.25, 0.04, 0.95);

// Metalness: valence-driven "clarity" 
material.metalness = clamp(0.05 + latticeValence * 0.65, 0.02, 0.9);

// Color: arousal modulates hue saturation toward red/orange when high
const baseColor = pitchToRGB(feat.f0);
const arousalTint = latticeArousal > 0.3 ? new THREE.Color(1.0, 0.4 + 0.4*latticeValence, 0.2) : baseColor;
material.color.lerpColors(baseColor, arousalTint, latticeArousal * 0.6);
