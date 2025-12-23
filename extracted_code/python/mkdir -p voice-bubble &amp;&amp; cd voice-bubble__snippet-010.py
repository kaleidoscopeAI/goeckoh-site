const latticeEnergy = lattice ? lattice.lattice_energy : (0.65 + 1.35 * feat.energy);
const latticeCoherence = lattice ? lattice.lattice_coherence : (1.0 - clamp((feat.zcr - 0.02) / 0.18, 0, 1));
const latticeValence = lattice ? lattice.lattice_valence : 0.5;
const latticeArousal = lattice ? lattice.lattice_arousal : feat.energy;

// Scale from lattice energy
