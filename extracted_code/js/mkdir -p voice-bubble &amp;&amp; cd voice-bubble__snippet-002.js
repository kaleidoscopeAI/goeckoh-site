// Modulate output based on lattice valence/arousal
const valenceMod = latticeValence > 0.5 ? 1.0 : 0.7;  // High valence = clear
const arousalTilt = latticeArousal * -0.8;             // High arousal = dark tilt
