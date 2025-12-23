// Normalize HNR into 0..1 where 0 = very noisy, 1 = very periodic
let h = clamp((hnr_db + 20.0) / 60.0, 0.0, 1.0);

// Normalize tilt into 0..1 where 0 = dark/muffled, 1 = bright
let t = clamp((tilt + 2.0) / 4.0, 0.0, 1.0);

// f0 factor: prefer 120..260 for "steady voiced" coherence boost
let f0 = if f0_hz.is_finite() { f0_hz.max(0.0) } else { 0.0 };
let f0_mid = if f0 > 0.0 {
    let x = (f0 - 190.0) / 70.0; // centered ~190Hz
    (1.0 - clamp(x.abs(), 0.0, 1.0)).powf(1.5)
} else {
    0.0
};

// “lattice_energy” ~ activation level (K-like)
let lattice_energy = 0.65 + 1.35 * (0.65 * e + 0.35 * sigmoid(3.5 * (z - 0.08)));

// Coherence: penalize high zcr and low hnr; small boost from mid f0
let chaos = 0.55 * z + 0.45 * (1.0 - h);
let lattice_coherence = clamp(1.0 - chaos + 0.10 * f0_mid, 0.0, 1.0);

// Arousal: energy + consonant “edge” (zcr) - calmness from periodicity
let lattice_arousal = clamp(
    0.10 + 0.75 * e + 0.25 * sigmoid(4.0 * (z - 0.10)) - 0.15 * h,
    0.0, 1.0
);

// Valence: brighter tilt + higher periodicity; too much arousal pulls down slightly
let lattice_valence = clamp(
    0.20 + 0.55 * t + 0.35 * h - 0.15 * lattice_arousal,
    0.0, 1.0
);

// dt sanity clamp (not used heavily, but returned if you want to log)
let _dt = clamp(dt, 0.002, 0.05);

let state = LatticeState {
    lattice_energy,
    lattice_coherence,
    lattice_valence,
    lattice_arousal,
};

serde_wasm_bindgen::to_value(&state).unwrap()
}
