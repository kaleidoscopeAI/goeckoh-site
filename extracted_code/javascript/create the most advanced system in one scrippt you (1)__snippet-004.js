let mut system = StateVector::<8> { values: [0.0; 8] };
let weights = synthesize_contractive_matrix(8);
for _ in 0..1000 { system.contractive_update(weights); }
assert!(system.values.iter().all(|v| v.abs() < 1.0));
