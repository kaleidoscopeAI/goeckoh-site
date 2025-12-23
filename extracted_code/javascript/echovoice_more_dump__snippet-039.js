pub fn verify_security(state_a: &[f32], state_b: &[f32], threshold: f32) -> bool {
let ca = compute_state_commitment(state_a);
let cb = compute_state_commitment(state_b);
