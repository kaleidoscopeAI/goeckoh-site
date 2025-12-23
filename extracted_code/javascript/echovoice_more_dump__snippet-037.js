pub fn compute_state_commitment(state: &[f32]) -> Vec<u8> {
let mut hasher = Sha256::new();
