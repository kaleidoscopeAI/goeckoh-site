fn compute_commitment_vec(vals: &[f32]) -> Vec<u8> {
let mut hasher = Sha256::new();
