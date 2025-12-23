pub fn entanglement_score(a: &[u8], b: &[u8]) -> f32 {
let mut same = 0usize;
let tot = std::cmp::min(a.len(), b.len());
