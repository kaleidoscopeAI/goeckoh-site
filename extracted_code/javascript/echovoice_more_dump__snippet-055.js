let ca = compute_commitment_vec(&va);
let cb = compute_commitment_vec(&vb);
let mut same = 0usize;
let tot = std::cmp::min(ca.len(), cb.len());
