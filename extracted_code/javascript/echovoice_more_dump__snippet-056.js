let commit = compute_commitment_vec(&buf);
let out = env.new_byte_array(commit.len() as i32).unwrap();
