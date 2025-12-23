    let mut output = vec![0u64; inputs[^58_0].len()];
    for (src, input_bits) in inputs.iter().enumerate() {
        if self.routing_matrix[idx][src] {
            for (i, &b) in input_bits.iter().enumerate() {
                output[i] |= b;
            }
        }
    }
    output
