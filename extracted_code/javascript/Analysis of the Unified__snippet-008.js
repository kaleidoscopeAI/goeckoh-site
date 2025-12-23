pub fn update_embeddings(&mut self, result: &TaskResult) {
    // Update node embeddings based on success/failure
    for node in &mut self.nodes {
        let delta = result.success as u128;
        node.embed ^= delta; // Uses bitwise XOR (^) for low-latency update
    }
