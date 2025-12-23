    pub fn add_node(&mut self, node: Node) -> Result<usize> {
        let idx = self.nodes.len();
        self.nodes.push(node);
        self.adjacency_list.insert(idx, Vec::new());
        Ok(idx)
    }

Now, let's remove the unstable features from src/lib.rs:

