pub fn new() -> Self { Self }

fn extract_entities(&self, text: &str) -> Vec<String> {
    // Simple word-based extraction as real NLP not available
    text.split_whitespace()
        .filter(|w| w.len() > 3 && w.chars().all(char::is_alphabetic))
        .map(|w| w.to_string())
        .collect()
}

fn build_graph(&self, entities: Vec<String>) -> MutagEntry {
    let num_nodes = entities.len();
    let mut x = Vec::with_capacity(num_nodes);
    let mut edge_index = Vec::new();
    let mut edge_attr = Vec::new();

    for (i, _ent) in entities.iter().enumerate() {
        // Simple feature: one-hot based on index mod 11
        let mut features = vec![0.0; 11];
        features[i % 11] = 1.0;
        x.push(features);

        // Chain edges
        if i > 0 {
            edge_index.push(vec![i - 1, i]);
            edge_index.push(vec![i, i - 1]);
            edge_attr.push(vec![1.0, 0.0, 0.0, 0.0]);
            edge_attr.push(vec![1.0, 0.0, 0.0, 0.0]);
        }
    }

    MutagEntry { x, edge_index, edge_attr, y: 0 }
}
