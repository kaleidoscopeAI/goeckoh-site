pub fn new() -> Self { Self }

async fn build_graph_from_kg(&self, content: &str) -> MutagEntry {
    // Simple triple parser for RDF-like
    let lines = content.lines();
    let mut node_map = HashMap::new();
    let mut next_id = 0;
    let mut edge_index = Vec::new();
    let mut edge_attr = Vec::new();
    for line in lines {
        let parts = line.split_whitespace().collect::<Vec<&str>>();
        if parts.len() >= 3 {
            let sub = parts[0].to_string();
            let pred = parts[1];
            let obj = parts[2].to_string();

            let sub_id = *node_map.entry(sub).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            let obj_id = *node_map.entry(obj).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });

            edge_index.push(vec![sub_id, obj_id]);
            edge_index.push(vec![obj_id, sub_id]);
            // Edge feature based on pred hash mod 4
            let mut attr = vec![0.0; 4];
            attr[(pred.len() % 4)] = 1.0;
            edge_attr.push(attr.clone());
            edge_attr.push(attr);
        }
    }

    let num_nodes = node_map.len();
    let mut x = vec![vec![0.0; 11]; num_nodes];
    for i in 0..num_nodes {
        x[i][i % 11] = 1.0;
    }

    MutagEntry { x, edge_index, edge_attr, y: 0 }
}
