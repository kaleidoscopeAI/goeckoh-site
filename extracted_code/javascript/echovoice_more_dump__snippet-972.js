pub fn new() -> Self {
    Self {
        node_feature_dim: 11, // Standard atomic features
        edge_feature_dim: 4,  // Bond type, aromatic, conjugated, ring
        atom_encoder: AtomEncoder::new(),
        bond_encoder: BondEncoder::new(),
    }
}

async fn parse_sdf_file(&self, content: &str) -> Result<Vec<ProcessedGraph>, ProcessingError> {
    let mut graphs = Vec::new();
    let molecules = self.split_sdf_molecules(content);

    for (idx, mol_block) in molecules.iter().enumerate() {
        if let Ok(graph) = self.sdf_to_graph(mol_block, idx).await {
            graphs.push(graph);
        }
    }

    Ok(graphs)
}

fn split_sdf_molecules(&self, content: &str) -> Vec<String> {
    content
        .split("$$$$")
        .filter(|block| !block.trim().is_empty())
        .map(|block| block.to_string())
        .collect()
}

async fn sdf_to_graph(&self, mol_block: &str, idx: usize) -> Result<ProcessedGraph, ProcessingError> {
    let start_time = Instant::now();
    let lines: Vec<&str> = mol_block.lines().collect();

    if lines.len() < 4 {
        return Err(ProcessingError::InvalidFormat("SDF block too short".to_string()));
    }

    // Parse counts line (line 3)
    let counts_line = lines.get(3).ok_or_else(|| 
        ProcessingError::InvalidFormat("Missing counts line".to_string()))?;

    let counts: Vec<&str> = counts_line.split_whitespace().collect();
    let num_atoms = counts.get(0).unwrap_or(&"0").parse::<usize>().unwrap_or(0);
    let num_bonds = counts.get(1).unwrap_or(&"0").parse::<usize>().unwrap_or(0);

    if num_atoms == 0 {
        return Err(ProcessingError::InvalidFormat("No atoms in molecule".to_string()));
    }

    // Parse atom block
    let mut atoms = Vec::new();
    let mut node_features = Vec::new();

    for i in 4..(4 + num_atoms) {
        if let Some(line) = lines.get(i) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                let atom_symbol = parts[3];
                let x = parts[0].parse::<f64>().unwrap_or(0.0);
                let y = parts[1].parse::<f64>().unwrap_or(0.0);
                let z = parts[2].parse::<f64>().unwrap_or(0.0);

                atoms.push(atom_symbol.to_string());

                // Encode atomic features
                let features = self.atom_encoder.encode(atom_symbol, x, y, z);
                node_features.push(features);
            }
        }
    }

    // Parse bond block
    let mut edges = Vec::new();
    let mut edge_features = Vec::new();
    let bond_start = 4 + num_atoms;

    for i in bond_start..(bond_start + num_bonds) {
        if let Some(line) = lines.get(i) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                let atom1 = parts[0].parse::<usize>().unwrap_or(1).saturating_sub(1);
                let atom2 = parts[1].parse::<usize>().unwrap_or(1).saturating_sub(1);
                let bond_type = parts[2].parse::<u8>().unwrap_or(1);

                if atom1 < num_atoms && atom2 < num_atoms {
                    // Add edge in both directions for undirected graph
                    edges.push(vec![atom1, atom2]);
                    edges.push(vec![atom2, atom1]);

                    let bond_features = self.bond_encoder.encode(bond_type);
                    edge_features.push(bond_features.clone());
                    edge_features.push(bond_features);
                }
            }
        }
    }

    // Determine if molecule is mutagenic (simplified heuristic)
    let y = self.predict_mutagenicity(&atoms, &edges);

    let mutag_entry = MutagEntry {
        x: node_features,
        edge_index: edges,
        edge_attr: edge_features,
        y,
    };

    let chemical_props = ChemicalProperties {
        molecular_weight: Some(self.calculate_molecular_weight(&atoms)),
        formula: Some(self.generate_molecular_formula(&atoms)),
        smiles: None, // Would require proper cheminformatics library
        atom_types: atoms.clone(),
        bond_types: vec!["single".to_string(), "double".to_string()], // Simplified
        ring_count: self.count_rings(&edges, num_atoms),
        is_aromatic: self.detect_aromaticity(&atoms, &edges),
    };

    let metadata = GraphMetadata {
        source_path: format!("molecule_{}", idx),
        data_type: DataType::Chemical,
        original_format: "SDF".to_string(),
        num_nodes: num_atoms,
        num_edges: edges.len() / 2, // Undirected, so divide by 2
        confidence_score: 0.9,
        chemical_properties: Some(chemical_props),
        processing_timestamp: chrono::Utc::now(),
    };

    let processing_stats = ProcessingStats {
        processing_time_ms: start_time.elapsed().as_millis() as u64,
        memory_used_kb: 0, // Would implement memory tracking
        warnings: Vec::new(),
        success: true,
        processor_used: "ChemicalProcessor".to_string(),
    };

    Ok(ProcessedGraph {
        id: Uuid::new_v4().to_string(),
        mutag_entry,
        metadata,
        processing_stats,
    })
}

fn predict_mutagenicity(&self, atoms: &[String], edges: &[Vec<usize>]) -> i32 {
    // Simplified mutagenicity prediction based on structural features
    let mut score = 0;

    // Presence of certain atoms increases mutagenicity likelihood
    for atom in atoms {
        match atom.as_str() {
            "N" => score += 2,
            "O" => score += 1,
            "S" => score += 3,
            "Cl" | "Br" | "F" => score += 4,
            _ => {}
        }
    }

    // High connectivity suggests aromatic systems
    let avg_connectivity = if !atoms.is_empty() {
        edges.len() as f32 / atoms.len() as f32
    } else {
        0.0
    };

    if avg_connectivity > 2.5 {
        score += 3;
    }

    // Threshold-based classification
    if score > 8 { 1 } else { 0 }
}

fn calculate_molecular_weight(&self, atoms: &[String]) -> f64 {
    atoms.iter().map(|atom| self.get_atomic_weight(atom)).sum()
}

fn get_atomic_weight(&self, symbol: &str) -> f64 {
    match symbol {
        "H" => 1.008,
        "C" => 12.011,
        "N" => 14.007,
        "O" => 15.999,
        "F" => 18.998,
        "P" => 30.974,
        "S" => 32.065,
        "Cl" => 35.453,
        "Br" => 79.904,
        "I" => 126.904,
        _ => 12.011, // Default to carbon
    }
}

fn generate_molecular_formula(&self, atoms: &[String]) -> String {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for atom in atoms {
        *counts.entry(atom.clone()).or_insert(0) += 1;
    }

    let mut formula = String::new();
    // Standard order: C, H, then alphabetical
    if let Some(&c_count) = counts.get("C") {
        if c_count == 1 {
            formula.push('C');
        } else {
            write!(&mut formula, "C{}", c_count).unwrap();
        }
    }
    if let Some(&h_count) = counts.get("H") {
        if h_count == 1 {
            formula.push('H');
        } else {
            write!(&mut formula, "H{}", h_count).unwrap();
        }
    }

    let mut other_elements: Vec<_> = counts.iter()
        .filter(|(element, _)| element.as_str() != "C" && element.as_str() != "H")
        .collect();
    other_elements.sort_by_key(|(element, _)| element.as_str());

    for (element, &count) in other_elements {
        if count == 1 {
            formula.push_str(element);
        } else {
            write!(&mut formula, "{}{}", element, count).unwrap();
        }
    }

    formula
}

fn count_rings(&self, edges: &[Vec<usize>], num_atoms: usize) -> usize {
    // Simplified ring detection using Euler characteristic heuristic for planar graphs
    let edge_count = edges.len() / 2; // Undirected edges
    if edge_count >= num_atoms {
        edge_count - num_atoms + 1
    } else {
        0
    }
}

fn detect_aromaticity(&self, atoms: &[String], edges: &[Vec<usize>]) -> bool {
    // Simple heuristic: high carbon content + high connectivity
    let carbon_ratio = atoms.iter().filter(|&a| a == "C").count() as f64 / atoms.len() as f64;
    let avg_connectivity = edges.len() as f64 / atoms.len() as f64;

    carbon_ratio > 0.6 && avg_connectivity > 2.0
}
