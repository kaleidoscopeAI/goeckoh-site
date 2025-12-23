    fn new() -> Self {
        let mut atom_types = HashMap::new();
        let atoms = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "H", "Unknown"];
        for (i, atom) in atoms.iter().enumerate() {
            atom_types.insert(atom.to_string(), i);
        }
        Self { atom_types }
    }

    fn encode(&self, symbol: &str, _x: f64, _y: f64, _z: f64) -> Vec<f64> { // Coordinates not used in encoding for simplicity
        let mut features = vec![0.0; 11];
        
        if let Some(&idx) = self.atom_types.get(symbol) {
            features[idx.min(10)] = 1.0;
        } else {
            features[10] = 1.0; // Unknown
        }

        features
    }
