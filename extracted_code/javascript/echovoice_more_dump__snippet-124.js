    fn new() -> Self {
        Self
    }

    fn encode(&self, bond_type: u8) -> Vec<f64> {
        let mut features = vec![0.0; 4];
        
        match bond_type {
            1 => features[0] = 1.0, // Single
            2 => features[1] = 1.0, // Double
            3 => features[2] = 1.0, // Triple
            _ => features[3] = 1.0, // Other/aromatic
        }

        features
    }
