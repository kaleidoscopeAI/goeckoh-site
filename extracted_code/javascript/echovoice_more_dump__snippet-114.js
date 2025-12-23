    fn detect_chemical_content(&self, content: &str) -> bool {
        let chemical_indicators = [
            "V2000", "V3000", // MDL format indicators
            "ATOM", "BOND", "HETATM", // PDB format
            r"^\s*\d+\s*$", // XYZ format (number of atoms)
            r"[A-Z][a-z]?\s+[-+]?\d*\.?\d+\s+[-+]?\d*\.?\d+\s+[-+]?\d*\.?\d+", // Atomic coordinates
        ];

        chemical_indicators.iter().any(|&pattern| {
            regex::Regex::new(pattern).unwrap().is_match(content)
        })
    }
