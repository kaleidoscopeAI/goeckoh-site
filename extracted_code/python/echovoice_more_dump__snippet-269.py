fn load_or_process_dataset(path: &str, url: &str) -> Result<Vec<Graph>, CrystalError> {
    ensure_dataset_exists(path, url)?;
    let p = Path::new(path);
    let ext = p.extension().and_then(|os| os.to_str()).unwrap_or("");

    if ext == "parquet" {
        load_mutag_dataset(path)
    } else {
        let engine = UniversalMutagEngine::new();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let processed = runtime.block_on(engine.process(p)).map_err(|e| CrystalError::Parameter(format!("Processing error: {:?}", e)))?;
        processed.into_iter().map(|pg| {
            Graph {
                x: pg.mutag_entry.x,
                edge_index: pg.mutag_entry.edge_index,
                edge_attr: pg.mutag_entry.edge_attr,
                y: pg.mutag_entry.y,
                // Assume ChemicalFeatures maps from ChemicalProperties
                chemical_features: pg.metadata.chemical_properties.map(|cp| ChemicalFeatures {
                    molecular_weight: cp.molecular_weight,
                    formula: cp.formula,
                    // Map other fields accordingly
                    // Assuming similar struct
                    smiles: cp.smiles,
                    atom_types: cp.atom_types,
                    bond_types: cp.bond_types,
                    ring_count: cp.ring_count,
                    is_aromatic: cp.is_aromatic,
                }),
            }
        }).collect()
    }
