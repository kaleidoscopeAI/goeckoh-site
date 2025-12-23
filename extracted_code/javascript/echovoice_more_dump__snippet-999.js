let p = Path::new(path);
let ext = p.extension().and_then(|os| os.to_str()).unwrap_or("");

if ext == "parquet" {
    if !p.exists() {
        if url.is_empty() {
            return Err(CrystalError::IO(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("File {} does not exist and no download URL provided", path),
            ));
        }
        ensure_dataset_exists(path, url)?;
    }
    load_mutag_dataset(path)
} else {
    // For non-parquet, we don't use the URL. We assume the file exists.
    if !p.exists() {
        return Err(CrystalError::IO(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("File {} does not exist", path),
        )));
    }
    let engine = UniversalMutagEngine::new();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let processed = runtime.block_on(engine.process(p)).map_err(|e| {
        match e {
            ProcessingError::InvalidFormat(s) => CrystalError::Parameter(format!("Invalid format: {}", s)),
            ProcessingError::UnsupportedFormat(s) => CrystalError::Parameter(format!("Unsupported format: {}", s)),
            ProcessingError::DataError(s) => CrystalError::Data(s),
            ProcessingError::IoError(e) => CrystalError::IO(e),
        }
    })?;
    Ok(processed.into_iter().map(|pg| {
        Graph {
            x: pg.mutag_entry.x,
            edge_index: pg.mutag_entry.edge_index,
            edge_attr: pg.mutag_entry.edge_attr,
            y: pg.mutag_entry.y,
            chemical_features: pg.metadata.chemical_properties.map(|cp| ChemicalFeatures {
                molecular_weight: cp.molecular_weight,
                formula: cp.formula,
                smiles: cp.smiles,
                atom_types: cp.atom_types,
                bond_types: cp.bond_types,
                ring_count: cp.ring_count,
                is_aromatic: cp.is_aromatic,
            }),
        }
    }).collect())
}
