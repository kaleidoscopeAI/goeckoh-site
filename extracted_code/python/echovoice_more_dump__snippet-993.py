let p = Path::new(path);
let ext = p.extension().and_then(|os| os.to_str()).unwrap_or("");

if ext == "parquet" {
    // Ensure the parquet file exists by downloading if needed and URL is not empty
    if !p.exists() && !url.is_empty() {
        ensure_dataset_exists(path, url)?;
    }
    load_mutag_dataset(path)
} else {
    // For non-parquet, we don't download from the URL. We assume the file exists locally.
    let engine = UniversalMutagEngine::new();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let processed = runtime.block_on(engine.process(p)).map_err(|e| ...)?;
    // Convert processed graphs (Vec<ProcessedGraph>) to Vec<Graph>
    processed.into_iter().map(|pg| {
        Graph {
            x: pg.mutag_entry.x,
            edge_index: pg.mutag_entry.edge_index,
            edge_attr: pg.mutag_entry.edge_attr,
            y: pg.mutag_entry.y,
            chemical_features: pg.metadata.chemical_properties.map(|cp| ...),
        }
    }).collect()
}
