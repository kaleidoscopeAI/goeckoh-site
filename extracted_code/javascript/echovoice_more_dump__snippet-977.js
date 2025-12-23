async fn can_process(&self, path: &Path) -> ProcessorScore {
    let ext = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "csv" | "tsv" => ProcessorScore::new(0.8, true, 0.6),
        "parquet" => ProcessorScore::new(0.9, true, 0.8),
        _ => ProcessorScore::cannot_process(),
    }
}

async fn process_to_mutag(&self, path: &Path) -> Result<Vec<ProcessedGraph>, ProcessingError> {
    let start_time = Instant::now();

    // Load data based on format
    let df = match path.extension().and_then(|e| e.to_str()) {
        Some("csv") => CsvReader::from_path(path).unwrap().has_header(true).finish().map_err(|e| ProcessingError::DataError(e.to_string()))?,
        Some("parquet") => ParquetReader::new(std::fs::File::open(path).map_err(ProcessingError::IoError)?).finish().map_err(|e| ProcessingError::DataError(e.to_string()))?,
        _ => return Err(ProcessingError::UnsupportedFormat("Not CSV or Parquet".to_string())),
    };

    // Check if this is already in MUTAG format
    if self.is_mutag_format(&df) {
        return self.convert_existing_mutag(&df, path).await;
    }

    // Try to convert tabular data to graph format
    self.tabular_to_graph(&df, path, start_time).await
}

fn processor_name(&self) -> &'static str {
    "TabularProcessor"
}

fn supported_extensions(&self) -> Vec<&'static str> {
    vec!["csv", "parquet", "tsv"]
}

fn priority(&self) -> u8 {
    60
}
