pub fn new() -> Self {
    // Register all available processors
    let processors: Vec<Arc<dyn UniversalProcessor>> = vec![
        Arc::new(ChemicalProcessor::new()),
        Arc::new(TabularProcessor),
        Arc::new(TextGraphProcessor::new()),
        Arc::new(ImageGraphProcessor::new()),
        Arc::new(KnowledgeGraphProcessor::new()),
    ];

    Self { processors }
}

/// Select the best processor for a given path
pub async fn select_processor(&self, path: &Path) -> Option<Arc<dyn UniversalProcessor>> {
    let mut scored: Vec<(ProcessorScore, Arc<dyn UniversalProcessor>)> = Vec::new();

    for p in &self.processors {
        let score = p.can_process(path).await;
        if score.can_process {
            scored.push((score, Arc::clone(p)));
        }
    }

    scored.sort_by(|(a, _), (b, _)| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().next().map(|(_, p)| p)
}

/// Main entry: process any file into MUTAG-compatible graphs
pub async fn process(&self, path: &Path) -> Result<Vec<ProcessedGraph>, ProcessingError> {
    if let Some(proc) = self.select_processor(path).await {
        info!("Selected processor: {}", proc.processor_name());
        proc.process_to_mutag(path).await
    } else {
        Err(ProcessingError::UnsupportedFormat(format!(
            "No processor available for {:?}",
            path
        )))
    }
}
