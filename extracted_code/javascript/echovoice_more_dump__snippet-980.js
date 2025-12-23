async fn can_process(&self, path: &Path) -> ProcessorScore {
    if path.extension().and_then(|e| e.to_str()) == Some("txt") {
        ProcessorScore::new(0.7, true, 0.6)
    } else {
        ProcessorScore::cannot_process()
    }
}

async fn process_to_mutag(&self, path: &Path) -> Result<Vec<ProcessedGraph>, ProcessingError> {
    let text = async_fs::read_to_string(path).await.map_err(ProcessingError::IoError)?;
    let entities = self.extract_entities(&text);
    let mutag_entry = self.build_graph(entities);

    Ok(vec![ProcessedGraph {
        id: Uuid::new_v4().to_string(),
        mutag_entry,
        metadata: GraphMetadata {
            source_path: path.to_string_lossy().to_string(),
            data_type: DataType::TextGraph,
            original_format: "txt".to_string(),
            num_nodes: mutag_entry.x.len(),
            num_edges: mutag_entry.edge_index.len() / 2,
            confidence_score: 0.7,
            chemical_properties: None,
            processing_timestamp: chrono::Utc::now(),
        },
        processing_stats: ProcessingStats {
            processing_time_ms: 10,
            memory_used_kb: 0,
            warnings: Vec::new(),
            success: true,
            processor_used: "TextGraphProcessor".to_string(),
        },
    }])
}

fn processor_name(&self) -> &'static str { "TextGraphProcessor" }
fn supported_extensions(&self) -> Vec<&'static str> { vec!["txt"] }
fn priority(&self) -> u8 { 40 }
