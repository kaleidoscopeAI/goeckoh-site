    async fn can_process(&self, path: &Path) -> ProcessorScore {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        if ["ttl", "rdf", "owl"].contains(&ext.as_str()) {
            ProcessorScore::new(0.85, true, 0.75)
        } else {
            ProcessorScore::cannot_process()
        }
    }

    async fn process_to_mutag(&self, path: &Path) -> Result<Vec<ProcessedGraph>, ProcessingError> {
        let content = async_fs::read_to_string(path).await.map_err(ProcessingError::IoError)?;
        let mutag_entry = self.build_graph_from_kg(&content).await;

        Ok(vec![ProcessedGraph {
            id: Uuid::new_v4().to_string(),
            mutag_entry,
            metadata: GraphMetadata {
                source_path: path.to_string_lossy().to_string(),
                data_type: DataType::KnowledgeGraph,
                original_format: "kg".to_string(),
                num_nodes: mutag_entry.x.len(),
                num_edges: mutag_entry.edge_index.len() / 2,
                confidence_score: 0.85,
                chemical_properties: None,
                processing_timestamp: chrono::Utc::now(),
            },
            processing_stats: ProcessingStats {
                processing_time_ms: 15,
                memory_used_kb: 0,
                warnings: Vec::new(),
                success: true,
                processor_used: "KnowledgeGraphProcessor".to_string(),
            },
        }])
    }

    fn processor_name(&self) -> &'static str { "KnowledgeGraphProcessor" }
    fn supported_extensions(&self) -> Vec<&'static str> { vec!["ttl", "rdf", "owl"] }
    fn priority(&self) -> u8 { 70 }
