    async fn can_process(&self, path: &Path) -> ProcessorScore {
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "sdf" | "mol" => ProcessorScore::perfect(),
            "xyz" | "pdb" => ProcessorScore::new(0.8, true, 0.7),
            "txt" | "csv" => {
                // Check if file contains chemical identifiers
                if let Ok(content) = async_fs::read_to_string(path).await {
                    if self.detect_chemical_content(&content) {
                        ProcessorScore::new(0.6, true, 0.5)
                    } else {
                        ProcessorScore::cannot_process()
                    }
                } else {
                    ProcessorScore::cannot_process()
                }
            }
            _ => ProcessorScore::cannot_process(),
        }
    }

    async fn process_to_mutag(&self, path: &Path) -> Result<Vec<ProcessedGraph>, ProcessingError> {
        let content = async_fs::read_to_string(path).await.map_err(ProcessingError::IoError)?;
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "sdf" => self.parse_sdf_file(&content).await,
            "mol" => {
                // Single molecule SDF
                self.sdf_to_graph(&content, 0).await.map(|graph| vec![graph])
            }
            _ => Err(ProcessingError::UnsupportedFormat(format!("Extension: {}", ext))),
        }
    }

    fn processor_name(&self) -> &'static str {
        "ChemicalProcessor"
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["sdf", "mol", "xyz", "pdb"]
    }

    fn priority(&self) -> u8 {
        95
    }
