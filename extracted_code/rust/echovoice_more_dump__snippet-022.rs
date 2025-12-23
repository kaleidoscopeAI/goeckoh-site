    async fn can_process(&self, path: &Path) -> ProcessorScore;
    async fn process_to_mutag(&self, path: &Path) -> Result<Vec<ProcessedGraph>, ProcessingError>;
    fn processor_name(&self) -> &'static str;
    fn supported_extensions(&self) -> Vec<&'static str>;
    fn priority(&self) -> u8;
