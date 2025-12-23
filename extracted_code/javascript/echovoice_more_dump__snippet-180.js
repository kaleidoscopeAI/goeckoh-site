    pub async fn process(&self, opportunity: Opportunity) -> Result<Vec<Action>, PipelineError> {
        let mut current = opportunity;
        for stage in &self.stages {
            current = stage.execute(current).await?;
        }
        self.generate_actions(current).await
    }
