    fn new() -> Self {
        Self {
            decision_engine: DecisionEngine::new(),
            metrics_history: Vec::with_capacity(1000),
            total_decisions: 0,
            last_update: 0,
        }
    }
