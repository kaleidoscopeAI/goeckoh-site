struct PredictiveTextEngine {
    ngram_cache: LruCache<Vec<String>, Vec<(String, f32)>>,
    max_context: usize,
}

impl PredictiveTextEngine {
    fn predict(&mut self, context: &str) -> String {
        let words: Vec<String> = context
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();
        
        // Take last N words as context
        let start = if words.len() > self.max_context {
            words.len() - self.max_context
        } else {
            0
        };
        let context_slice = &words[start..];
        
        // Look up in cache or compute prediction
        if let Some(predictions) = self.ngram_cache.get(context_slice) {
            predictions[0].0.clone()
        } else {
            // Fallback to simple completion
            let last_word = context_slice.last().unwrap_or(&"".to_string());
            format!("{}...", last_word)
        }
    }
}
