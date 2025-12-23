fn new(model_path: &str) -> Self {
    let ctx = WhisperContext::new(model_path).unwrap();
    let state = ctx.create_state().unwrap();

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_translate(false);
    params.set_language(Some("en"));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(true);  // Enable partial results
    params.set_no_context(true);      // Faster for streaming

    Self {
        ctx,
        state,
        params,
        partial_buffer: Vec::with_capacity(16000 * 30), // 30s buffer
    }
}

async fn process_chunk(&mut self, audio: &[f32]) -> Option<String> {
    self.partial_buffer.extend_from_slice(audio);

    // Process when we have enough audio (e.g., 500ms)
    if self.partial_buffer.len() >= 8000 {
        let result = self.state.full(&self.params, &self.partial_buffer).ok()?;

        // Get partial text
        let num_segments = self.state.full_n_segments().unwrap_or(0);
        let mut text = String::new();

        for i in 0..num_segments {
            if let Ok(segment) = self.state.full_get_segment_text(i) {
                text.push_str(&segment);
            }
        }

        // Clear processed audio (keep last 1s for context)
        let keep_len = 16000.min(self.partial_buffer.len());
        self.partial_buffer = self.partial_buffer
            .split_off(self.partial_buffer.len() - keep_len);

        Some(text)
    } else {
        None
    }
}
