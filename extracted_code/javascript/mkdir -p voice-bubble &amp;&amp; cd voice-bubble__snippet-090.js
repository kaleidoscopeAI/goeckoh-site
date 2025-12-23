async fn run(mut self) {
    let (audio_tx, mut audio_rx) = mpsc::channel(100);
    let (text_tx, mut text_rx) = mpsc::channel(100);
    let (tts_tx, mut tts_rx) = mpsc::channel(100);

    self.audio_tx = audio_tx;
    self.text_tx = text_tx;
    self.tts_tx = tts_tx;

    // Start audio ingestion task
    tokio::spawn(async move {
        while let Some(frame) = self.audio_pipeline.input_queue.pop() {
            if self.vad.is_speech(&frame) {
                self.audio_tx.send(frame).await.unwrap();
            }
        }
    });

    // STT processing task
    let stt_task = tokio::spawn(async move {
        while let Some(frame) = audio_rx.recv().await {
            if let Some(text) = self.stt.process_chunk(&frame.samples).await {
                self.text_tx.send(text).await.unwrap();
            }
        }
    });

    // Predictive text + TTS task
    let tts_task = tokio::spawn(async move {
        while let Some(text) = text_rx.recv().await {
            // Get prediction
            let prediction = self.prediction_engine.predict(&text);

            // Synthesize
            let audio = self.tts.synthesize_streaming(&prediction).await;

            // Queue for playback
            self.audio_pipeline.output_queue.push_slice(&audio);
        }
    });

    tokio::join!(stt_task, tts_task);
}
