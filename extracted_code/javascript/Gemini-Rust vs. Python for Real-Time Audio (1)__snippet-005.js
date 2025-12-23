/// Constructor: Called by the Android/iOS app on startup.
#[uniffi::constructor]
pub fn new() -> Self {
    #[cfg(target_os = "android")]
    android_logger::init_once(
        android_logger::Config::default().with_min_level(log::Level::Info),
    );

    Self {
        inner_state: Arc::new(Mutex::new(InnerState {
            is_running: false,
            audio_stream: None, // Initially, silence.
        })),
    }
}

/// Starts the audio processing loop.
pub fn start_engine(&self) -> Result<(), GoeckohError> {
    let mut state = self.inner_state.lock().unwrap();

    if state.is_running {
        return Err(GoeckohError::AlreadyRunning);
    }

    log::info!("Goeckoh Engine Starting: Acquiring Audio Hardware...");

    // Attempt to build the audio stream using our new module.
    // If it fails (e.g., mic permission denied), we return an error to the UI.
    let stream = AudioStreamManager::new().map_err(|e| {
        log::error!("CRITICAL: Audio initialization failed: {}", e);
        GoeckohError::AudioInitFailed
    })?;

    // Store the active stream. This keeps the audio thread alive.
    state.audio_stream = Some(stream);
    state.is_running = true;

    log::info!("Engine Active. Feedback loop running.");
    Ok(())
}

/// Stops the engine and releases audio resources immediately.
pub fn stop_engine(&self) {
    let mut state = self.inner_state.lock().unwrap();
    if state.is_running {
        log::info!("Goeckoh Engine Stopping...");

        // This is the magic line. 
        // Setting this to None calls the Destructor of AudioStreamManager,
        // which stops the CPAL stream and releases the Microphone.
        state.audio_stream = None; 

        state.is_running = false;
    }
}

pub fn get_current_state(&self) -> EmotionalState {
    EmotionalState {
        valence: 0.75,
        arousal: 0.45,
        coherence: 0.98,
        is_stable: true,
    }
}
