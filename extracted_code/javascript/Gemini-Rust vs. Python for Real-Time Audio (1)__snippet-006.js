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
            lattice: CrystallineLattice::new(), // Initialize the physics grid
            audio_stream: None,
        })),
    }
}

/// Starts the audio processing loop.
pub fn start_engine(&self) -> Result<(), GoeckohError> {
    let mut state = self.inner_state.lock().unwrap();

    if state.is_running {
        return Err(GoeckohError::AlreadyRunning);
    }

    log::info!("Goeckoh Engine Starting...");

    // 1. Initialize Audio
    // Note: In the next iteration, we will pass the 'lattice' into the 
    // AudioStreamManager so the audio thread can drive the physics.
    let stream = AudioStreamManager::new().map_err(|e| {
        log::error!("CRITICAL: Audio initialization failed: {}", e);
        GoeckohError::AudioInitFailed
    })?;

    state.audio_stream = Some(stream);
    state.is_running = true;

    log::info!("Engine Active. Physics & Audio Systems Online.");
    Ok(())
}

/// Stops the engine.
pub fn stop_engine(&self) {
    let mut state = self.inner_state.lock().unwrap();
    if state.is_running {
        log::info!("Goeckoh Engine Stopping...");
        state.audio_stream = None; // Kills audio hardware connection
        state.is_running = false;
    }
}

/// Called by the UI loop (e.g., every 60fps).
/// Now returns REAL physics data from the lattice.
pub fn get_current_state(&self) -> EmotionalState {
    let state = self.inner_state.lock().unwrap();

    // 1. Ask the lattice for its current energy metrics
    let (valence, arousal, coherence) = state.lattice.measure_affective_state();

    // 2. Determine stability (Safety Throttle)
    // If coherence drops too low, the system is "Unstable"
    let is_stable = coherence > 0.4;

    EmotionalState {
        valence,
        arousal,
        coherence,
        is_stable,
    }
}

/// MANUAL TICK (For Testing/Visualization without Audio)
/// Allows the UI to advance the physics simulation even if the mic is off.
pub fn tick_simulation(&self, dt: f32) {
    let mut state = self.inner_state.lock().unwrap();
    state.lattice.update(dt);
}
