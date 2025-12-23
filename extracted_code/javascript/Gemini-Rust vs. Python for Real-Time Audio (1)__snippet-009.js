#[uniffi::constructor]
pub fn new() -> Self {
    #[cfg(target_os = "android")]
    android_logger::init_once(
        android_logger::Config::default().with_min_level(log::Level::Info),
    );

    Self {
        inner_state: Arc::new(Mutex::new(InnerState {
            is_running: false,
            // Initialize the lattice wrapped in its own Arc<Mutex>
            lattice: Arc::new(Mutex::new(CrystallineLattice::new())),
            audio_stream: None,
        })),
    }
}

pub fn start_engine(&self) -> Result<(), GoeckohError> {
    let mut state = self.inner_state.lock().unwrap();

    if state.is_running {
        return Err(GoeckohError::AlreadyRunning);
    }

    log::info!("Goeckoh Engine Starting...");

    // Pass a CLONE of the lattice handle to the audio manager.
    // This increments the reference count, allowing the audio thread
    // to access the same memory address as the UI thread.
    let lattice_ref = state.lattice.clone();

    let stream = AudioStreamManager::new(lattice_ref).map_err(|e| {
        log::error!("CRITICAL: Audio initialization failed: {}", e);
        GoeckohError::AudioInitFailed
    })?;

    state.audio_stream = Some(stream);
    state.is_running = true;

    Ok(())
}

pub fn stop_engine(&self) {
    let mut state = self.inner_state.lock().unwrap();
    if state.is_running {
        log::info!("Goeckoh Engine Stopping...");
        state.audio_stream = None; 
        state.is_running = false;
    }
}

pub fn get_current_state(&self) -> EmotionalState {
    let state = self.inner_state.lock().unwrap();

    // Lock the lattice to read stats. 
    // NOTE: This might block the audio thread for a microsecond.
    // The audio thread uses try_lock() to avoid waiting on us.
    let lattice = state.lattice.lock().unwrap();

    let (valence, arousal, coherence) = lattice.measure_affective_state();
    let is_stable = coherence > 0.4;

    EmotionalState {
        valence,
        arousal,
        coherence,
        is_stable,
    }
}
