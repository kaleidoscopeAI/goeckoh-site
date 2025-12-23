use std::sync::{Arc, Mutex};
use thiserror::Error;

// This macro generates the bindings for Kotlin and Swift automatically.
uniffi::setup_scaffolding!("goeckoh_core");

/// Represents the "Affective State" of the Crystalline Heart.
/// This simple struct is passed back to the UI for visualization.
#[derive(Debug, Clone, uniffi::Record)]
pub struct EmotionalState {
    pub valence: f32,    // Positivity (0.0 to 1.0)
    pub arousal: f32,    // Energy/Intensity (0.0 to 1.0)
    pub coherence: f32,  // Global Coherence Level (GCL)
    pub is_stable: bool, // "Safety Throttle" indicator
}

/// Errors that can occur within the engine.
/// These map to standard Exceptions/Errors in Java and Swift.
#[derive(Error, Debug, uniffi::Error)]
pub enum GoeckohError {
    #[error("Failed to initialize audio device")]
    AudioInitFailed,
    #[error("The crystalline lattice has destabilized")]
    LatticeDestabilized,
    #[error("System is already running")]
    AlreadyRunning,
}

/// The main entry point for the mobile application.
/// It holds the state and manages the lifecycle of the engine.
#[derive(uniffi::Object)]
pub struct GoeckohEngine {
    // We use Arc<Mutex<T>> because this state is shared between
    // the UI thread (calling this object) and the background Audio thread.
    inner_state: Arc<Mutex<InnerState>>,
}

/// Internal state not exposed to the outside world.
struct InnerState {
    is_running: bool,
    // Placeholder: This will eventually hold your ODE Solver
    // lattice: CrystallineLattice, 
}

#[uniffi::export]
impl GoeckohEngine {
    /// Constructor: Called by the Android/iOS app on startup.
    #[uniffi::constructor]
    pub fn new() -> Self {
        // Initialize logging (Android Logcat / iOS OSLog)
        #[cfg(target_os = "android")]
        android_logger::init_once(
            android_logger::Config::default().with_min_level(log::Level::Info),
        );
        
        Self {
            inner_state: Arc::new(Mutex::new(InnerState {
                is_running: false,
            })),
        }
    }

    /// Starts the audio processing loop and the physics engine.
    pub fn start_engine(&self) -> Result<(), GoeckohError> {
        let mut state = self.inner_state.lock().unwrap();
        
        if state.is_running {
            return Err(GoeckohError::AlreadyRunning);
        }

        // TODO: Here is where we will spin up the CPAL audio stream
        // and the RingBuf consumer.
        log::info!("Goeckoh Engine Starting: Initializing Crystalline Heart...");
        
        state.is_running = true;
        Ok(())
    }

    /// Stops the engine and releases audio resources.
    pub fn stop_engine(&self) {
        let mut state = self.inner_state.lock().unwrap();
        if state.is_running {
            log::info!("Goeckoh Engine Stopping...");
            state.is_running = false;
            // TODO: Drop the CPAL stream here to stop audio
        }
    }

    /// Called by the UI loop (e.g., every 60fps) to visualize the heart.
    /// Returns a snapshot of the current physics state.
    pub fn get_current_state(&self) -> EmotionalState {
        // In the real implementation, this will read from the atomic
        // state of the ODE solver. For now, we return dummy data.
        EmotionalState {
            valence: 0.75,
            arousal: 0.45,
            coherence: 0.98,
            is_stable: true,
        }
    }
}
