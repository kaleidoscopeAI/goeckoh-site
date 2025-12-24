# Project Progress and To-Do List for Echo V4.0 Unified System

This document summarizes the current state of the Echo V4.0 project and outlines the remaining tasks based on our agreed-upon roadmap and the "ARCHITECTURAL AND NEUROBIOLOGICAL BLUEPRINT FOR ECHO V4.0: THE FIRST-PERSON COGNITIVE MIRROR".

## Current Progress (Completed Tasks)

All core modules have been implemented and integrated into a unified, in-process Python application with a KivyMD GUI.

*   **Repository Structure:** Established `ECHO_V4_UNIFIED` as the project root with the following subdirectories: `echo_core`, `crystal_brain`, `avatar`, `gui`, `config`, `data`, `scripts`, `installers`, `tests`.
*   **Core Configuration:** `config.py` centralizes all tunables and paths.
*   **Shared Event Types:** `events.py` defines all data transfer objects (`EchoEvent`, `HeartMetrics`, `BrainMetrics`, `AvatarFrame`, `SystemSnapshot`).
*   **Crystal Brain Module (`crystal_brain/`):**
    *   `math.py`: Implemented core energetic formulas (information energy, field stability, Lyapunov loss, coherence, integrated information).
    *   `store.py`: Manages SQLite database with schemas for memories, energetics, voice profiles, audio snippets, and triggers.
    *   `core.py`: Implements `CrystalBrain` for hash-based embeddings, memory storage, annealing, metrics computation, and caption generation.
    *   `api.py`: Initial stub for API boundary (though `SystemState` is the primary in-process communication).
*   **Echo Core Module (`echo_core/`):**
    *   `heart.py`: Full 1024-node ODE Crystalline Heart simulation for continuous affective state modeling.
    *   `text_normalizer.py`: Grammar correction and rigorous first-person pronoun rewriting.
    *   `speech_io.py`: Audio input stream management using `sounddevice` and Silero VAD.
    *   `speech_loop.py`: The central orchestrator, integrating VAD, ASR (`faster-whisper`), TTS (`TTS` for voice cloning), Heart, Brain, Avatar, Trigger Engine, Voice Timbre, and updating the `SystemState`.
*   **Avatar Module (`avatar/`):**
    *   `types.py`: Defines `AvatarFrame` data structure and default frame generation.
    *   `controller.py`: Maps Heart and Brain metrics to avatar visual properties (positions, colors, sizes).
*   **Trigger Engine & Voice Timbre:**
    *   `audio_features.py`: Computes MFCC-based audio embeddings for non-speech sounds.
    *   `voice_timbre.py`: Manages `jackson_auto_timbre.wav` for Phonemic/Prosodic Patching (PPP), allowing the TTS to adapt to any vocalizations.
    *   `trigger_engine.py`: Implements the "Guardian Lexicon" for mapping non-verbal audio patterns to first-person phrases via guardian-labeled triggers.
*   **Shared System State:** `system_state.py` provides an in-memory, thread-safe `SystemSnapshot` for real-time GUI updates.
*   **GUI (KivyMD - `gui/`):**
    *   `gui/avatar_widget.py`: Renders the 2D projection of the Crystal Avatar.
    *   `gui/app.py`: Main KivyMD application with `ChildScreen` (Jackson view), `ParentScreen` (Molly Dashboard), and `VoiceSetupScreen` (for recording and testing voice samples, mic check, volume tuning).
*   **Main Entrypoint:** `main.py` at the project root to launch the `EchoGuiApp`.
*   **Dependencies:** `requirements.txt` has been updated to reflect the actual, in-process, Python 3.11 compatible dependencies.
*   **Unit Tests:** Initial unit tests have been created for `echo_core/text_normalizer.py`, `crystal_brain/math.py`, and `echo_core/heart.py`.
*   **Environment Setup Instructions:** Detailed instructions provided for setting up a Python 3.11 virtual environment and running the application.

## Remaining Roadmap & To-Do List

This section outlines the remaining tasks to fully realize the Echo V4.0 system, following the original roadmap and the neurobiological blueprint.

### 7. Parent GUI (Molly dashboard)
*   [ ] 7.2 Implement Voice Profile Growth Chart: While backend data logging (`store.py`) is in place, the GUI integration for proper charting (e.g., using Kivy's `garden.graph` or similar) is pending.

### 8. Data, logging, and privacy
*   [ ] 8.1 Central log policy: Implement explicit logging to files (`data/logs/echo.log`, `brain.log`, etc.) with log rotation and size caps, replacing current `print` statements.
*   [ ] 8.3 Configurable retention: Add configuration options for data retention periods for events and metrics.

### 9. Packaging and installers
*   [ ] 9.2 Desktop packaging: Create `scripts/build_desktop.py` using `PyInstaller` (or equivalent) to generate platform-specific executables (Windows/macOS/Linux).
*   [ ] 9.3 Mobile packaging: Use `Buildozer` to package for Android (and potentially iOS, with considerations for mobile-optimized models).

### 10. Testing and verification
*   [ ] 10.2 Integration tests: Develop tests that simulate the full audio processing pipeline, verifying end-to-end functionality and latency.
*   [ ] 10.3 Real-device sanity tests: Conduct tests on target hardware (desktop, Android phone) to confirm app launch, offline operation, and functionality.

### New Strategic To-Do Items (from Blueprint & Discussions)

*   **[IN PROGRESS] DRC Gating Implementation:** The initial implementation of the `DRCRequestGate` is complete within the `SpeechLoop`. It now actively suppresses the standard TTS echo and plays a calming script during "RED" states. Further refinement of gated actions is pending.
*   **[IN-PROGRESS] Phenotyping & Annotation:** The `phenotyping_tracker.py` module has been created and fully integrated into the `SpeechLoop` to classify and log every vocal fragment. The GUI for caregiver annotation is pending.
*   [ ] **Autopoietic Adaptation Refinements:**
    *   [ ] Implement explicit "Voice Congruence Metrics" tracking (cosine similarity drift for PPP blend) to trigger AIM updates.
    *   [ ] Implement full Hamiltonian (H) modeling for the Crystalline Heart lattice tension and energy balancing.
*   [ ] **GCL-Derived Adaptive Phrasing:** Explore how the Inner Voice Engine can select tone, pacing, or prosody dynamically based on the current GCL state for more nuanced affective regulation.

---
**Next Steps for User:**

Please ensure your environment is set up with **Python 3.11** as per the previous instructions, and then attempt to install the dependencies and run the application. Once you confirm it launches, we can proceed with further development based on the remaining roadmap.
