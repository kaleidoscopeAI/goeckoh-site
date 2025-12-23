  mandate is to function as a private, 100% offline, and low-latency Corollary Discharge Proxy. Its primary therapeutic function
  is to address Auditory Prediction Error (APE) by capturing a user's speech, correcting it to a coherent first-person
  narrative, and echoing it back in the user's own cloned voice with their original prosodic characteristics intact.

  The following architecture finalizes the primary "voice-to-clone" pipeline as a self-contained, high-performance loop. It also
  details the integration points for essential secondary systems, such as the emotional core and visualization engine, ensuring
  they support the primary pipeline without introducing dependencies that would compromise its integrity or performance.

  2. Core Architectural Principles

   * Hybrid Gated Dual-Core Architecture: The system is bifurcated into a Fast Path for real-time therapeutic mirroring and a
     Slow Path for deeper cognitive reasoning. This separation ensures that the critical low-latency feedback loop is never
     blocked by more intensive computation.
   * GCL (Global Coherence Level) Gating: The Crystalline Heart emotional model produces a real-time GCL score, which acts as a
     system-wide "conscience." This metric is used to gate the behavior of various components, such as disabling linguistic
     corrections during periods of high user stress to provide a "safety mode."
   * Offline and Private by Design: All components in the finalized pipeline, from VAD and STT to TTS and analysis, are designed
     to run entirely on-device, ensuring no personal data is ever sent to the cloud.
   * Modular and Component-Based: Each step in the pipeline is handled by a distinct, specialized script or module, allowing for
     clear separation of concerns and future maintainability.

  3. Primary Pipeline: The Low-Latency Voice-to-Clone Loop

  This sequence represents the finalized, non-negotiable data flow for real-time operation.

  Step 1: Audio Capture & Utterance Detection
   * Objective: To listen continuously to the microphone and isolate complete spoken utterances, even with long,
     autism-characteristic pauses.
   * Key Scripts: src/audio_manager.py, src/audio_desktop.py, goeckoh_cloner/goeckoh_loop.py (VADUtteranceDetector).
   * Process: The platform-specific AudioDriver captures raw PCM audio chunks and places them in a queue. The
     VADUtteranceDetector consumes these chunks, using webrtcvad to identify speech segments. It intelligently handles silence,
     respecting a 1.2-second pause threshold before segmenting an utterance, ensuring fragmented speech is captured as a whole.
   * Data Flow:
       * Input: Live microphone audio stream.
       * Output: A single audio buffer (np.ndarray) containing one complete utterance. This buffer is passed to two subsequent
         steps in parallel: STT and Prosody Extraction.

  Step 2: Speech-to-Text (STT)
   * Objective: To transcribe the captured audio utterance into a raw text string with minimal latency.
   * Key Script: src/neuro_backend.py (utilizing the sherpa-onnx library).
   * Process: The utterance audio buffer is fed into the sherpa-onnx OnlineRecognizer. This model is highly optimized for
     streaming, on-device ASR, making it ideal for this real-time pipeline.
   * Data Flow:
       * Input: Utterance Audio (np.ndarray).
       * Output: Raw transcribed text (str).

  Step 3: Linguistic Correction & First-Person Mirroring
   * Objective: To transform the raw, potentially fragmented ASR output into a coherent, first-person sentence, while remaining
     sensitive to the user's emotional state.
   * Key Scripts: goeckoh_cloner/correction_engine.py and src/grammar.py.
   * Process: The correct_text function receives the raw text and the current GCL from the Crystalline Heart. If the GCL is
     above a safety threshold (e.g., > 0.35), it applies a series of regex-based rules to normalize whitespace, flip pronouns
     ("you" → "I"), and fix common grammatical errors. If the GCL is low, it enters "safety mode" and returns the raw text
     unmodified.
   * Data Flow:
       * Input: Raw Text (str) and Global Coherence Level (float).
       * Output: Corrected, first-person text (str).

  Step 4: Voice Synthesis (TTS with Cloning)
   * Objective: To synthesize the corrected text into speech using the child's unique cloned voice.
   * Key Scripts: tts.py (using Coqui TTS) or goeckoh/voice/rtvc_engine.py.
   * Process: The TTS engine receives the corrected text. It uses the pre-enrolled speaker_wav and speaker embedding (from the
     one-time enrollment process) to generate an audio waveform that has the child's vocal identity and timbre.
   * Data Flow:
       * Input: Corrected Text (str) and the child's Voice Profile assets.
       * Output: Synthesized, cloned audio (np.ndarray).

  Step 5: Prosody Transfer (Applying Speech Characteristics)
   * Objective: To ensure the cloned speech retains the rhythm, pitch contour, and energy of the original utterance, making the
     echo feel truly like the user's own.
   * Key Script: goeckoh/voice/prosody.py.
   * Process: This crucial step uses librosa to:
       1. Extract ProsodyFeatures (pitch, energy, timing) from the original utterance audio captured in Step 1.
       2. Modify the synthesized audio from Step 4, warping its pitch and tempo to precisely match the prosodic characteristics
          of the original speech.
   * Data Flow:
       * Input: Synthesized Audio (np.ndarray) and ProsodyFeatures from the original utterance.
       * Output: The final, prosody-matched, cloned audio (np.ndarray).

  Step 6: Audio Playback
   * Objective: To play the final audio back to the user with minimal delay.
   * Key Script: src/audio_manager.py (and its platform-specific backends).
   * Process: The final audio buffer is placed into a dedicated playback queue. A background thread reads from this queue and
     plays the sound through the speakers via sounddevice. This non-blocking architecture ensures the main loop can immediately
     begin processing the next utterance.
   * Data Flow:
       * Input: Final Audio (np.ndarray).
       * Output: Sound played to the user.

  ---

  4. Supporting Systems Integration

  The following systems are complete and integrate with the primary pipeline in a well-defined, non-blocking manner.

  A. The Crystalline Heart (Emotional Core)
   * Purpose: Provides real-time emotional state tracking (GCL, Stress).
   * Integration Point: The rms energy of each raw audio chunk from Step 1 is fed into the heart.pulse() method. The resulting
     gcl is then passed to the Correction module in Step 3 to gate its behavior. This allows the system to react emotionally
     without adding latency to the core pipeline.

  B. The Voice Bubble (Real-Time Visualization)
   * Purpose: To provide a synchronized, visual representation of the synthesized voice.
   * Integration Point: This system runs in a parallel thread. The corrected text from Step 3 and the user's VoiceFingerprint
     are fed into the bubble_synthesizer.py. This generates control curves that are broadcast via WebSocket by the
     visual_engine_bridge.py to a web-based bubble_viewer.html for 3D rendering. This process does not block the audio playback.

  C. Voice Enrollment (One-Time Setup)
   * Purpose: To create the unique voice profile required for cloning and synthesis.
   * Integration Point: This is a prerequisite, offline process. The goeckoh_cloner/voice_logger.py script is run once per user
     to generate the necessary assets: a VoiceFingerprint (JSON), a reference .wav file, and a speaker embedding .npy file.
     These files are then loaded by the TTS (Step 4) and Prosody Transfer (Step 5) modules during the real-time loop.

  ---
  This finalized architecture meets the core requirement of a self-contained, low-latency voice-to-clone pipeline while
  integrating the project's unique emotional and visual feedback systems in a robust, parallel, and non-blocking manner.
