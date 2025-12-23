def __init__(self):
    self.q = queue.Queue()
    self.running = True

    # Initialize Physics
    self.bio_engine = bio_audio.BioEngine() if RUST_AVAILABLE else None

    # Initialize AI Voice
    self.neural_engine = neural_speech.VoiceEngine() if NEURAL_AVAILABLE else None

    # Audio Thread
    self.thread = threading.Thread(target=self._playback_loop, daemon=True)
    self.thread.start()

def enqueue_response(self, text: str, arousal: float):
    """
    Master Synthesis Logic:
    1. Attempt High-Res Cloning (Python).
    2. If success -> Apply Rust Physics Modulation (Source-Filter).
    3. If fail -> Generate Bio-Signal Tone directly from Rust.
    """
    pcm_output = None

    # PATH A: Neural Clone + Physics Modulation
    if NEURAL_AVAILABLE and self.neural_engine.use_neural:
        # 1. Generate clean voice (numpy array)
        # In a real deployment, you would point this to 'assets/user_voice.wav'
        raw_clone = self.neural_engine.generate_speech_pcm(text, clone_ref_wav=None)

        if raw_clone is not None and RUST_AVAILABLE:
            # 2. Pass through Crystalline Heart Physics
            # This shakes or dampens the voice based on stress (arousal)
            # We convert numpy -> list -> rust -> list -> numpy for FFI
            processed_data = self.bio_engine.modulate_pcm(raw_clone.tolist(), arousal)
            pcm_output = np.array(processed_data, dtype=np.float32)
        else:
            pcm_output = raw_clone

    # PATH B: Pure Bio-Signal (Safety Tone / Fallback)
    if pcm_output is None and RUST_AVAILABLE:
        # Synthesize pure tone modulated by arousal
        pcm_output = np.array(self.bio_engine.synthesize(len(text), arousal), dtype=np.float32)

    # Enqueue for Hardware
    if pcm_output is not None:
        self.q.put(pcm_output)

def _playback_loop(self):
    """Dedicated Hardware Thread."""
    while self.running:
        if not AUDIO_AVAILABLE:
            time.sleep(1)
            continue

        try:
            data = self.q.get(timeout=1)

            # Robust playback call
            sd.play(data, samplerate=22050, blocking=True)

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Audio Output Failure]: {e}")

