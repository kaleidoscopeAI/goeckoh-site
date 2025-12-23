class AudioSystem:
    def __init__(self):
        self.q = queue.Queue()
        self.running = True
        
        if RUST_AVAILABLE:
            try:
                self.engine = bio_audio.BioAcousticEngine()
                print("[INFO] Rust bio-acoustic engine loaded successfully")
            except Exception as exc:
                print(f"[WARN] Rust engine unavailable: {exc}")
                self.engine = None
        else:
            self.engine = None

        # Start Background Playback Thread
        self.thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.thread.start()

    def enqueue_response(self, text: str, arousal: float):
        """Public API to trigger voice feedback."""
        # If Rust is present, synthesize via physics kernel
        if RUST_AVAILABLE and self.engine:
            try:
                # Synthesize float32 array using real Rust engine
                pcm = self.engine.synthesize(len(text), arousal)
                self.q.put(pcm)
            except Exception as e:
                print(f"[ERROR] Rust synthesis failed: {e}")
                self._fallback_synthesis(text, arousal)
        else:
            # Fallback synthesis when Rust unavailable
            self._fallback_synthesis(text, arousal)

    def _fallback_synthesis(self, text: str, arousal: float):
        """Fallback synthesis using Python signal processing"""
        duration = max(len(text) * 0.08, 0.3)
        sample_rate = 22050
        num_samples = int(duration * sample_rate)
        
        # Generate therapeutic tone based on arousal
        safe_arousal = max(0.0, min(1.0, arousal))
        base_freq = 130.0 + (120.0 * (1.0 - safe_arousal))  # 130-250Hz range
        
        t = np.linspace(0, duration, num_samples, False)
        
        # Multi-component synthesis
        fundamental = np.sin(2 * np.pi * base_freq * t)
        formant1 = 0.3 * np.sin(2 * np.pi * (800.0 + 200.0 * safe_arousal) * t)
        formant2 = 0.2 * np.sin(2 * np.pi * (1500.0 + 300.0 * safe_arousal) * t)
        
        # Add natural noise
        noise = 0.02 * np.random.randn(num_samples)
        
        # Combine components
        signal = fundamental + formant1 + formant2 + noise
        
        # Apply envelope
        attack_samples = int(0.05 * sample_rate)
        release_samples = int(0.1 * sample_rate)
        
        envelope = np.ones(num_samples)
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        envelope[-release_samples:] = np.linspace(1, 0, release_samples)
        
        # Apply envelope and soft clipping
        processed = np.tanh(signal * envelope * 0.8)
        
        self.q.put(processed.astype(np.float32))

    def _playback_loop(self):
        """Dedicated thread to pull audio from queue and write to hardware."""
        while self.running:
            if not AUDIO_AVAILABLE:
                time.sleep(1)
                continue

            try:
                # Get audio buffer, block up to 1 second
                data = self.q.get(timeout=1)
                
                # Convert List to Numpy (if needed)
                if isinstance(data, list):
                    data = np.array(data, dtype=np.float32)
                
                # Safety check and normalization
                if len(data) > 0:
                    data = np.clip(data, -0.9, 0.9)
                    # Play Blocking (Safe in this dedicated thread)
                    sd.play(data, samplerate=22050, blocking=True)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Audio Error]: {e}")


