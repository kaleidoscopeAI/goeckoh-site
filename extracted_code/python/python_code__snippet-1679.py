class UnifiedAudioSystem:
    """Complete audio system integrating all audio components"""
    
    def __init__(self, clone_ref_wav: Optional[str] = None):
        self.audio_queue = queue.Queue(maxsize=10)
        self.running = True
        self.bio_engine = None
        self.neural_engine = None
        self.rtvc_engine = None
        self.voice_mimic_engine = None
        self.clone_ref_wav = clone_ref_wav
        self.external_audio = None
        
        # Prefer packaged audio system if present (wraps Rust + playback)
        if ExternalAudioSystem is not None:
            try:
                self.external_audio = ExternalAudioSystem()
                print("🔊 External AudioSystem loaded from goeckoh.audio.audio_system")
            except Exception as e:
                print(f"⚠️  External AudioSystem failed: {e}")

        # Initialize Rust bio-acoustic engine
        if RUST_AVAILABLE:
            try:
                if hasattr(bio_audio, "BioAcousticEngine"):
                    self.bio_engine = bio_audio.BioAcousticEngine()
                elif hasattr(bio_audio, "BioEngine"):
                    self.bio_engine = bio_audio.BioEngine()
                else:
                    raise AttributeError("No BioAcousticEngine/BioEngine symbol exposed")
                print("🦀 Rust bio-acoustic engine initialized")
            except Exception as e:
                print(f"⚠️  Rust engine failed: {e}")

        # Initialize RTVC engine (speaker-conditioned cloning)
        use_rtvc = os.getenv("GOECKOH_USE_RTVC", "").lower() in ("1", "true", "yes")
        if RTVC_AVAILABLE and (use_rtvc or (self.clone_ref_wav and os.path.exists(self.clone_ref_wav))):
            try:
                self.rtvc_engine = VoiceEngineRTVC()
                print("🧠 RTVC voice cloning initialized")
            except Exception as e:
                print(f"⚠️  RTVC init failed: {e}")

        # Initialize VoiceMimic (pyttsx3 + optional Coqui XTTS) if available
        if VoiceMimicAdapter is not None:
            try:
                self.voice_mimic_engine = VoiceMimicAdapter(
                    tts_model_name=os.getenv("GOECKOH_TTS_MODEL"),
                    ref_wav=self.clone_ref_wav,
                    sample_rate=16000,
                )
                if self.voice_mimic_engine.available:
                    print("🗣️  VoiceMimic adapter initialized")
            except Exception as e:
                print(f"⚠️  VoiceMimic init failed: {e}")
        
        # Initialize neural TTS engine
        if ExternalVoiceEngine is not None:
            try:
                self.neural_engine = ExternalVoiceEngine()
                print("🧠 Neural TTS engine initialized (package)")
            except Exception as e:
                print(f"⚠️  Neural TTS failed: {e}")
        elif NEURAL_TTS_AVAILABLE:
            try:
                self.neural_engine = VoiceEngineImpl()
                print("🧠 Neural TTS engine initialized (inline)")
            except Exception as e:
                print(f"⚠️  Neural TTS failed: {e}")
        
        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.audio_thread.start()
    
    def _audio_worker(self):
        """Background audio processing thread"""
        while self.running:
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                if AUDIO_AVAILABLE and audio_data is not None:
                    sd.play(audio_data, samplerate=22050)
                    sd.wait()
                elif PLAYSOUND_AVAILABLE and audio_data is not None:
                    self._play_with_playsound(audio_data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio playback error: {e}")
    
    def _play_with_playsound(self, audio_data: np.ndarray):
        """Fallback playback using playsound by writing a temp WAV."""
        if not PLAYSOUND_AVAILABLE or audio_data is None or audio_data.size == 0:
            return
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                path = tmp.name
                # Normalize to int16 for wave module
                scaled = np.clip(audio_data, -1.0, 1.0)
                pcm16 = (scaled * 32767).astype(np.int16)
                with wave.open(tmp, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(22050)
                    wf.writeframes(pcm16.tobytes())
            play_sound(path, block=False)
        except Exception as e:
            print(f"playsound fallback failed: {e}")
    
    def synthesize_response(self, text: str, arousal: float, style: str = "neutral", clone_ref_wav: Optional[str] = None) -> Optional[np.ndarray]:
        """Unified speech synthesis with multiple engines"""
        
        # If the external audio stack can render directly, let it handle synthesis/playback
        if self.external_audio is not None:
            try:
                self.external_audio.enqueue_response(text, arousal)
            except Exception as e:
                print(f"External audio enqueue failed, falling back: {e}")
            # Still compute audio locally for downstream consumers

        # Try RTVC voice cloning first if a reference wav is provided
        if self.rtvc_engine and clone_ref_wav and os.path.exists(clone_ref_wav):
            rtvc_audio = self.rtvc_engine.synthesize(text, clone_ref_wav)
            if rtvc_audio is not None:
                return rtvc_audio

        # Try VoiceMimic (pyttsx3/Coqui hybrid) if available
        if self.voice_mimic_engine and getattr(self.voice_mimic_engine, "available", False):
            vm_audio = self.voice_mimic_engine.synthesize(text, clone_ref_wav)
            if vm_audio is not None and vm_audio.size > 0:
                return vm_audio

        # Try neural TTS (Coqui) next
        if self.neural_engine and NEURAL_TTS_AVAILABLE:
            try:
                neural_audio = self.neural_engine.generate_speech_pcm(text, clone_ref_wav)
                if neural_audio is not None:
                    return neural_audio
            except Exception as e:
                print(f"Neural TTS failed: {e}")
        
        # Try Rust bio-acoustic engine
        if self.bio_engine and RUST_AVAILABLE:
            try:
                rust_audio = self.bio_engine.synthesize(len(text), arousal)
                return np.array(rust_audio, dtype=np.float32)
            except Exception as e:
                print(f"Rust synthesis failed: {e}")
        
        # Fallback to pure Python synthesis
        return self._python_synthesis(text, arousal, style)
    
    def _python_synthesis(self, text: str, arousal: float, style: str) -> np.ndarray:
        """Enhanced Python speech synthesis with formant filtering"""
        duration = 1.0 + len(text) * 0.05
        sample_rate = 22050
        num_samples = int(duration * sample_rate)
        
        # Enhanced style-based parameters with formant frequencies
        style_params = {
            "neutral": {
                "pitch": 120, "energy": 0.5,
                "formants": [500, 1500, 2500],  # F1, F2, F3 for neutral
                "vibrato": 0.02, "breath": 0.1
            },
            "calm": {
                "pitch": 100, "energy": 0.3,
                "formants": [400, 1200, 2000],  # Lower formants for calm
                "vibrato": 0.01, "breath": 0.15
            }, 
            "excited": {
                "pitch": 180, "energy": 0.8,
                "formants": [600, 1800, 3000],  # Higher formants for excited
                "vibrato": 0.05, "breath": 0.05
            }
        }
        
        params = style_params.get(style, style_params["neutral"])
        base_pitch = params["pitch"] * (1.0 + arousal * 0.3)
        energy = params["energy"] * (1.0 + arousal * 0.2)
        
        # Generate time vector
        t = np.linspace(0, duration, num_samples)
        
        # Create complex voice source with harmonics
        fundamental = np.sin(2 * np.pi * base_pitch * t)
        
        # Add harmonics for richness
        harmonics = np.zeros(num_samples)
        for h in range(2, 6):
            harmonic_freq = base_pitch * h
            harmonic_amp = 1.0 / h  # Natural harmonic decay
            harmonics += harmonic_amp * np.sin(2 * np.pi * harmonic_freq * t)
        
        # Add vibrato
        vibrato_mod = 1.0 + params["vibrato"] * np.sin(2 * np.pi * 5 * t)  # 5Hz vibrato
        
        # Combine source signal
        voice_source = (fundamental + harmonics * 0.3) * vibrato_mod * energy
        
        # Apply formant filtering using resonators
        filtered_audio = np.zeros(num_samples)
        for formant_freq in params["formants"]:
            # Create formant resonator (simplified bandpass filter)
            bandwidth = formant_freq * 0.1  # 10% bandwidth
            Q = formant_freq / bandwidth
            
            # Simple resonator implementation
            omega = 2 * np.pi * formant_freq / sample_rate
            alpha = np.exp(-omega / (2 * Q))
            
            # Apply resonator to create formant
            formant_output = np.zeros(num_samples)
            for i in range(1, num_samples):
                formant_output[i] = alpha * formant_output[i-1] + (1 - alpha) * voice_source[i]
            
            filtered_audio += formant_output * (1.0 / len(params["formants"]))
        
        # Add breath noise
        breath_noise = np.random.randn(num_samples) * params["breath"] * 0.01
        
        # Combine all components
        audio = filtered_audio + breath_noise
        
        # Apply natural envelope with attack, sustain, and decay
        attack_time = 0.05  # 50ms attack
        sustain_time = duration * 0.7
        decay_time = duration - attack_time - sustain_time
        
        attack_samples = int(attack_time * sample_rate)
        sustain_samples = int(sustain_time * sample_rate)
        
        envelope = np.ones(num_samples)
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        if sustain_samples + attack_samples < num_samples:
            decay_start = attack_samples + sustain_samples
            envelope[decay_start:] = np.exp(-3 * np.linspace(0, 1, num_samples - decay_start))
        
        audio *= envelope
        
        # Normalize and convert
        audio = np.clip(audio, -1.0, 1.0)
        return audio.astype(np.float32)
    
    def enqueue_audio(self, text: str, arousal: float, style: str = "neutral", clone_ref_wav: Optional[str] = None):
        """Enqueue audio for playback"""
        audio_data = self.synthesize_response(text, arousal, style, clone_ref_wav)
        if audio_data is not None:
            try:
                self.audio_queue.put(audio_data, timeout=0.1)
            except queue.Full:
                pass  # Drop if overloaded

    def set_clone_wav(self, path: Optional[str]):
        """Update clone reference wav and initialize RTVC if needed."""
        self.clone_ref_wav = path
        if path:
            self.clone_ref_wav = os.path.abspath(path)
        # Lazy-init RTVC if now eligible
        if self.rtvc_engine is None and RTVC_AVAILABLE and self.clone_ref_wav and os.path.exists(self.clone_ref_wav):
            try:
                self.rtvc_engine = VoiceEngineRTVC()
                print("🧠 RTVC voice cloning initialized (late)")
            except Exception as e:
                print(f"⚠️  RTVC init failed: {e}")

class VoiceEngineImpl:
    """Neural voice cloning engine"""
    
    def __init__(self, use_gpu=False):
        self.use_neural = False
        self.model = None
        if NEURAL_TTS_AVAILABLE:
            try:
                self.model = TTS("tts_models/en/vctk/vits", gpu=use_gpu)
                self.use_neural = True
            except Exception as e:
                print(f"TTS initialization failed: {e}")
    
    def generate_speech_pcm(self, text: str, clone_ref_wav: str = None) -> Optional[np.ndarray]:
        """Generate speech with neural TTS"""
        if not self.use_neural or self.model is None:
            return None
        
        try:
            if clone_ref_wav and os.path.exists(clone_ref_wav):
                wav = self.model.tts(text=text, speaker_wav=clone_ref_wav, language="en")
                return np.array(wav, dtype=np.float32)
            else:
                wav = self.model.tts(text=text)
                return np.array(wav, dtype=np.float32)
        except Exception as e:
            print(f"Neural TTS generation failed: {e}")
            return None

