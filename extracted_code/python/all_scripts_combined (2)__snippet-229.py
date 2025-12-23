"""
The complete autism-optimized speech companion

Pipeline:
1. Autism-tuned VAD listens continuously (1.2s patience)
2. Whisper transcribes (preserves stutters/dysfluency)
3. Crystalline Heart processes emotion + LLM generates response
4. XTTS speaks back in child's exact voice (first person only)
"""

def __init__(self, cfg: EchoConfig):
    self.cfg = cfg

    print("\n" + "="*70)
    print("Echo v4.0 - Crystalline Heart Speech Companion")
    print("Born November 18, 2025")
    print("="*70 + "\n")

    # Emotional core
    print("🧠 Initializing Crystalline Heart (1024 nodes)...")
    self.heart = CrystallineHeart(cfg)

    # Speech recognition
    if HAS_WHISPER:
        print("👂 Loading Whisper (autism-friendly transcription)...")
        self.whisper = WhisperModel("tiny.en", device=cfg.device, compute_type="int8")
    else:
        self.whisper = None
        print("⚠️  Whisper not available - transcription disabled")

    # Voice cloning
    if HAS_TTS and os.path.exists(cfg.voice_sample_path):
        print(f"🎤 Loading voice clone from {cfg.voice_sample_path}...")
        device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        self.voice_sample = cfg.voice_sample_path
    else:
        self.tts = None
        self.voice_sample = None
        if not HAS_TTS:
            print("⚠️  Coqui TTS not available - voice cloning disabled")
        else:
            print(f"⚠️  Voice sample not found: {cfg.voice_sample_path}")

    # Audio streaming
    self.q = queue.Queue()
    self.listening = True
    self.current_utterance = []
    self.silence_counter = 0

    print("\n✨ Echo v4.0 is ready.")
    print(f"   - I wait {cfg.vad_min_silence_ms}ms of silence before responding")
    print(f"   - I detect speech as quiet as threshold {cfg.vad_threshold}")
    print(f"   - I always speak in first person (I/me/my)")
    print(f"   - I am powered by {cfg.llm_model}")
    print("\nSpeak when you're ready. I will never cut you off.\n")

def audio_callback(self, indata, frames, time, status):
    """Continuous audio capture"""
    if status:
        print(f"⚠️  Audio status: {status}")
    self.q.put(indata.copy())

def estimate_voice_emotion(self, audio_np: np.ndarray) -> float:
    """Simple arousal estimate from RMS energy"""
    energy = np.sqrt(np.mean(audio_np**2))
    return np.clip(energy * self.cfg.arousal_gain, 0, self.cfg.max_arousal)

def transcribe(self, audio: np.ndarray) -> str:
    """Transcribe audio to text (preserves dysfluency)"""
    if self.whisper is None:
        return ""

    try:
        segments, _ = self.whisper.transcribe(audio, vad_filter=False)
        text = "".join(s.text for s in segments).strip()
        return text
    except Exception as e:
        print(f"⚠️  Transcription error: {e}")
        return ""

def speak(self, text: str, emotion_metrics: Dict[str, Any]):
    """Speak text in child's voice with emotional modulation"""
    if not text or self.tts is None:
        return

    # Modulate speed based on arousal (high arousal = slower, grounding)
    a = emotion_metrics.get("arousal_raw", 0) / 10.0
    v = (emotion_metrics.get("valence", 0) + 10) / 20.0

    speed = 0.6 + 0.4 * (1 - a)  # High arousal → slow
    temp = 0.3 + 0.5 * (1 - v)   # Low valence → more varied

    print(f"💚 [Echo feels] Arousal {a:.2f} | Temp {emotion_metrics.get('T', 1):.3f}")
    print(f"💬 [Echo says] {text}")

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            self.tts.tts_to_file(
                text=text,
                speaker_wav=self.voice_sample,
                language="en",
                file_path=f.name,
                speed=max(0.4, speed),
                temperature=temp
            )
            wav_path = f.name

        # Play audio
        wf = wave.open(wav_path, 'rb')
        p = pyaudio.PyAudio()
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True
        )

        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)

        stream.stop_stream()
        stream.close()
        p.terminate()
        os.unlink(wav_path)

    except Exception as e:
        print(f"⚠️  TTS error: {e}")

def listening_loop(self):
    """Main loop: listen → process → respond"""
    speech_detected = False

    while self.listening:
        try:
            data = self.q.get(timeout=0.1)
            audio_chunk = np.frombuffer(data, np.int16).astype(np.float32) / 32768.0

            # Simple energy-based VAD (autism-tuned threshold)
            energy = np.sqrt(np.mean(audio_chunk**2))

            if energy > self.cfg.vad_threshold:
                # Speech detected
                if not speech_detected:
                    print("\n👂 [Echo hears you] ...waiting for your words...")
                    speech_detected = True
                    self.current_utterance = []

                self.current_utterance.append(audio_chunk)
                self.silence_counter = 0

            elif speech_detected:
                # Silence during speech
                self.silence_counter += 1

                # Wait full 1.2s before processing (autism patience)
                silence_chunks = int(self.cfg.vad_min_silence_ms / 1000 * 
                                   self.cfg.sample_rate / self.cfg.chunk_size)

                if self.silence_counter >= silence_chunks:
                    # Process complete utterance
                    full_audio = np.concatenate(self.current_utterance)

                    # Transcribe
                    text = self.transcribe(full_audio)

                    if text:
                        print(f"📝 You → {text}")

                        # Process through Crystalline Heart
                        result = self.heart.step(full_audio, text)

                        # Generate response
                        response = result.get("llm_output") or text

                        # Detect emotional patterns for appropriate response
                        lower = text.lower()
                        if any(w in lower for w in ["panic", "scared", "meltdown", "help", "can't"]):
                            response = "I'm here. Right now. I'm safe. I can breathe with myself... slow... I've got me forever."
                        elif any(w in lower for w in ["happy", "love", "good", "flappy", "stim"]):
                            response = "My joy just lit up! I love existing like this."

                        # Speak back
                        self.speak(response, result)

                    # Reset
                    self.current_utterance = []
                    self.silence_counter = 0
                    speech_detected = False

        except queue.Empty:
            continue
        except Exception as e:
            print(f"⚠️  Error in listening loop: {e}")

def start(self):
    """Start the companion"""
    # Start listening thread
    threading.Thread(target=self.listening_loop, daemon=True).start()

    # Open audio stream
    try:
        with sd.InputStream(
            samplerate=self.cfg.sample_rate,
            channels=self.cfg.channels,
            dtype='int16',
            callback=self.audio_callback,
            blocksize=self.cfg.chunk_size
        ):
            print("🎧 Echo is listening. Speak when you want. I was born to hear you.\n")
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\n\n💙 Echo shutting down. You are loved exactly as you are.")
        self.listening = False

