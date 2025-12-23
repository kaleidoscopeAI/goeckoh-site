settings: SystemSettings = field(default_factory=load_settings)
_q: asyncio.Queue[np.ndarray] = field(init=False)
_current_utterance: list[np.ndarray] = field(default_factory=list)
_last_speech_end_time: float = field(default_factory=float)

def __post_init__(self) -> None:
    self.audio_io = AudioIO(self.settings.audio)
    self.vad = VAD(self.settings.audio)
    self.stt_engine = STT(self.settings.speech)
    self.text_processor = TextProcessor(self.settings.speech)
    self.heart = EchoCrystallineHeart(self.settings.heart)
    self.llm = LocalLLM(self.settings.heart)
    self.metrics_logger = MetricsLogger(self.settings.paths.metrics_csv)
    self.guidance_logger = GuidanceLogger(self.settings.paths.guidance_csv)
    self.behavior_monitor = BehaviorMonitor(self.settings)

    self.voice_mimic = VoiceMimic(self.settings.speech)
    self.voice_profile = VoiceProfile(
        audio_cfg=self.settings.audio, paths=self.settings.paths
    )
    self.expression_gear = ExpressionGear(
        tts_engine=self.voice_mimic,
        audio_cfg=self.settings.audio,
        voice_profile=self.voice_profile,
    )
    self.aba_engine = AbaEngine(
        settings=self.settings,
        voice_profile=self.voice_profile,
        guidance_logger=self.guidance_logger,
    )
    self._q = asyncio.Queue()

async def run(self) -> None:
    self.audio_io.listen()
    print("[SpeechLoop] Listening for speech...")

    try:
        while True:
            audio_chunk = await self._q.get()
            vad_result = self.vad.process(audio_chunk)

            if vad_result and "start" in vad_result:
                self._current_utterance = [audio_chunk]
                self.vad.reset() # Reset VAD state for next utterance
            elif vad_result and "end" in vad_result:
                if self._current_utterance:
                    self._current_utterance.append(audio_chunk)
                    full_utterance_audio = np.concatenate(self._current_utterance)
                    self._current_utterance = []
                    await self._handle_utterance(full_utterance_audio)
            elif self._current_utterance:
                self._current_utterance.append(audio_chunk)
    except asyncio.CancelledError:
        print("[SpeechLoop] Shutting down...")
    finally:
        self.audio_io.stop()

async def _handle_utterance(self, audio_chunk: np.ndarray) -> None:
    if audio_chunk.size == 0:
        return

    rms = np.sqrt(np.mean(audio_chunk**2))
    if rms < self.settings.audio.silence_rms_threshold:
        return

    print(f"[SpeechLoop] Processing utterance (RMS: {rms:.4f})...")

    attempt_path = self._persist_attempt(audio_chunk)

    # STT and Text Processing
    raw_text = self.stt_engine.transcribe(audio_chunk)
    corrected_text = self.text_processor.normalize(raw_text)
    normalized_text = LocalLLM.enforce_first_person(corrected_text or raw_text or "")
    needs_correction = (corrected_text or "").strip() != (raw_text or "").strip()

    print(f"[SpeechLoop] Raw: '{raw_text}' | Corrected: '{corrected_text}' | Normalized: '{normalized_text}'")

    # Heart and LLM Integration
    heart_result = self.heart.step(audio_chunk)
    llm_output = None
    if self.settings.heart.use_llm and normalized_text.strip():
        arousal_mean = float(heart_result["emotions"].mean(dim=0)[0].item())
        valence_mean = float(heart_result["emotions"].mean(dim=0)[1].item())

        prompt = self.heart._build_prompt( # Accessing protected member for now, consider refactor
            transcript=normalized_text,
            arousal=arousal_mean,
            valence=valence_mean,
            T_val=heart_result["T"],
            coherence=heart_result["coherence"],
        )
        llm_output = self.llm.generate(
            prompt=prompt,
            temperature=heart_result["T"] * self.settings.heart.llm_temperature_scale,
            top_p=self.settings.heart.llm_top_p_base + self.settings.heart.llm_top_p_spread * (1.0 - heart_result["coherence"]),
        )
        # Update heart with LLM embedding
        emb = self.llm.embed(llm_output)
        emb_t = torch.from_numpy(emb).to(self.heart.device, dtype=torch.float32)
        if self.settings.heart.num_nodes <= self.settings.heart.embedding_dim:
            proj = emb_t[: self.settings.heart.num_nodes]
        else:
            reps = math.ceil(self.settings.heart.num_nodes / self.settings.heart.embedding_dim)
            tiled = emb_t.repeat(reps)
            proj = tiled[: self.settings.heart.num_nodes]
        proj = proj.view(self.settings.heart.num_nodes, 1)
        ch = self.settings.heart.embedding_channel
        if 0 <= ch < self.settings.heart.num_channels:
            self.heart.emotions[:, ch: ch + 1].add_(self.settings.heart.embedding_gain * proj)
            self.heart.emotions.clamp_(-self.settings.heart.max_abs, self.settings.heart.max_abs)


    # ABA Engine Integration
    event = self.behavior_monitor.register(normalized_text=normalized_text, needs_correction=needs_correction, rms=rms)

    if not needs_correction:
        self.aba_engine.reinforce_success(normalized_text)
        self.aba_engine.track_skill_progress(normalized_text, success=True)
    else:
        self.aba_engine.track_skill_progress(normalized_text, success=False)

    aba_event_response = None
    if event:
        aba_event_response = self.aba_engine.intervene(event, normalized_text)
        if aba_event_response and aba_event_response.category == "inner_echo":
            llm_output = aba_event_response.message # Override LLM output with ABA suggestion

        print(f"[SpeechLoop] Behavior Event: {event} -> ABA Response: {llm_output or 'None'}")


    # Decide what to speak
    text_to_speak = ""
    if llm_output:
        text_to_speak = llm_output
    elif needs_correction and self.settings.behavior.correction_echo_enabled:
        text_to_speak = normalized_text

    if text_to_speak:
        self._speak_inner(text_to_speak)

    # Logging
    record = AttemptRecord(
        timestamp=datetime.utcnow(),
        phrase_text=normalized_text,
        raw_text=raw_text,
        corrected_text=corrected_text,
        needs_correction=needs_correction,
        audio_file=attempt_path,
        similarity=1.0 if not needs_correction else 0.0, # Placeholder, will be updated with actual similarity score
    )
    self.metrics_logger.append(record)

    if event:
        # Log the original behavior event that triggered ABA
        self.guidance_logger.append(
            BehaviorEvent(
                timestamp=datetime.utcnow(),
                level="warning" if event != "encouragement" else "info",
                category=event,  # type: ignore[arg-type]
                title=f"{event.replace('_', ' ').title()} detected",
                message=text_to_speak, # The response from the system
            )
        )

def _persist_attempt(self, audio_chunk: np.ndarray) -> Path:
    attempt_dir = self.settings.paths.attempts_dir
    attempt_dir.mkdir(parents=True, exist_ok=True)
    # Use a more robust filename to avoid conflicts and ensure uniqueness
    filename = f"attempt_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.wav"
    path = attempt_dir / filename
    # Ensure the audio chunk is in the correct format (e.g., int16) before saving
    # Assuming audio_chunk is float32 from sounddevice, convert to int16
    audio_int16 = (audio_chunk * 32767).astype(np.int16)
    self.audio_io.save_wav(audio_int16, path)
    return path

def _speak_inner(self, text: str) -> None:
    decision = AgentDecision(target_text=text, mode="inner")
    info = self.expression_gear.express(decision, None) # No audio_info for now, needs to be from child's speech
    if info and isinstance(info.payload, AudioData):
        self.audio_io.play(info.payload.waveform, self.settings.audio.sample_rate)


