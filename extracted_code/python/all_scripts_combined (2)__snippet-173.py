config: CompanionConfig

def __post_init__(self) -> None:
    self.audio_io = AudioIO(self.config.audio)
    self.processor = SpeechProcessor(self.config.speech)
    self.data = DataStore(self.config)
    self.similarity = SimilarityScorer(self.config.audio)
    self.behavior = BehaviorMonitor()
    self.advisor = StrategyAdvisor()

    self.voice_tts = VoiceMimic(self.config.speech)
    profile_dir = self.config.paths.voices_dir / "voice_profile"
    self.voice_profile = VoiceProfile(audio=self.config.audio, base_dir=profile_dir)
    self.voice_crystal = VoiceCrystal(
        tts=self.voice_tts,
        audio=self.config.audio,
        profile=self.voice_profile,
        config=VoiceCrystalConfig(sample_rate=self.config.audio.sample_rate),
    )
    self.inner_voice = InnerVoiceEngine(
        voice=self.voice_crystal,
        data_store=self.data,
        config=InnerVoiceConfig(),
    )
    self.coach = GuidanceCoach(self.voice_crystal, self.audio_io, self.data)

    self.phrases = {p.phrase_id: p for p in self.data.list_phrases()}

def record_phrase(self, text: str, seconds: float) -> Phrase:
    pid = f"phrase_{int(time.time())}"
    audio = self.audio_io.record_phrase(seconds)
    filepath = self.config.paths.voices_dir / f"{pid}_{int(time.time())}.wav"
    self.audio_io.save_wav(audio, filepath)
    phrase = Phrase(
        phrase_id=pid,
        text=text,
        audio_file=filepath,
        duration=seconds,
        normalized_text=normalize_simple(text),
    )
    self.data.save_phrase(pid, text, filepath, seconds)
    self.phrases[pid] = phrase
    return phrase

async def handle_chunk(self, chunk: np.ndarray) -> None:
    if chunk.size == 0:
        return
    rms = self.audio_io.rms(chunk)
    if rms < self.config.audio.silence_rms_threshold:
        return

    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    tmp = Path(tmp_path)
    raw, corrected = await self.processor.process(chunk, tmp, self.config.audio.sample_rate)
    tmp.unlink(missing_ok=True)

    normalized_attempt = normalize_simple(corrected or raw or "")
    best: Optional[Phrase] = None
    best_text_score = 0.0
    for phrase in self.phrases.values():
        score = text_similarity(normalized_attempt, phrase.normalized_text)
        if score > best_text_score:
            best = phrase
            best_text_score = score

    attempt_path = self.config.paths.voices_dir / f"attempt_{int(time.time())}.wav"
    self.audio_io.save_wav(chunk, attempt_path)

    audio_score = 0.0
    if best and best_text_score >= 0.4:
        try:
            audio_score = self.similarity.compare(best.audio_file, attempt_path)
        except Exception:
            audio_score = 0.0

    needs_correction = audio_score < 0.85

    if needs_correction and self.config.behavior.correction_echo_enabled:
        self.inner_voice.speak_corrected(
            corrected_text=corrected,
            raw_text=raw,
            prosody_source_wav=chunk,
            prosody_source_sr=self.config.audio.sample_rate,
        )

    if best and not needs_correction and audio_score >= 0.85:
        style = "calm" if rms < self.config.audio.silence_rms_threshold * 2 else "neutral"
        self.voice_profile.maybe_adapt_from_attempt(
            attempt_wav=chunk,
            style=style,
            quality_score=audio_score,
        )

    self.data.log_attempt(
        phrase_id=best.phrase_id if best else None,
        phrase_text=best.text if best else None,
        attempt_audio=attempt_path,
        stt_text=raw,
        corrected_text=corrected,
        similarity=audio_score,
        needs_correction=needs_correction,
    )

    event = self.behavior.register(
        normalized_text=normalized_attempt,
        needs_correction=needs_correction,
        rms=rms,
    )
    if event:
        suggestions = self.advisor.suggest(event)
        if suggestions:
            print(f"[EVENT] {event} detected")
            for s in suggestions[:3]:
                print(f" - {s.title}: {s.description}")

async def run(self) -> None:
    stream = self.audio_io.microphone_stream()
    chunk_generator = chunked_audio(
        stream,
        self.config.audio.chunk_seconds,
        self.config.audio.sample_rate,
    )
    async for chunk in _async_iter(chunk_generator):
        await self.handle_chunk(chunk)


