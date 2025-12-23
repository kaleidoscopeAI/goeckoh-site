config: CompanionConfig = CONFIG

def __post_init__(self) -> None:
    self.processor = SpeechProcessor(self.config.speech)
    self.data = DataStore(self.config.paths)
    self.monitor = BehaviorMonitor()
    self.advisor = StrategyAdvisor()

    profile_dir = self.config.paths.voices_dir / "profile"
    self.voice_profile = VoiceProfile(base_dir=profile_dir)
    self.voice_profile.load_existing()

    self.voice_crystal = VoiceCrystal(
        tts=VoiceMimic(self.config.speech),
        profile=self.voice_profile,
        config=VoiceCrystalConfig(sample_rate=self.config.audio.sample_rate),
    )

    self.inner_voice = InnerVoiceEngine(
        voice=self.voice_crystal,
        data_store=self.data,
    )

    self.recognizer = SpeechRecognizer(model_size=self.config.speech.whisper_model)  # New STT
    self.recognizer.start()

async def run(self) -> None:
    print("Starting real-time voice mimicry loop with STT...")
    while True:
        result = self.recognizer.transcribe_chunk()
        if result is None:
            await asyncio.sleep(0.1)
            continue

        raw, audio_chunk = result

        # Process as before
        tmpdir = tempfile.mkdtemp()
        tmp_path = Path(tmpdir) / "attempt.wav"
        sf.write(tmp_path, audio_chunk, self.config.audio.sample_rate)

        corrected = await self.processor.process(audio_chunk, tmp_path, self.config.audio.sample_rate)  # Assuming process returns corrected

        normalized_attempt = raw.strip().lower()

        # ... (phrase matching, scoring, echo logic as before)

        await asyncio.sleep(0.1)  # Throttle

