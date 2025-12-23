class InnerVoiceEngine:
    voice: VoiceCrystal
    data_store: DataStore
    config: InnerVoiceConfig = InnerVoiceConfig()
    last_spoke_ts: float = 0.0

    def speak_corrected(
        self,
        corrected_text: Optional[str],
        raw_text: Optional[str],
        prosody_source_wav: Optional[np.ndarray],
        prosody_source_sr: int,
    ) -> None:
        now = time.time()
        if now - self.last_spoke_ts < self.config.min_gap_s:
            return

        phrase = (corrected_text or raw_text or "").strip()
        if not phrase:
            return

        self.voice.speak(
            phrase,
            style="calm",
            mode="inner",
            prosody_source_wav=prosody_source_wav,
            prosody_source_sr=prosody_source_sr,
        )

        self.last_spoke_ts = now

        if self.config.log_events:
            self.data_store.log_guidance_event(
                event="inner_echo",
                title="Inner Echo",
                message=phrase,
            )
