"""High-level gear for generating expressive speech."""

tts_engine: VoiceMimic
audio_cfg: AudioSettings
voice_profile: VoiceProfile
inner_volume_scale: float = 0.5
coach_volume_scale: float = 1.1

def _apply_mode_acoustics(self, wav: np.ndarray, mode: Mode) -> np.ndarray:
    """Applies acoustic effects based on the speech mode."""
    if mode == "inner":
        return (wav * self.inner_volume_scale).astype(np.float32)
    elif mode == "coach":
        return (wav * self.coach_volume_scale).astype(np.float32)
    return wav

def express(
    self, decision: AgentDecision, audio_info: Optional[Information]
) -> Optional[Information]:
    """
    Generates speech based on an agent's decision.
    Returns an Information object with the final audio, or None.
    """
    text_to_speak = decision.target_text
    if not text_to_speak:
        return None

    # 1. Select a voice reference for cloning
    ref_path = self.voice_profile.pick_reference()
    if ref_path:
        self.tts_engine.update_voiceprint(ref_path)

    # 2. Synthesize the base waveform
    base_wav = self.tts_engine.synthesize(text_to_speak)
    if base_wav.size == 0:
        return None

    # 3. Apply prosody transfer if a source is available
    final_wav = base_wav
    if audio_info and isinstance(audio_info.payload, AudioData):
        prosody_source_wav = audio_info.payload.waveform
        prosody_source_sr = audio_info.payload.sample_rate
        try:
            prosody = extract_prosody(prosody_source_wav, prosody_source_sr)
            final_wav = apply_prosody_to_tts(
                base_wav, self.audio_cfg.sample_rate, prosody
            )
        except Exception as e:
            print(f"Warning: Prosody transfer failed. Using base TTS. Error: {e}")

    # 4. Apply mode-specific acoustics
    final_wav = self._apply_mode_acoustics(final_wav, decision.mode)

    return Information(
        payload=AudioData(
            waveform=final_wav, sample_rate=self.audio_cfg.sample_rate
        ),
        source_gear="ExpressionGear",
        metadata={"decision": decision},
    )
