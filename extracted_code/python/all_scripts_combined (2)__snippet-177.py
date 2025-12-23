"""Delivers friendly prompts via voice and logs the event."""

def __init__(self, voice: VoiceCrystal, audio_io: AudioIO, data_store: DataStore) -> None:
    self.voice = voice
    self.audio_io = audio_io
    self.data_store = data_store

def speak(self, event: str, override_text: str | None = None) -> None:
    script = GUIDANCE_SCRIPTS.get(event)
    if not script and not override_text:
        return

    text = override_text or self._build_message(script)
    style = script.style if script else "neutral"
    audio = self.voice.speak(text, style=style, mode="outer")
    if audio.size > 0:
        self.audio_io.play(audio)
    self.data_store.log_guidance_event(event, (script.title if script else "Custom"), text)


def _build_message(self, script: GuidanceScript | None) -> str:
    if not script:
        return ""
    parts = [script.message]
    if script.breathing_steps:
        parts.extend(script.breathing_steps)
    return " ".join(parts)
