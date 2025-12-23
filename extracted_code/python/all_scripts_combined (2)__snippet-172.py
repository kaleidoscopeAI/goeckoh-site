"""Wrap whisper and LanguageTool into a streamlined API."""

settings: SpeechModelSettings

def __post_init__(self) -> None:
    self._whisper = whisper.load_model(self.settings.whisper_model)
    self._tool = LanguageTool("en-US", server_url=self.settings.language_tool_server)

async def transcribe(self, audio_path: Path) -> str:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, self._whisper.transcribe, str(audio_path))
    return result.get("text", "").strip()

def normalize(self, text: str) -> str:
    if not text:
        return ""
    return normalize_text(text).normalized_text

def correct_text(self, text: str) -> str:
    if not text:
        return ""
    matches = self._tool.check(text)
    return LanguageTool.correct(text, matches)

async def process(self, audio: np.ndarray, tmp_path: Path, sample_rate: int = 16_000) -> tuple[str, str]:
    """Write temp wav, transcribe, correct, then delete."""
    import soundfile as sf

    sf.write(tmp_path, audio, sample_rate)
    raw = await self.transcribe(tmp_path)
    normalized = self.normalize(raw)
    corrected = self.correct_text(normalized)
    tmp_path.unlink(missing_ok=True)
    return raw, corrected
