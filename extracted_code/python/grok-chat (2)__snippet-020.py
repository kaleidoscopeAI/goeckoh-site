def __init__(self, model_name: str = \"base.en\", language_tool_url: str = \"http://localhost:8081\") -> None:
    self.model = whisper.load_model(model_name)
    self.tool = LanguageTool('en-US', remote_server=language_tool_url)

async def process(self, audio: np.ndarray, tmp_path: Path, sample_rate: int) -> tuple[str, str]:
    import asyncio
    import soundfile as sf

    sf.write(tmp_path, audio, sample_rate)
    result = self.model.transcribe(str(tmp_path))
    raw = result[\"text\"]
    corrected = self.tool.correct(raw)
    return raw, corrected
