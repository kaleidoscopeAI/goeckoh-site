def __init__(self):
    self.model = whisper.load_model("tiny")  # runs on CPU
    self.tool = language_tool_python.LanguageTool('en-US')

async def process(self, audio, tmp_path):
    import soundfile as sf
    sf.write(tmp_path, audio, self.cfg.audio.sample_rate)
    result = self.model.transcribe(tmp_path, language="en", vad_filter=True)
    raw = result["text"]
    matches = self.tool.check(raw)
    corrected = self.tool.correct(raw) if matches else raw
    return raw.strip(), corrected.strip()

