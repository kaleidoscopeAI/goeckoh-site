def __init__(self):
    self.whisper = WhisperModel("tiny", device="cpu", compute_type="int8")
    self.tool = language_tool_python.LanguageTool('en-US')

def transcribe(self, audio_path: str) -> str:
    segments, _ = self.whisper.transcribe(audio_path, beam_size=5, language="en", vad_filter=True)
    return " ".join(seg.text for seg in segments).strip()

def correct(self, text: str) -> str:
    matches = self.tool.check(text)
    corrected = language_tool_python.utils.correct(text, matches)
    # Hard first-person lock
    replacements = [
        ("you are", "I am"), ("your", "my"), ("you ", "I "), ("You are", "I am"), ("Your", "My")
    ]
    for a, b in replacements:
        corrected = corrected.replace(a, b)
    return corrected

