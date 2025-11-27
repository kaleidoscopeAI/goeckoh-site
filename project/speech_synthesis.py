# speech_synthesis.py

from gtts import gTTS
from typing import Dict

class SpeechSynthesizer:
    """Handles the conversion of textual knowledge into speech"""

    def __init__(self):
        self.voice_cache: Dict[str, str] = {}  # Cache for synthesized speech files

    def synthesize(self, text: str, language: str = "en") -> str:
        """
        Converts text to speech and saves the audio to a file.
        Returns the path to the audio file.
        """
        if text in self.voice_cache:
            return self.voice_cache[text]

        try:
            tts = gTTS(text=text, lang=language)
            file_path = f"speech_{hash(text)}.mp3"
            tts.save(file_path)
            self.voice_cache[text] = file_path
            return file_path
        except Exception as e:
            raise RuntimeError(f"Speech synthesis failed: {e}")

    def clear_cache(self):
        """Clears the voice cache"""
        self.voice_cache.clear()
