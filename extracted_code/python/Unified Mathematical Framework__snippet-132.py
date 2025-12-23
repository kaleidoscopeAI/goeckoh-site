from pathlib import Path
import json
import soundfile as sf
from advanced_voice_mimic import VoiceCrystal
import logging

class CalmingPhrases:
    def __init__(self, voice_crystal: VoiceCrystal, config):
        self.vc = voice_crystal
        self.config = config
        self.phrases_path = config.base_dir / "calming" / "phrases.json"
        self.phrases_path.parent.mkdir(exist_ok=True)
        self.phrases = self.load_phrases()

    def load_phrases(self):
        if self.phrases_path.exists():
            try:
                return json.loads(self.phrases_path.read_text())
            except:
                pass
        return {"meltdown": "Everything is okay. I can close my eyes and breathe. I am safe."}

    def save_phrases(self):
        self.phrases_path.write_text(json.dumps(self.phrases, indent=2))

    def play_calming(self, key="meltdown"):
        text = self.phrases.get(key, "I am safe.")
        self.vc.say(text, style="calm")

    def record_new_phrase(self, key: str, audio: np.ndarray):
        text = input("Enter the phrase text (first-person): ")  # GUI will replace this
        path = self.config.base_dir / "calming" / f"{key}.wav"
        sf.write(path, audio, self.config.sample_rate)
        self.phrases[key] = text
        self.save_phrases()

