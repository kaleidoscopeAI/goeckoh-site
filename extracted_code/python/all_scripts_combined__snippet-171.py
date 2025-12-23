from __future__ import annotations
import numpy as np
from faster_whisper import WhisperModel
import language_tool_python
from core.settings import SpeechSettings
from .gears import Information, AudioData, SpeechData

class SpeechProcessorGear:
    """
    A gear that processes audio data into raw and corrected text.
    """
    def __init__(self, speech_cfg: SpeechSettings, device: str = "cpu"):
        self.config = speech_cfg
        self.device = device
        
        # Load Whisper model (using int8 for CPU performance)
        print(f"Loading Whisper model: {self.config.whisper_model}...")
        self.model = WhisperModel(self.config.whisper_model, device=self.device, compute_type="int8")
        
        # Initialize LanguageTool
        print("Loading LanguageTool...")
        try:
            if self.config.language_tool_server:
                self.lt = language_tool_python.LanguageTool('en-US', remote_server=self.config.language_tool_server)
            else:
                self.lt = language_tool_python.LanguageTool('en-US')
        except Exception as e:
            print(f"Warning: Could not initialize LanguageTool. Grammar correction will be disabled. Error: {e}")
            self.lt = None

    def process(self, audio_info: Information) -> Information:
        """
        Transcribes and corrects speech from an audio Information object.
        """
        if not isinstance(audio_info.payload, AudioData):
            raise TypeError("SpeechProcessorGear expects an Information object with an AudioData payload.")

        audio_waveform = audio_info.payload.waveform
        
        # 1. Transcribe audio using faster-whisper
        # The model expects a float32 numpy array.
        segments, info = self.model.transcribe(audio_waveform, beam_size=5)
        raw_text = " ".join([s.text for s in segments]).strip()
        
        if not raw_text:
            return audio_info.new(
                payload=SpeechData(raw_text="", corrected_text="", is_final=True),
                source_gear="SpeechProcessorGear"
            )

        # 2. Correct grammar using LanguageTool
        corrected_text = raw_text
        if self.lt:
            try:
                matches = self.lt.check(raw_text)
                corrected_text = language_tool_python.utils.correct(raw_text, matches)
            except Exception as e:
                print(f"Warning: LanguageTool correction failed. Using raw text. Error: {e}")

        return audio_info.new(
            payload=SpeechData(raw_text=raw_text, corrected_text=corrected_text, is_final=True),
            source_gear="SpeechProcessorGear"
        )

