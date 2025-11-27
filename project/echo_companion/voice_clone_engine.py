import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf

from config import (
    ENCODER_MODEL_FPATH,
    SYNTHESIZER_MODEL_FPATH,
    VOCODER_MODEL_FPATH,
    SPEAKER_REF_DIR,
    RTVC_SAMPLE_RATE,
)


def _lazy_import_rtvc():
    try:
        from encoder import inference as encoder  # type: ignore
        from synthesizer.inference import Synthesizer  # type: ignore
        from vocoder import inference as vocoder  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Real-Time-Voice-Cloning not installed. Run echo_companion/setup.sh first "
            "or drop a pre-cloned repo under third_party/Real-Time-Voice-Cloning."
        ) from exc
    return encoder, Synthesizer, vocoder


class VoiceCloneEngine:
    """
    Wrapper around Real-Time-Voice-Cloning for:
    - Speaker enrollment from reference WAVs
    - Text-to-speech in the enrolled voice
    """

    def __init__(self) -> None:
        encoder, Synthesizer, vocoder = _lazy_import_rtvc()
        for path in (ENCODER_MODEL_FPATH, SYNTHESIZER_MODEL_FPATH, VOCODER_MODEL_FPATH):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing RTVC model: {path}. "
                    "Run setup.sh to auto-download the RTVC checkpoints "
                    "or place them under third_party/Real-Time-Voice-Cloning/."
                )

        encoder.load_model(str(ENCODER_MODEL_FPATH))
        self.encoder = encoder
        self.synthesizer = Synthesizer(str(SYNTHESIZER_MODEL_FPATH))
        vocoder.load_model(str(VOCODER_MODEL_FPATH))
        self.vocoder = vocoder
        self._speaker_embedding: Optional[np.ndarray] = None

    def enroll_from_directory(self, ref_dir: Path = SPEAKER_REF_DIR) -> None:
        """
        Load all WAV files from ref_dir, compute embeddings, and average them
        into a single speaker embedding.
        """
        if not ref_dir.exists():
            raise FileNotFoundError(f"Speaker ref dir not found: {ref_dir}")

        wav_paths: List[Path] = [p for p in ref_dir.glob("*.wav") if p.is_file()]
        if not wav_paths:
            raise RuntimeError(f"No .wav files found in {ref_dir}")

        embeds = []
        for wav_path in wav_paths:
            wav, sr = sf.read(wav_path)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            wav = self.encoder.preprocess_wav(wav, sr)
            embed = self.encoder.embed_utterance(wav)
            embeds.append(embed)

        self._speaker_embedding = np.mean(embeds, axis=0)

    def is_enrolled(self) -> bool:
        return self._speaker_embedding is not None

    def speak(self, text: str) -> np.ndarray:
        """
        Synthesize text in the enrolled voice.
        Returns float32 waveform (RTVC_SAMPLE_RATE).
        """
        if not self.is_enrolled():
            raise RuntimeError("VoiceCloneEngine: speaker not enrolled yet.")

        texts = [text]
        embeds = [self._speaker_embedding]  # type: ignore[list-item]

        mels = self.synthesizer.synthesize_spectrograms(texts, embeds)
        mel = mels[0]

        wav = self.vocoder.infer_waveform(mel)
        wav = np.pad(wav, (0, 4000), mode="constant")
        peak = np.abs(wav).max()
        if peak > 0:
            wav = wav / peak

        return wav.astype(np.float32)
