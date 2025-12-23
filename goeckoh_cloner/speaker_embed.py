from pathlib import Path
from typing import Optional

import torch
import torchaudio
import numpy as np
from speechbrain.inference import EncoderClassifier


class SpeakerEmbedder:
    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device},
        )

    def embed_wav(self, wav_path: Path) -> np.ndarray:
        signal, fs = torchaudio.load(str(wav_path))
        if fs != 16000:
            signal = torchaudio.functional.resample(signal, fs, 16000)
        with torch.no_grad():
            emb = self.classifier.encode_batch(signal.to(self.device))
        emb = emb.squeeze().cpu().numpy()
        return emb
