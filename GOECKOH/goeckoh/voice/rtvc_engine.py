"""
Wrapper for the Real-Time-Voice-Cloning pipeline (encoder + synthesizer + vocoder).

This module is deliberately thin: it imports the third-party code from the checked-in
`third_party/Real-Time-Voice-Cloning` directory and exposes a simple synth API that fits
the rest of the system. Models must already be present on disk; we do not auto-download
them to avoid unexpected network calls.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np


class VoiceEngineRTVC:
    """
    Real-Time-Voice-Cloning inference wrapper.

    - Expects the upstream repo under GOECKOH_RTVC_PATH (default:
      echovoice/project/echo_companion/third_party/Real-Time-Voice-Cloning).
    - Expects pretrained models under GOECKOH_RTVC_MODELS (default: <repo>/saved_models/default/).
      Required files: encoder.pt, synthesizer.pt, vocoder.pt.
    """

    def __init__(
        self,
        repo_path: Optional[str | Path] = None,
        model_root: Optional[str | Path] = None,
        device: Optional[str] = None,
    ):
        self.repo_path = Path(
            repo_path
            or os.getenv(
                "GOECKOH_RTVC_PATH",
                "echovoice/project/echo_companion/third_party/Real-Time-Voice-Cloning",
            )
        ).expanduser().resolve()

        if not self.repo_path.exists():
            raise FileNotFoundError(f"RTVC repo not found at {self.repo_path}")

        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))

        # Lazy imports from the RTVC repo
        try:
            import torch  # type: ignore
            from encoder import inference as encoder  # type: ignore
            from encoder.params_model import model_embedding_size as speaker_embedding_size  # type: ignore
            from synthesizer.inference import Synthesizer  # type: ignore
            from vocoder import inference as vocoder  # type: ignore
        except Exception as e:  # pragma: no cover - upstream import guard
            raise ImportError(f"Failed to import RTVC modules: {e}") from e

        self.torch = torch
        self.encoder = encoder
        self.Synthesizer = Synthesizer
        self.vocoder = vocoder
        self.speaker_embedding_size = speaker_embedding_size

        # Resolve model paths
        model_root_path = Path(
            model_root
            or os.getenv("GOECKOH_RTVC_MODELS", self.repo_path / "saved_models" / "default")
        ).expanduser().resolve()
        self.encoder_path = Path(
            os.getenv("GOECKOH_RTVC_ENCODER", model_root_path / "encoder.pt")
        )
        self.synthesizer_path = Path(
            os.getenv("GOECKOH_RTVC_SYNTH", model_root_path / "synthesizer.pt")
        )
        self.vocoder_path = Path(
            os.getenv("GOECKOH_RTVC_VOCODER", model_root_path / "vocoder.pt")
        )

        for p in [self.encoder_path, self.synthesizer_path, self.vocoder_path]:
            if not p.exists():
                raise FileNotFoundError(f"RTVC model missing: {p}")

        # Device selection: explicit > env > GPU if available
        self.device = device or os.getenv("GOECKOH_RTVC_DEVICE")
        if self.device is None:
            self.device = "cuda" if self.torch.cuda.is_available() else "cpu"

        # Load models
        self._load_models()

    def _load_models(self):
        self.encoder.load_model(self.encoder_path, device=self.device)
        self.synthesizer = self.Synthesizer(self.synthesizer_path)
        self.vocoder.load_model(self.vocoder_path, device=self.device)

    def synthesize(self, text: str, ref_wav_path: str) -> Optional[np.ndarray]:
        """
        Generate a waveform that mimics the reference speaker.
        Returns mono float32 PCM in [-1, 1], or None on failure.
        """
        try:
            # Build speaker embedding
            preprocessed_wav = self.encoder.preprocess_wav(Path(ref_wav_path))
            embed = self.encoder.embed_utterance(preprocessed_wav)

            # Text → mel
            mels = self.synthesizer.synthesize_spectrograms([text], [embed])
            mel = mels[0]

            # Mel → waveform
            wav = self.vocoder.infer_waveform(mel)
            # Pad and trim per upstream guidance
            wav = np.pad(wav, (0, self.synthesizer.sample_rate), mode="constant")
            wav = self.encoder.preprocess_wav(wav)
            return wav.astype(np.float32)
        except Exception as e:
            print(f"[RTVC] Synthesis failed: {e}")
            return None
