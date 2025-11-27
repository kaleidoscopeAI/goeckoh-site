# voice_crystal.py
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional
import hashlib
import scipy.io.wavfile as wavfile
from scipy.spatial.distance import cosine


class DefaultAudioConfig:
    def __init__(self):
        self.sample_rate = 16000

class DefaultModelPaths:
    def __init__(self):
        self.voice_samples_dir = Path("voice_samples")

class DefaultConfig:
    def __init__(self):
        self.audio = DefaultAudioConfig()
        self.models = DefaultModelPaths()

CONFIG = DefaultConfig()
CONFIG.models.voice_samples_dir.mkdir(parents=True, exist_ok=True)


def mock_extract_embedding(audio: np.ndarray) -> np.ndarray:
    return np.random.rand(256).astype(np.float32)


class VoiceCrystal:
    def __init__(self) -> None:
        self.voice_dir = CONFIG.models.voice_samples_dir
        self.ppp_cache: dict[str, np.ndarray] = {}
        self.embedding_history: list[np.ndarray] = []
        self.current_embedding: Optional[np.ndarray] = None
        self.voice_drift_threshold = 0.08

    def _hash_audio(self, audio: np.ndarray) -> str:
        h = hashlib.sha256()
        h.update(audio.astype(np.float32).tobytes())
        return h.hexdigest()

    def _load_fragments(self) -> list[np.ndarray]:
        fragments = []
        for p in sorted(self.voice_dir.glob("fragment_*.wav")):
            try:
                sr, data = wavfile.read(p)
                if data.ndim > 1:
                    data = data[:, 0]
                if data.dtype.kind in 'iu':
                    data = data.astype(np.float32) / np.iinfo(data.dtype).max
                elif data.dtype.kind == 'f':
                    data = data.astype(np.float32)
                fragments.append(data)
            except Exception:
                continue
        return fragments

    def _blend_fragments(self, fragments: list[np.ndarray]) -> np.ndarray:
        if not fragments:
            raise ValueError("No fragments available to blend.")
        min_len = min(len(f) for f in fragments)
        trimmed = [f[:min_len] for f in fragments]
        stacked = np.stack(trimmed, axis=0)
        blend = np.mean(stacked, axis=0)
        return blend / (np.max(np.abs(blend)) + 1e-6)

    def build_ppp_voice(self) -> Path:
        fragments = self._load_fragments()
        if not fragments:
            raise RuntimeError("No fragments found to construct PPP voice.")
        voice_wave = self._blend_fragments(fragments)
        out_path = self.voice_dir / "jackson_ppp_auto.wav"
        wavfile.write(out_path, CONFIG.audio.sample_rate, (voice_wave * 32767).astype(np.int16))
        return out_path

    def update_with_fragment(self, audio: np.ndarray) -> None:
        key = self._hash_audio(audio)
        if key in self.ppp_cache:
            return
        frag_name = f"fragment_{key[:12]}.wav"
        path = self.voice_dir / frag_name
        wavfile.write(path, CONFIG.audio.sample_rate, (audio * 32767).astype(np.int16))
        self.ppp_cache[key] = audio

        new_emb = mock_extract_embedding(audio)
        self.embedding_history.append(new_emb)

        if self.current_embedding is None:
            self.current_embedding = new_emb
        else:
            dist = cosine(self.current_embedding, new_emb)
            if dist > self.voice_drift_threshold:
                self.current_embedding = (self.current_embedding + new_emb) / 2.0
                self.build_ppp_voice()

    def get_speaker_wav(self) -> Optional[str]:
        candidates = sorted(self.voice_dir.glob("jackson_*.wav"))
        if candidates:
            return str(candidates[0])
        ppp_path = self.voice_dir / "jackson_ppp_auto.wav"
        return str(ppp_path) if ppp_path.exists() else None


voice_crystal = VoiceCrystal()
