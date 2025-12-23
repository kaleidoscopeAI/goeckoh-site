# ECHO_V4_UNIFIED/trigger_engine.py
# Guardian-labeled triggers: noises -> first-person phrases
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
try:
    from crystal_brain.store import MemoryStore, TriggerRecord  # type: ignore
except Exception:
    from .src.core.store import MemoryStore, TriggerRecord  # type: ignore
try:
    from audio_features import compute_audio_embedding  # type: ignore
except Exception:
    from .audio_features import compute_audio_embedding  # type: ignore
try:
    from config import CONFIG  # type: ignore
except Exception:
    from .config import CONFIG  # type: ignore
from pathlib import Path
import time

@dataclass
class TriggerMatch:
    trigger_id: int
    phrase: str
    similarity: float

class TriggerEngine:
    """
    Backend logic for:
    - storing all utterance snippets (including noise)
    - registering triggers (guardian labels)
    - matching new noises to triggers
    """
    def __init__(self) -> None:
        self.store = MemoryStore()
        self.snippet_dir: Path = CONFIG.db.db_path.parent / "snippets"
        self.snippet_dir.mkdir(parents=True, exist_ok=True)
        self._triggers: List[TriggerRecord] = []
        self._last_load_time: float = 0.0

    def store_snippet(self,
                      audio: np.ndarray,
                      sample_rate: int,
                      t: float,
                      asr_text: str) -> int:
        """
        Persists audio to disk and stores metadata+embedding in DB.
        """
        try:
            from scipy.io import wavfile  # type: ignore
        except Exception as e:
            raise ImportError("scipy is required to write audio snippets.") from e
        tmp_name = f"snippet_{int(t)}_{int(time.time() * 1000)}.wav"
        path = self.snippet_dir / tmp_name
        
        try:
            if np.issubdtype(audio.dtype, np.floating):
                audio_int16 = (audio * 32767).astype(np.int16)
            else:
                audio_int16 = audio.astype(np.int16)
            wavfile.write(path, sample_rate, audio_int16)
        except Exception as e:
            print(f"Error writing snippet file {path}: {e}")
        
        emb = compute_audio_embedding(audio, sample_rate)
        
        return self.store.store_audio_snippet(
            time=t,
            path=str(path),
            embedding=emb,
            asr_text=asr_text or "",
        )

    def register_trigger(self,
                         snippet_id: int,
                         phrase_first_person: str,
                         threshold: float = 0.85) -> int:
        """
        Guardian calls this (directly or via UI) to map one snippet to a phrase.
        The phrase MUST already be first-person ("I ...").
        """
        snip = self.store.get_audio_snippet(snippet_id)
        emb = snip.embedding
        
        # Invalidate cache so it reloads on next match
        self._last_load_time = 0.0
        
        return self.store.store_trigger(
            snippet_id=snippet_id,
            phrase=phrase_first_person,
            threshold=threshold,
            embedding=emb,
        )

    def _load_triggers(self) -> None:
        """Loads triggers from DB, caching them for performance."""
        # Cache for 10 seconds to reduce DB reads
        if time.time() - self._last_load_time > 10.0:
            self._triggers = self.store.get_triggers()
            self._last_load_time = time.time()

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Calculates cosine similarity, handling potential zero vectors."""
        if a.shape != b.shape:
            return 0.0
            
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-9:
            return 0.0
        
        return float(np.dot(a, b) / denom)

    def match_for_audio(self,
                        audio: np.ndarray,
                        sample_rate: int) -> Optional[TriggerMatch]:
        """
        Given a new noise utterance, return the best-matching trigger if any.
        """
        self._load_triggers()
        if not self._triggers:
            return None

        emb = compute_audio_embedding(audio, sample_rate)
            
        best: Optional[TriggerMatch] = None
        for tr in self._triggers:
            sim = self._cosine_sim(emb, tr.embedding)
            if sim >= tr.threshold:
                if best is None or sim > best.similarity:
                    best = TriggerMatch(
                        trigger_id=tr.id,
                        phrase=tr.phrase,
                        similarity=sim,
                    )
        return best
