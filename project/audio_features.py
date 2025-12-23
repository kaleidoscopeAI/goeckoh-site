# ECHO_V4_UNIFIED/audio_features.py
# Turn arbitrary audio (including noise) into a compact embedding
# for trigger matching.
from __future__ import annotations
import numpy as np
try:
    import librosa  # type: ignore
    _HAS_LIBROSA = True
except Exception:
    librosa = None  # type: ignore
    _HAS_LIBROSA = False

def compute_audio_embedding(
    audio: np.ndarray,
    sample_rate: int,
    emb_dim: int = 64,
) -> np.ndarray:
    """
    Compute a simple, stable embedding for an audio snippet using MFCCs.
    Works even for noisy/non-speech sounds.
    Returns a float32 vector of length emb_dim, L2-normalized.
    """
    if not _HAS_LIBROSA:
        raise ImportError("Audio embedding requires librosa.")
    if audio.ndim > 1:
        audio = audio.mean(axis=1) # Average channels if stereo
    if audio.size == 0:
        return np.zeros((emb_dim,), dtype="float32")

    try:
        # MFCC feature summary
        mfcc = librosa.feature.mfcc(
            y=audio.astype(np.float32),
            sr=sample_rate,
            n_mfcc=13,
        )
        feat = np.mean(mfcc, axis=1) # shape (13,)

        # Tile/trim to emb_dim
        reps = (emb_dim + feat.shape[0] - 1) // feat.shape[0]
        tiled = np.tile(feat, reps)[:emb_dim].astype("float32")

        # Normalize
        norm = np.linalg.norm(tiled)
        if norm > 0:
            tiled /= norm
            
        return tiled
    except Exception as e:
        print(f"Error computing audio embedding: {e}")
        return np.zeros((emb_dim,), dtype="float32")
