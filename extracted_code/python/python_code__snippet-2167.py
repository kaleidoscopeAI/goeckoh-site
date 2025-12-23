"""
Full speaker description:
- fingerprint: psychoacoustic bubble parameters
- embedding: neural timbre embedding for the TTS backbone
"""

user_id: str
fingerprint: VoiceFingerprint
embedding: np.ndarray  # 1D float32 array (e.g., 256–1024 dims)

def to_dict(self) -> Dict[str, Any]:
    return {
        "user_id": self.user_id,
        "fingerprint": self.fingerprint.to_dict(),
        "embedding": self.embedding.tolist(),
    }


