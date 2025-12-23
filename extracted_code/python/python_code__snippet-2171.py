"""
Coqui TTS integration with proper fallback handling.
"""
def __init__(self) -> None:
    self._tts = None
    if COQUI_AVAILABLE:
        try:
            self._tts = TTS("tts_models/en/ljspeech/vits")
        except Exception as e:
            print(f"[WARN] Coqui TTS initialization failed: {e}")

def g2p(self, text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    # Use real phoneme conversion
    real_vocoder = RealVocoderBackend()
    return real_vocoder.g2p(text)

def synthesize(
    self,
    phonemes: List[str],
    speaker_embedding: np.ndarray,
    pitch_contour: np.ndarray,
    energy_contour: np.ndarray,
    hnr_contour: np.ndarray,
    tilt_contour: np.ndarray,
    dt: float,
) -> np.ndarray:
    if self._tts is None:
        # Use real formant synthesis as fallback
        real_vocoder = RealVocoderBackend()
        return real_vocoder.synthesize(
            phonemes, speaker_embedding, pitch_contour, 
            energy_contour, hnr_contour, tilt_contour, dt
        )

    try:
        text = "".join(phonemes)
        audio = self._tts.tts(text=text)
        return np.array(audio, dtype=np.float32)
    except Exception as e:
        print(f"[WARN] Coqui TTS failed: {e}, using formant synthesis")
        real_vocoder = RealVocoderBackend()
        return real_vocoder.synthesize(
            phonemes, speaker_embedding, pitch_contour, 
            energy_contour, hnr_contour, tilt_contour, dt
        )


