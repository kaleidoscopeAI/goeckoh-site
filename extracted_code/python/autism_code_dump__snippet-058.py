"""Voice-activity detection based on RMS and silence patience."""

def __init__(self, cfg: AudioConfig):
    self.cfg = cfg
    self._buffer_chunks: List[np.ndarray] = []
    self._is_speech: bool = False
    self._silence_start: Optional[float] = None

def process_block(self, audio_block: np.ndarray) -> List[np.ndarray]:
    return self._process_block_rms(audio_block)

def _process_block_rms(self, audio_block: np.ndarray) -> List[np.ndarray]:
    cfg = self.cfg
    block = audio_block.astype(np.float32)
    rms = float(np.sqrt(np.mean(block**2)))
    utterances: List[np.ndarray] = []

    if rms > cfg.rms_voice_threshold:
        self._buffer_chunks.append(block)
        self._is_speech = True
        self._silence_start = None
    else:
        if self._is_speech and self._silence_start is None:
            self._silence_start = time.time()
        elif self._is_speech and self._silence_start is not None:
            if (time.time() - self._silence_start) * 1000.0 > cfg.min_silence_ms:
                if self._buffer_chunks:
                    full = np.concatenate(self._buffer_chunks, axis=0)
                    utterances.append(full.reshape(-1))
                self._buffer_chunks = []
                self._is_speech = False
                self._silence_start = None

    return utterances


