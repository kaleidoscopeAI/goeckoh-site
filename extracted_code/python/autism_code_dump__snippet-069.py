"""
Collects audio frames, detects utterances with tuned silence thresholds.
Silero refinement optional.
"""

def __init__(self, settings: VADSettings):
    self.settings = settings
    self.buffer: List[np.ndarray] = []
    self.is_speech = False
    self.silence_start: Optional[float] = None
    self.out_queue: "queue.Queue[np.ndarray]" = queue.Queue()

def audio_callback(self, indata, frames, time_info, status):
    if status:
        print(f"[VADStream] status: {status}")
    y = indata.copy()
    rms = float(np.sqrt(np.mean(y**2)))
    now = time.time()

    if rms > self.settings.silence_rms_threshold:
        self.buffer.append(y)
        self.is_speech = True
        self.silence_start = None
    else:
        if self.is_speech and self.silence_start is None:
            self.silence_start = now
        elif self.is_speech and self.silence_start:
            dur_ms = (now - self.silence_start) * 1000.0
            if dur_ms > self.settings.vad_min_silence_ms:
                full_audio = np.concatenate(self.buffer, axis=0)
                self.buffer = []
                self.is_speech = False
                self.silence_start = None
                min_samples = int(
                    self.settings.vad_min_speech_ms
                    / 1000.0
                    * self.settings.samplerate
                )
                if full_audio.shape[0] >= min_samples:
                    if SILERO_AVAILABLE:
                        try:
                            full_audio_16 = (full_audio.flatten() * 32767).astype(
                                np.int16
                            )
                            timestamps = get_speech_timestamps(
                                full_audio_16,
                                sampling_rate=self.settings.samplerate,
                            )
                            if timestamps:
                                speech_chunks = collect_chunks(
                                    timestamps, full_audio_16
                                )
                                full_audio = (
                                    speech_chunks.astype(np.float32) / 32767.0
                                )[:, None]
                        except Exception as e:
                            print(f"[VADStream] Silero failed: {e}")
                    self.out_queue.put(full_audio)


