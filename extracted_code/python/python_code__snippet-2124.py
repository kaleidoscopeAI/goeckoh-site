"""Autism-optimized Voice Activity Detection"""

def __init__(self):
    # Autism-tuned parameters
    self.threshold = 0.45
    self.min_silence_duration_ms = 1200
    self.speech_pad_ms = 400
    self.min_speech_duration_ms = 250
    self.sample_rate = 16000
    self.accumulated_speech_energy = 0.0
    self.speech_threshold = 2.0

def process_audio_chunk(self, audio_chunk: np.ndarray) -> Tuple[bool, bool]:
    """Process audio with autism-optimized parameters"""
    # Calculate speech energy with autism-optimized sensitivity
    energy = np.sum(audio_chunk ** 2)
    self.accumulated_speech_energy += energy

    # Dynamic threshold adjustment for quiet/monotone speech
    dynamic_threshold = self.threshold * 1000 * (1.0 + 0.2 * np.sin(time.time() * 0.1))
    is_speech = energy > dynamic_threshold

    # Autism-optimized pause detection with longer tolerance
    if self.accumulated_speech_energy > self.speech_threshold:
        should_transcribe = True
        self.accumulated_speech_energy = 0.0

        # Log speech detection for autism analytics
        if hasattr(self, 'speech_detections'):
            self.speech_detections.append(time.time())
        else:
            self.speech_detections = [time.time()]
    else:
        should_transcribe = False

    return is_speech, should_transcribe

def get_pause_analysis(self) -> Dict[str, float]:
    """Analyze speech patterns for autism support"""
    if not hasattr(self, 'speech_detections') or len(self.speech_detections) < 2:
        return {'avg_pause_duration': 0.0, 'speech_rate': 0.0, 'pause_variance': 0.0}

    # Calculate pause durations
    pauses = []
    for i in range(1, len(self.speech_detections)):
        pause_duration = self.speech_detections[i] - self.speech_detections[i-1]
        pauses.append(pause_duration)

    if pauses:
        avg_pause = np.mean(pauses)
        speech_rate = 1.0 / avg_pause if avg_pause > 0 else 0.0
        pause_variance = np.var(pauses)

        return {
            'avg_pause_duration': avg_pause,
            'speech_rate': speech_rate,
            'pause_variance': pause_variance,
            'long_pause_count': sum(1 for p in pauses if p > self.min_silence_duration_ms / 1000.0)
        }

    return {'avg_pause_duration': 0.0, 'speech_rate': 0.0, 'pause_variance': 0.0}

