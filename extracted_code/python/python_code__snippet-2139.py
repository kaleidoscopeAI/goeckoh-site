"""Pure NumPy implementation of autism-optimized VAD"""

def __init__(self):
    # Autism-tuned parameters from documents
    self.threshold = 0.45
    self.min_silence_duration_ms = 1200
    self.speech_pad_ms = 400
    self.min_speech_duration_ms = 250
    self.sample_rate = 16000
    self.accumulated_speech_energy = 0.0
    self.speech_threshold = 2.0

def process_audio_chunk(self, audio_chunk: np.ndarray) -> Tuple[bool, bool]:
    """Process audio with autism-optimized parameters"""
    # Calculate speech energy
    energy = np.sum(audio_chunk ** 2)
    self.accumulated_speech_energy += energy

    # Simple VAD simulation using energy threshold
    is_speech = energy > self.threshold * 1000

    # Only trigger if accumulated energy exceeds threshold
    if self.accumulated_speech_energy > self.speech_threshold:
        should_transcribe = True
        self.accumulated_speech_energy = 0.0
    else:
        should_transcribe = False

    return is_speech, should_transcribe

def reset(self):
    """Reset for new utterance"""
    self.accumulated_speech_energy = 0.0

