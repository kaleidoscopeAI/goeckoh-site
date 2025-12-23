class AutismOptimizedVAD:
    """
    Silero VAD parameters specifically tuned for autistic speech patterns
    From documents: threshold=0.45, min_silence=1200ms, speech_pad=400ms
    """
    
    def __init__(self):
        # Autism-tuned parameters from documents
        self.threshold = 0.45                 # Lower for quiet/monotone speech
        self.min_silence_duration_ms = 1200   # 1.2s for processing pauses
        self.speech_pad_ms = 400               # Capture slow starts/trailing thoughts
        self.min_speech_duration_ms = 250      # Allow single-word responses
        self.sample_rate = 16000
        self.accumulated_speech_energy = 0.0
        self.speech_threshold = 2.0           # Minimum energy before transcription
        
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> bool:
        """Process audio with autism-optimized parameters"""
        # Calculate speech energy
        energy = np.sum(audio_chunk ** 2)
        self.accumulated_speech_energy += energy
        
        # Simple VAD simulation (would use real Silero in production)
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

