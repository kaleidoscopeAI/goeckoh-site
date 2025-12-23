class AuralCommandInterface:
    def __init__(self, node_name: str, sample_rate: int = 44100):
        self.node_name = node_name
        self.sample_rate = sample_rate
        self.audio_buffer: Optional[np.ndarray] = None

    def update_buffer_from_environment(self, sound_level: str):
        amplitude = 0.05 if sound_level.lower() != "speaking" else 0.6
        duration_sec = 0.5
        num_samples = int(self.sample_rate * duration_sec)
        self.audio_buffer = np.random.normal(0, 0.01, num_samples) * amplitude

    def dispatch_latest_chunk(self, orches: 'AGIOrchestrator'):
        if self.audio_buffer is None: return
        raw_data = self.audio_buffer
        insight = {"content": "Aural input simulated", "modality": "sound"}
        orches.graph.add_insight(insight)

