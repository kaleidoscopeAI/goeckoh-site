from .semantic_engine import SemanticEngine

class SpeechLoop:
    # ...

    def __post_init__(self) -> None:
        # ...
        self.semantic = SemanticEngine()

    async def handle_chunk(self, chunk: np.ndarray) -> None:
        # ... (transcription)

        semantic_intent = self.semantic.analyze(raw)

        # Feed to lattice as stimulus (e.g., vector from intent/emotion)
        stimulus = np.random.rand(self.heart.n_nodes, 5)  # Placeholder; map semantic to vector
        if semantic_intent.emotion == "anxious":
            stimulus[:, 0] += 0.5  # Boost arousal
        self.heart.step(stimulus)

        style = semantic_intent.emotion  # Adjust mimicry style

        # ... (echo with adjusted style)
