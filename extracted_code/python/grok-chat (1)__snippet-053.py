from .pragmatic_engine import PragmaticEngine

class SpeechLoop:
    # ...

    async def handle_chunk(self, chunk: np.ndarray) -> None:
        # ... (transcription)

        analysis = self.semantic.analyze(raw)  # Now dict with semantic + pragmatic

        pragmatic = analysis["pragmatic"]

        # Adjust based on pragmatics
        if pragmatic.type == "sarcasm":
            style = "calm"  # Soften response
            corrected = f"I sense frustration... {corrected}"  # Optional rephrase

        # Stimulus from pragmatic confidence
        stimulus = np.random.rand(self.heart.n_nodes, 5)
        stimulus[:, 1] += pragmatic.confidence  # Boost valence based on pragmatics

        self.heart.step(stimulus)

        # ... (echo with pragmatic-aware style)
