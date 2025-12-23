class EchoCompanion:
    """
    Orchestrates the SpeechLoop and provides an API for external interfaces
    like the Flask dashboard or mobile clients.
    """

    settings: SystemSettings = field(default_factory=load_settings)
    speech_loop: SpeechLoop = field(init=False)
    _loop_task: asyncio.Task | None = None
    _metrics_cache: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.speech_loop = SpeechLoop(settings=self.settings)
        # Start the speech loop in a separate thread/task for continuous operation
        # For simplicity, we'll start it here directly, but in a real app
        # you'd manage this with a proper asyncio event loop.
        # self.start_loop() # This will be called externally by the main entry point

    async def start_loop(self) -> None:
        """Starts the main speech processing loop."""
        if not self._loop_task or self._loop_task.done():
            self._loop_task = asyncio.create_task(self.speech_loop.run())
            print("[EchoCompanion] Speech loop started.")

    async def stop_loop(self) -> None:
        """Stops the main speech processing loop."""
        if self._loop_task:
            self._loop_task.cancel()
            await asyncio.gather(self._loop_task, return_exceptions=True)
            self._loop_task = None
            print("[EchoCompanion] Speech loop stopped.")

    def get_latest_metrics(self) -> Dict[str, Any]:
        """Returns the latest metrics for the dashboard."""
        # This is a simplified cache for demonstration.
        # In a real system, SpeechLoop would push updates or expose a stream.
        latest_attempt = self.speech_loop.metrics_logger.tail(limit=1)
        if latest_attempt:
            record = latest_attempt[0]
            # Fetch heart state from the speech loop
            heart_state = self.speech_loop.heart.step(np.array([])) # dummy audio for state
            self._metrics_cache = {
                "timestamp_iso": record.timestamp.isoformat(),
                "raw_text": record.raw_text,
                "corrected_text": record.corrected_text,
                "arousal": heart_state.get("arousal_raw", 0.0),
                "valence": heart_state.get("emotions", np.array([0,0]))[0,1] if "emotions" in heart_state else 0.0, # Placeholder
                "temperature": heart_state.get("T", 0.0),
                "coherence": heart_state.get("coherence", 0.0),
            }
        return self._metrics_cache

    def get_phrase_stats(self) -> Dict[str, Any]:
        """Returns statistics about phrase correction for the dashboard."""
        # This will be more complex, involving parsing the metrics CSV
        # For now, return dummy data or process what's available
        stats: Dict[str, Dict[str, Any]] = {}
        for record in self.speech_loop.metrics_logger.tail(limit=100):
            phrase = record.phrase_text or "<empty>"
            if phrase not in stats:
                stats[phrase] = {"attempts": 0, "corrections": 0, "correction_rate": 0.0}
            stats[phrase]["attempts"] += 1
            if record.needs_correction:
                stats[phrase]["corrections"] += 1
            stats[phrase]["correction_rate"] = stats[phrase]["corrections"] / stats[phrase]["attempts"]
        return stats

    async def process_utterance_for_mobile(self, audio_np: np.ndarray) -> Dict[str, Any]:
        """
        Processes an audio utterance received from a mobile client.
        This bypasses the microphone stream and directly feeds into _handle_utterance.
        """
        print("[EchoCompanion] Processing mobile utterance...")
        await self.speech_loop._handle_utterance(audio_np)
        return {"status": "processed", "latest_metrics": self.get_latest_metrics()}

