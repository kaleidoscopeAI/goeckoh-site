class GuidanceEngine:
    logger: GuidanceLogger
    parent_phrases: Dict[EventType, List[str]] = field(default_factory=lambda: {
        "meltdown_risk": ["Everything is okay. Close my eyes and breathe."],
        "anxious": ["I am safe. Let's take a deep breath."],
        "success": ["Great job! I did it."]
    })

    def add_parent_phrase(self, event_type: EventType, phrase: str) -> None:
        if event_type not in self.parent_phrases:
            self.parent_phrases[event_type] = []
        self.parent_phrases[event_type].append(phrase)

    def get_guidance(self, event_type: EventType) -> Optional[str]:
        phrases = self.parent_phrases.get(event_type, [])
        if phrases:
            return np.random.choice(phrases)
        return None

    def trigger(self, event_type: EventType) -> None:
        message = self.get_guidance(event_type)
        if message:
            self.logger.log(event_type, "Guidance Triggered", message)
            # Integrate with voice: speak(message, style="calm", mode="coach")
            print(f"Guiding: {message}")
