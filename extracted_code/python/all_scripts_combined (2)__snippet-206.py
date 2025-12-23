settings: SystemSettings

def __post_init__(self) -> None:
    self._phrases: Deque[str] = deque(maxlen=self.settings.behavior.max_phrase_history)
    self._correction_streak = 0
    self._last_event: Optional[str] = None

def register(self, normalized_text: str, needs_correction: bool, rms: float) -> Optional[str]:
    event: Optional[str] = None

    if needs_correction:
        self._correction_streak += 1
    else:
        if self._correction_streak >= self.settings.behavior.anxious_threshold:
            event = "encouragement"
        self._correction_streak = 0

    self._phrases.append(normalized_text)

    if self._correction_streak >= self.settings.behavior.anxious_threshold:
        event = "anxious"
    elif normalized_text and list(self._phrases).count(normalized_text) >= self.settings.behavior.perseveration_threshold:
        event = event or "perseveration"
    elif rms >= self.settings.behavior.high_energy_rms:
        event = event or "high_energy"

    if event == self._last_event and event not in {"perseveration", "encouragement"}:
        return None
    if event:
        self._last_event = event
    return event

