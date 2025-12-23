"""Tiny helper used by the realtime loop for context-aware nudges."""

def __init__(self) -> None:
    self.events_seen: Dict[str, int] = {}

def suggest(self, event: str) -> List[Strategy]:
    self.events_seen[event] = self.events_seen.get(event, 0) + 1
    return suggest_for_event(event)
