def list_categories() -> List[str]:
    return sorted({s.category for s in STRATEGIES})


def by_category(category: str) -> List[Strategy]:
    return [s for s in STRATEGIES if s.category == category]


def suggest_for_event(event: str) -> List[Strategy]:
    cats = EVENT_TO_CATEGORIES.get(event)
    if not cats:
        return STRATEGIES[:3]
    ordered: List[Strategy] = []
    for cat in cats:
        ordered.extend(by_category(cat))
    return ordered or STRATEGIES[:3]


class StrategyAdvisor:
    """Tiny helper used by the realtime loop for context-aware nudges."""

    def __init__(self) -> None:
        self.events_seen: Dict[str, int] = {}

    def suggest(self, event: str) -> List[Strategy]:
        self.events_seen[event] = self.events_seen.get(event, 0) + 1
        return suggest_for_event(event)


