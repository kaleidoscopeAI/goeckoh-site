class StrategyAdvisor:
    """Returns a small list of strategy hints for a given event."""

    def suggest(self, event: str) -> List[Strategy]:
        cats = EVENT_TO_STRATEGIES.get(event)
        if not cats:
            return STRATEGIES[:1]
        ordered: List[Strategy] = []
        for cat in cats:
            ordered.extend([s for s in STRATEGIES if s.category == cat])
        return ordered or STRATEGIES[:1]


