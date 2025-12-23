class StrategyAdvisor:
    """Returns curated strategies for caregiver dashboards."""

    def suggest(self, event: str, limit: int = 3) -> List[Strategy]:
        cats = EVENT_TO_CATEGORIES.get(event)
        if not cats:
            return STRATEGIES[:limit]
        ordered: List[Strategy] = []
        for cat in cats:
            ordered.extend([s for s in STRATEGIES if s.category == cat])
        return (ordered or STRATEGIES)[:limit]

