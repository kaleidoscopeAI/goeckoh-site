class StrategyAdvisor:
    def suggest(self, event: str) -> List[Strategy]:
        # Simple mapping for now
        if event == \"anxious\":
            return [s for s in STRATEGIES if s.category in [\"mindfulness\", \"sensory_tools\"]]
        # Add more mappings
        return []
