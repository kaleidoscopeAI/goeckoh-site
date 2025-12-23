def __init__(self, reflection_interval: int = 100):
    self.reflection_interval = reflection_interval
    self.action_history = []
    self.insights = []

def reflect(self, actions: List[Dict]) -> Dict:
    if len(actions) < self.reflection_interval:
        return {}

    success_rates = [a['success'] for a in actions if 'success' in a]
    average_success = np.mean(success_rates) if success_rates else 0.0
    insight = {"average_success": average_success, "timestamp": time.time()}
    self.insights.append(insight)
    return insight

