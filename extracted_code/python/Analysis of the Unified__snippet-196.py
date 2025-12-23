"""Maintains and learns from ethical decisions and their consequences."""
def __init__(self):
    self.decisions = []
    self.consequences = defaultdict(list)
    self.learning_rate = 0.1

def record_decision(self, decision: Dict, context: Dict):
    """Record an ethical decision and its context."""
    self.decisions.append({
        "decision": decision,
        "context": context,
        "timestamp": datetime.now(),
        "outcome_pending": True
    })

def update_consequence(self, decision_id: str, consequence: Dict):
    """Record the consequence of a previous decision."""
    self.consequences[decision_id].append({
        "impact": consequence,
        "timestamp": datetime.now()
    })

    # Update learning rate based on consequence severity
    severity = consequence.get("severity", 0.5)
    self.learning_rate = max(0.01, min(0.5, self.learning_rate * (1 + severity - 0.5)))

