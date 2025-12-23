def __init__(self):
    self.rules = {
        "anxious": "Breathe deeply... calm waves flow.",
        "happy": "Great energy... keep shining.",
        "neutral": "Steady path... proceed mindfully."
    }

def reason(self, input_text, gcl):
    if gcl < 0.3:
        return "GCL low... resting state."  # Gated low
    elif gcl < 0.7:
        return "Partial insight: " + random.choice(list(self.rules.values()))  # Medium
    else:
        # Full reasoning: Simple response
        for key in self.rules:
            if key in input_text.lower():
                return self.rules[key]
        return "Deep thought: Echoing your intent."

