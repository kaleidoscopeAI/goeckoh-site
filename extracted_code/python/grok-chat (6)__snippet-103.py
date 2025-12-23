class DeepReasoningCore:
    def __init__(self):
        self.rules = {"help": "I guide: Breathe calm.", "happy": "Joy flows: Well done."}

    def execute(self, text, gcl):
        if gcl > 0.7:
            for key in self.rules:
                if key in text.lower():
                    return self.rules[key]
            return "Reason: Insight unlocked."
        elif gcl > 0.4:
            return "Affirm: Steady."
        else:
            return "Calm: I am safe."

