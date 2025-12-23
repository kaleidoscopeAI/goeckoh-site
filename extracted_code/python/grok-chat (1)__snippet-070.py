class DeepReasoningCore:
    def __init__(self):
        self.strategies = {
            "green": ["Excited guidance: Great job, let's build on that!"],
            "yellow": ["Simple affirm: You're doing well."],
            "red": ["Calm script: Breathe deep... I am safe."]
        }

    def execute(self, text, gcl):
        if gcl > 0.7:  # GREEN
            return random.choice(self.strategies["green"])
        elif gcl > 0.4:  # YELLOW
            return random.choice(self.strategies["yellow"])
        else:  # RED
            return random.choice(self.strategies["red"])

