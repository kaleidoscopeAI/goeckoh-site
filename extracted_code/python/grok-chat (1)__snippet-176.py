def __init__(self):
    self.strategies = {
        "green": "Deep plan: Let's advance that skill.",
        "yellow": "Simple guide: Steady now.",
        "red": "Calm affirm: I am safe, breathe."
    }

def execute(self, text, gcl):
    if gcl > 0.7:  # GREEN
        return self.strategies["green"]
    elif gcl > 0.5:  # YELLOW (0.5-0.7 per query)
        return self.strategies["yellow"]
    else:  # RED
        return self.strategies["red"]

