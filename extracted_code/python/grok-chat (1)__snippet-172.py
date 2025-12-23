def __init__(self):
    self.strategies = {
        "green": "Full guidance: Let's explore that idea together.",
        "yellow": "Simple affirm: You're on the right track.",
        "red": "Calm mode: Take a breath... all is well."
    }

def execute(self, text, gcl):
    if gcl > 0.7:
        return self.strategies["green"]
    elif gcl > 0.4:
        return self.strategies["yellow"]
    else:
        return self.strategies["red"]

