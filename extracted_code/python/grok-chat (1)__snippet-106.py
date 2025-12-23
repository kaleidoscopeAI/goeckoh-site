class AutismMetrics:
    def __init__(self):
        self.attempts = 0
        self.successes = 0
        self.streak = 0

    def register_attempt(self, success):
        self.attempts += 1
        if success:
            self.successes += 1
            self.streak += 1
        else:
            self.streak = 0

    def success_rate(self):
        return self.successes / self.attempts if self.attempts else 0

