import numpy as np

class PatternRecognition:
    def __init__(self):
        self.patterns = []

    def identify_patterns(self, data):
        """Analyze data and extract patterns."""
        numerical_data = [value for value in data.values() if isinstance(value, (int, float))]
        pattern = {
            "mean": np.mean(numerical_data),
            "std_dev": np.std(numerical_data),
            "max": max(numerical_data),
            "min": min(numerical_data)
        }
        self.patterns.append(pattern)
        return pattern

    def match_pattern(self, new_data):
        """Match new data to existing patterns."""
        for pattern in self.patterns:
            if abs(np.mean(new_data) - pattern["mean"]) < pattern["std_dev"]:
                return True
        return False

