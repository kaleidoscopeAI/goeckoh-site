import numpy as np

class AttentionModule:
    """Adjust learning/attention based on local Phi"""
    def adjust_learning(self, phi_local):
        base_rate = 0.01
        return base_rate * (1 + phi_local)
