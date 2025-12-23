import numpy as np

class PerspectiveEngine:
    """Generates alternative speculative perspectives."""
    def process(self, nodes):
        speculative = []
        for node in nodes:
            val = node.act(nodes) * 1.1  # slight perturbation for speculation
            speculative.append(val)
        return speculative
