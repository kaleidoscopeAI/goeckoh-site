import numpy as np

class KaleidoscopeEngine:
    """Processes nodes and extracts insights."""
    def __init__(self, gears=None):
        self.gears = gears or []

    def process(self, nodes):
        insights = []
        for node in nodes:
            output = node.act(nodes)
            insights.append(output)
        return insights
