from dataclasses import dataclass
import time
import math
from goeckoh.persistence.session_persistence import SessionLog

@dataclass
class Metrics:
    gcl: float
    stress: float
    mode_label: str
    gui_color: tuple

class CrystallineHeart:
    def __init__(self):
        self.nodes = [0.0] * 1024
        self.logger = SessionLog()
        
    def compute_metrics(self) -> Metrics:
        # Logic: Calculate Mean Absolute Energy
        total_energy = sum(abs(n) for n in self.nodes)
        avg_stress = total_energy / float(len(self.nodes))
        
        # GCL Algorithm: Inverse of Stress with Sigmoid damping
        gcl = 1.0 / (1.0 + (avg_stress * 5.0))
        
        # State Determination
        if gcl < 0.5:
            return Metrics(gcl, avg_stress, "MELTDOWN", (1.0, 0.2, 0.2, 1.0))
        elif gcl < 0.8:
            return Metrics(gcl, avg_stress, "STABILIZING", (1.0, 0.6, 0.0, 1.0))
        else:
            return Metrics(gcl, avg_stress, "FLOW", (0.0, 1.0, 1.0, 1.0))

    def process_input(self, raw_text: str):
        # 1. Semantic Mirror (Agency Correction)
        if not raw_text:
            corrected = ""
        else:
            corrected = raw_text.lower() \
                .replace("you are", "i am") \
                .replace("you", "i") \
                .replace("your", "my") \
                .capitalize()

        # 2. Physics Injection
        if raw_text:
            base_arousal = 0.1 + (len(raw_text) * 0.005)
            if "!" in raw_text: base_arousal += 0.4
            if raw_text.isupper(): base_arousal += 0.5
            
            # Injection with bounds checking
            step = 50
            for i in range(0, len(self.nodes), step):
                if i < len(self.nodes):
                    self.nodes[i] += base_arousal

        # 3. Lattice Diffusion (Time Step)
        new_nodes = [0.0] * 1024
        for i in range(1024):
            # Decay factor
            val = self.nodes[i] * 0.92
            
            # Neighbor Diffusion (Left and Right)
            left = self.nodes[(i - 1) % 1024]
            right = self.nodes[(i + 1) % 1024]
            
            # Smoothing function
            val += (left + right - (2.0 * val)) * 0.05
            new_nodes[i] = val
        self.nodes = new_nodes

        # 4. Final Output
        metrics = self.compute_metrics()
        
        if not raw_text:
            response = "Listening..."
        elif metrics.gcl < 0.5:
            response = f"I am safe. I am breathing. ({corrected})"
        else:
            response = corrected
            
        if raw_text:
            self.logger.log_interaction(raw_text, response, metrics)

        return response, metrics
