import random
import math
import time
from dataclasses import dataclass

# Import Persistence Layer
from goeckoh.persistence.session_persistence import SessionLog

# --- METRICS STRUCTURE ---
@dataclass
class Metrics:
    gcl: float
    stress: float
    mode_label: str
    gui_color: tuple

class CrystallineHeart:
    def __init__(self):
        self.nodes = [0.0] * 1024    # Physics Lattice
        self.temperature = 0.0       # Annealing Temp
        self.logger = SessionLog()   # Long-term Memory
        
        print("[SYSTEM] Logic Core + Physics Engine Online")

    def compute_metrics(self) -> Metrics:
        """Calculates GCL based on 128-equation lattice state."""
        # 1. Average Lattice Energy (Entropy/Stress proxy)
        avg_stress = sum(abs(n) for n in self.nodes) / len(self.nodes)
        
        # 2. Global Coherence Level (0.0 to 1.0)
        gcl = 1.0 / (1.0 + (avg_stress * 5.0))
        
        # 3. Gate Logic / State Machine
        if gcl < 0.5:
            label = "MELTDOWN" # Red Zone: System locks AI, enables calming
            color = (1.0, 0.2, 0.2, 1)
        elif gcl < 0.8:
            label = "STABILIZING" # Orange Zone: Limited features
            color = (1.0, 0.6, 0.0, 1)
        else:
            label = "FLOW" # Cyan Zone: Full capabilities
            color = (0.0, 1.0, 1.0, 1)

        return Metrics(gcl, avg_stress, label, color)

    def process_input(self, raw_text: str):
        """
        The Master Loop. Accepts text (or empty string for idle updates),
        updates physics, logs data, and returns response.
        """
        
        # --- 1. SEMANTIC MIRROR (AGENCY CORRECTION) ---
        # Transforms 'You/Your' to 'I/My' to foster ownership
        if raw_text.strip():
            corrected = raw_text.lower() \
                .replace("you are", "i am") \
                .replace("you", "i") \
                .replace("your", "my") \
                .capitalize()
        else:
            corrected = ""
            
        # --- 2. ENERGY INJECTION (PHYSICS) ---
        if raw_text:
            # Calculate input energy based on length and punctuation
            arousal = 0.05 + (len(raw_text) * 0.01)
            if "!" in raw_text: arousal += 0.3
            if raw_text.isupper(): arousal += 0.5
            
            # Inject into Lattice (Diffused injection)
            for i in range(0, len(self.nodes), 50):
                self.nodes[i] += arousal
        
        # --- 3. LATTICE DYNAMICS (TIME STEP) ---
        for i in range(len(self.nodes)):
            # Damping/Decay (Stabilization)
            self.nodes[i] *= 0.9 
            
            # Diffusion (Neighbor coupling - Wave propagation)
            neighbor = self.nodes[(i - 1) % len(self.nodes)]
            self.nodes[i] += (neighbor - self.nodes[i]) * 0.1

        metrics = self.compute_metrics()
        
        # --- 4. GATING LOGIC (RESPONSE GENERATION) ---
        if raw_text:
            if metrics.gcl < 0.5:
                # High Stress Response: Robotic, Simple, Calming
                response_text = f"I am safe. I am breathing. ({corrected})"
            else:
                # Normal Response: Reflection
                response_text = corrected
        else:
            # Idle / System status update
            response_text = "Listening..."

        # --- 5. MEMORY LOGGING ---
        # Log meaningful events to local JSON-L database
        if raw_text.strip():
            self.logger.log_interaction(raw_text, response_text, metrics)

        return response_text, metrics
