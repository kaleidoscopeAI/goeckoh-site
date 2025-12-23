import json
import time
import os

class SessionLog:
    def __init__(self, log_dir="assets"):
        self.log_path = os.path.join(log_dir, "session_history.jsonl")
        
        # Recursive creation to ensure path exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def log_interaction(self, input_text: str, output_text: str, metrics: object):
        # Handle different metrics formats
        if hasattr(metrics, 'mode_label'):
            mode = str(metrics.mode_label)
        elif hasattr(metrics, 'mode'):
            mode = str(metrics.mode)
        else:
            mode = "UNKNOWN"
            
        entry = {
            "timestamp": time.time(),
            "input": str(input_text),
            "output": str(output_text),
            "gcl": float(metrics.gcl),
            "stress": float(metrics.stress),
            "mode": mode
        }
        
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except IOError as e:
            print(f"[LOG CRITICAL] Could not write to persistence layer: {e}")