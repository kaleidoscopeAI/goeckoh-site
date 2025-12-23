import json
import time
import os

class SessionLog:
    def __init__(self, log_dir="assets"):
        self.log_path = os.path.join(log_dir, "session_history.jsonl")
        if not os.path.exists(log_dir): os.makedirs(log_dir)

    def log_interaction(self, input_text: str, output_text: str, metrics: object):
        entry = {
            "timestamp": time.time(),
            "input": input_text,
            "output": output_text,
            "gcl": metrics.gcl,
            "mode": metrics.mode_label
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[Log Error] {e}")

