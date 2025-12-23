    class SessionLog:
        """Session persistence fallback (inline implementation)."""
        
        def __init__(self, log_dir="assets"):
            self.log_path = os.path.join(log_dir, "session_history.jsonl")
            
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        
        def log_interaction(self, input_text: str, output_text: str, metrics: SystemMetrics):
            """Log interaction to persistent storage"""
            entry = {
                "timestamp": time.time(),
                "input": str(input_text),
                "output": str(output_text),
                "gcl": float(metrics.gcl),
                "stress": float(metrics.stress),
                "mode": str(metrics.mode)
            }
            
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            except IOError as e:
                print(f"[LOG CRITICAL] Could not write to persistence layer: {e}")
