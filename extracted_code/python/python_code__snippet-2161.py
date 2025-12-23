def __init__(self, log_dir="assets"):
    self.log_path = os.path.join(log_dir, "session_history.jsonl")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

def log_interaction(self, input_text: str, output_text: str, metrics: object):
    """
    Saves a discrete interaction frame to the append-only log.
    """
    entry = {
        "timestamp": time.time(),
        "human_input": input_text,
        "system_output": output_text,
        "telemetry": {
            "gcl": metrics.gcl,
            "stress_level": metrics.stress,
            "system_mode": metrics.mode_label
        }
    }

    try:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[Persistence Error]: {e}")

def get_session_summary(self):
    """
    Reads logs to provide start-up context.
    (Could be expanded to generate reports)
    """
    count = 0
    avg_gcl = 0.0
    try:
        if not os.path.exists(self.log_path): return 0, 0

        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            count = len(lines)
            # rudimentary stat parsing
            # in prod, perform pandas analysis here
    except:
        pass

    return count

