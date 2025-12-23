"""ABA skill progress tracking from documents"""
attempts: int = 0
successes: int = 0
last_attempt_ts: float = 0.0
current_level: int = 1  # 1: Basic, 2: Partial, 3: Independent

