progress: Dict[str, ABAProgress] = field(default_factory=dict)
success_streak: int = 0
success_threshold: float = 0.95

def register_attempt(self, skill: str, similarity: float) -> None:
    prog = self.progress.get(skill, ABAProgress())
    prog.attempts += 1
    success = similarity >= self.success_threshold
    if success:
        prog.successes += 1
        prog.streak += 1
        prog.last_success_ts = time.time()
        self.success_streak += 1
        if prog.streak % 10 == 0:
            prog.current_level = min(5, prog.current_level + 1)
    else:
        prog.streak = 0
        self.success_streak = 0
    self.progress[skill] = prog

def get_progress_report(self) -> Dict[str, Dict]:
    report = {}
    for skill, prog in self.progress.items():
        mastery_pct = (prog.successes / max(prog.attempts, 1)) * 100
        report[skill] = {
            "mastery_percent": mastery_pct,
            "level": prog.current_level,
            "streak": prog.streak,
            "attempts": prog.attempts,
            "successes": prog.successes,
        }
    return report
