def __init__(self, config: CompanionConfig):
    self.config = config
    self.root = tk.Tk()
    self.root.title(f"{config.child_name}'s Companion Dashboard")

    self._refresh_interval_ms = 3000

    self.metrics_logger = MetricsLogger(config.paths.metrics_csv)
    self.guidance_logger = GuidanceLogger(config.paths.guidance_csv)
    self.aba = ABAEngine()

    # Parent view
    tk.Label(self.root, text="Parent Dashboard").pack()

    self.attempts_var = tk.StringVar(value="Attempts: 0")
    tk.Label(self.root, textvariable=self.attempts_var).pack()

    self.success_rate_var = tk.StringVar(value="Success Rate: 0%")
    tk.Label(self.root, textvariable=self.success_rate_var).pack()

    self.streak_var = tk.StringVar(value="Success Streak: 0")
    tk.Label(self.root, textvariable=self.streak_var).pack()

    self.last_raw_var = tk.StringVar(value="Last Raw: —")
    tk.Label(self.root, textvariable=self.last_raw_var).pack()

    self.last_corrected_var = tk.StringVar(value="Last Corrected: —")
    tk.Label(self.root, textvariable=self.last_corrected_var).pack()

    self.last_status_var = tk.StringVar(value="Status: Waiting...")
    tk.Label(self.root, textvariable=self.last_status_var).pack()

    # Child view
    tk.Label(self.root, text="Child View").pack()

    self.child_status_var = tk.StringVar(value="Listening...")
    tk.Label(self.root, textvariable=self.child_status_var).pack()

    self.child_phrase_var = tk.StringVar(value="Waiting...")
    tk.Label(self.root, textvariable=self.child_phrase_var).pack()

    self.child_encouragement_var = tk.StringVar(value="Let's try together.")
    tk.Label(self.root, textvariable=self.child_encouragement_var).pack()

    self._refresh_ui()

def _refresh_ui(self) -> None:
    metrics = self.metrics_logger.read_latest()
    aba_report = self.aba.get_progress_report()

    self.attempts_var.set(f"Attempts: {metrics.total_attempts}")
    self.success_rate_var.set(f"Success Rate: {metrics.success_rate * 100:.1f}%")
    self.streak_var.set(f"Success Streak: {metrics.success_streak}")
    self.last_raw_var.set(metrics.last_raw or "—")
    self.last_corrected_var.set(metrics.last_corrected or "—")
    self.last_status_var.set("Last attempt good!" if not metrics.last_needs_correction else "Correction suggested.")

    # Child view update based on metrics
    if metrics.total_attempts == 0:
        self.child_status_var.set("Listening for your voice…")
        self.child_phrase_var.set("Waiting for the first try…")
        self.child_encouragement_var.set("When you talk, I listen.")
    else:
        if metrics.last_needs_correction:
            self.child_status_var.set("Let's try it like this.")
            phrase = metrics.last_corrected or "Let's try together."
            self.child_phrase_var.set(phrase)
            self.child_encouragement_var.set("It’s okay if it’s hard. We’ll slow down.")
        else:
            self.child_status_var.set("That sounded great!")
            phrase = metrics.last_raw or "Nice job."
            self.child_phrase_var.set(phrase)
            self.child_encouragement_var.set("Nice work. Take a breath.")

    self.root.after(self._refresh_interval_ms, self._refresh_ui)

def run(self) -> None:
    self.root.mainloop()

