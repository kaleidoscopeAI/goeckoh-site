def compute_meltdown_risk(summary: BehaviorSummary) -> MeltdownRisk:
    total = summary.total
    if total == 0:
        return MeltdownRisk(
            score=0,
            level="No data yet",
            message=(
                "No behaviour events logged yet. As the system runs, this card will "
                "show early warning signals."
            ),
        )
    # Weighted scoring inspired by AGI stress metrics: emphasise serious events
    score_raw = (
        15 * summary.anxious
        + 12 * summary.perseveration
        + 10 * summary.high_energy
        + 25 * summary.meltdown
    )
    normalized = min(100.0, (score_raw / (max(total, 1) * 25.0)) * 100.0)
    score = int(round(normalized))
    if score < 25:
        level = "Low"
        message = "Signals look calm overall. Keep routines steady and keep praising effort."
    elif score < 60:
        level = "Elevated"
        message = (
            "Some stress signals showing up. Consider a short movement or breathing break."
        )
    else:
        level = "High"
        message = (
            "Frequent stress signals. Stay close, simplify demands, and lean on go-to calming tools."
        )
    return MeltdownRisk(score=score, level=level, message=message)


class CompanionGUI:
    """Main window containing parent and child views."""

    def __init__(self, config: Optional[CompanionConfig] = None) -> None:
        self.config = config or CONFIG
        # Tk root
        self.root = tk.Tk()
        self.root.title(f"{self.config.child_name}'s Companion")
        self.root.geometry("960x640")
        self.root.configure(bg="#020617")
        # Style definitions using ttk
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#020617")
        style.configure("Card.TFrame", background="#020617")
        style.configure(
            "Title.TLabel",
            background="#020617",
            foreground="#e5e7eb",
            font=("Segoe UI", 16, "bold"),
        )
        style.configure(
            "Body.TLabel",
            background="#020617",
            foreground="#cbd5f5",
            font=("Segoe UI", 10),
        )
        style.configure(
            "StatLabel.TLabel",
            background="#020617",
            foreground="#9ca3af",
            font=("Segoe UI", 9),
        )
        style.configure(
            "StatValue.TLabel",
            background="#020617",
            foreground="#e5e7eb",
            font=("Segoe UI", 12, "bold"),
        )
        style.configure(
            "Risk.TLabel",
            background="#020617",
            foreground="#e5e7eb",
            font=("Segoe UI", 11, "bold"),
        )
        # Notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=12, pady=12)
        self.parent_frame = ttk.Frame(notebook, style="TFrame")
        self.child_frame = ttk.Frame(notebook, style="TFrame")
        notebook.add(self.parent_frame, text="Parent View")
        notebook.add(self.child_frame, text="Child View")
        # Build UI components
        self._build_parent_view()
        self._build_child_view()
        # Speech loop in background thread
        self._loop = SpeechLoop(self.config)
        self._loop_thread = threading.Thread(target=self._run_speech_loop, daemon=True)
        self._loop_thread.start()
        # Refresh interval (ms)
        self._refresh_interval_ms = 2500
        self.root.after(self._refresh_interval_ms, self._refresh_ui)

    # ---------------- Parent view ----------------

    def _build_parent_view(self) -> None:
        pf = self.parent_frame
        # Header
        header = ttk.Frame(pf, style="TFrame")
        header.pack(fill="x", pady=(0, 12))
        caregiver = getattr(self.config, "caregiver_name", "Caregiver")
        title = ttk.Label(
            header,
            text=f"{caregiver}'s dashboard for {self.config.child_name}",
            style="Title.TLabel",
        )
        title.pack(anchor="w")
        subtitle = ttk.Label(
            header,
            text=(
                "Live view of practice, corrections, and calming support — "
                "without needing a browser."
            ),
            style="Body.TLabel",
        )
        subtitle.pack(anchor="w", pady=(4, 0))
        # Stats row
        stats = ttk.Frame(pf, style="TFrame")
        stats.pack(fill="x", pady=(8, 12))
        self.total_attempts_var = tk.StringVar(value="0")
        self.overall_rate_var = tk.StringVar(value="0.0 %")
        self._make_stat_block(stats, "Total attempts", self.total_attempts_var).pack(
            side="left", padx=(0, 16)
        )
        self._make_stat_block(
            stats, "Overall correction rate", self.overall_rate_var
        ).pack(side="left", padx=(0, 16))
        # Meltdown risk card
        risk_card = ttk.Frame(pf, style="TFrame", relief="groove")
        risk_card.pack(fill="x", pady=(4, 12))
        risk_title = ttk.Label(
            risk_card,
            text="Meltdown risk (last 50 events)",
            style="Risk.TLabel",
        )
        risk_title.pack(anchor="w", padx=8, pady=(6, 2))
        self.meltdown_level_var = tk.StringVar(value="No data yet")
        self.meltdown_score_var = tk.StringVar(value="0 %")
        self.meltdown_message_var = tk.StringVar(
            value=(
                "No behaviour events logged yet. As the system runs, this card will "
                "show early warning signals."
            )
        )
        risk_row = ttk.Frame(risk_card, style="TFrame")
        risk_row.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Label(
            risk_row, textvariable=self.meltdown_level_var, style="Risk.TLabel"
        ).pack(side="left", padx=(0, 8))
        ttk.Label(
            risk_row, textvariable=self.meltdown_score_var, style="Body.TLabel"
        ).pack(side="left")
        self.meltdown_bar = ttk.Progressbar(
            risk_card, orient="horizontal", mode="determinate", maximum=100
        )
        self.meltdown_bar.pack(fill="x", padx=8, pady=(0, 4))
        self.meltdown_message_label = ttk.Label(
            risk_card,
            textvariable=self.meltdown_message_var,
            style="Body.TLabel",
            wraplength=800,
            justify="left",
        )
        self.meltdown_message_label.pack(fill="x", padx=8, pady=(0, 8))
        # Last attempt card
        last_card = ttk.Frame(pf, style="TFrame", relief="groove")
        last_card.pack(fill="both", expand=True, pady=(4, 0))
        ttk.Label(
            last_card,
            text="Live session snapshot",
            style="Risk.TLabel",
        ).pack(anchor="w", padx=8, pady=(6, 2))
        self.last_phrase_var = tk.StringVar(value="—")
        self.last_raw_var = tk.StringVar(value="—")
        self.last_corrected_var = tk.StringVar(value="—")
        self.last_status_var = tk.StringVar(value="Waiting for first attempt...")
        ttk.Label(
            last_card, text="Target phrase:", style="StatLabel.TLabel"
        ).pack(anchor="w", padx=8, pady=(4, 0))
        ttk.Label(
            last_card, textvariable=self.last_phrase_var, style="Body.TLabel"
        ).pack(anchor="w", padx=16, pady=(0, 4))
        ttk.Label(
            last_card, text="Child said:", style="StatLabel.TLabel"
        ).pack(anchor="w", padx=8, pady=(4, 0))
        ttk.Label(
            last_card,
            textvariable=self.last_raw_var,
            style="Body.TLabel",
            wraplength=800,
        ).pack(anchor="w", padx=16, pady=(0, 4))
        ttk.Label(
            last_card, text="Companion model:", style="StatLabel.TLabel"
        ).pack(anchor="w", padx=8, pady=(4, 0))
        ttk.Label(
            last_card,
            textvariable=self.last_corrected_var,
            style="Body.TLabel",
            wraplength=800,
        ).pack(anchor="w", padx=16, pady=(0, 4))
        ttk.Label(
            last_card,
            textvariable=self.last_status_var,
            style="Body.TLabel",
        ).pack(anchor="w", padx=8, pady=(6, 8))

    def _make_stat_block(self, parent: ttk.Frame, label: str, value_var: tk.StringVar) -> ttk.Frame:
        f = ttk.Frame(parent, style="TFrame")
        ttk.Label(f, text=label, style="StatLabel.TLabel").pack(anchor="w")
        ttk.Label(f, textvariable=value_var, style="StatValue.TLabel").pack(
            anchor="w"
        )
        return f

    # ---------------- Child view ----------------

    def _build_child_view(self) -> None:
        cf = self.child_frame
        top = ttk.Frame(cf, style="TFrame")
        top.pack(fill="x", pady=(0, 12))
        self.child_status_var = tk.StringVar(value="Listening for your voice…")
        greet = ttk.Label(
            top, text=f"Hi {self.config.child_name} 👋", style="Title.TLabel"
        )
        greet.pack(anchor="w")
        status = ttk.Label(top, textvariable=self.child_status_var, style="Body.TLabel")
        status.pack(anchor="w", pady=(4, 0))
        phrase_card = ttk.Frame(cf, style="TFrame", relief="groove")
        phrase_card.pack(fill="both", expand=True, pady=(4, 0))
        self.child_helper_var = tk.StringVar(value="Say it like this")
        self.child_phrase_var = tk.StringVar(value="Waiting for the first try…")
        self.child_encouragement_var = tk.StringVar(
            value="When you talk, I listen. If it’s tricky, we’ll try together."
        )
        ttk.Label(
            phrase_card, textvariable=self.child_helper_var, style="StatLabel.TLabel"
        ).pack(anchor="w", padx=8, pady=(6, 0))
        ttk.Label(
            phrase_card,
            textvariable=self.child_phrase_var,
            style="StatValue.TLabel",
            wraplength=800,
        ).pack(anchor="w", padx=16, pady=(4, 8))
        ttk.Label(
            phrase_card,
            textvariable=self.child_encouragement_var,
            style="Body.TLabel",
            wraplength=800,
        ).pack(anchor="w", padx=16, pady=(0, 8))

    # ---------------- Background speech loop ----------------

    def _run_speech_loop(self) -> None:
        async def _runner():
            await self._loop.run()
        asyncio.run(_runner())

    # ---------------- Periodic refresh ----------------

    def _refresh_ui(self) -> None:
        metrics = load_metrics(self.config)
        guidance = load_guidance(self.config)
        summary = compute_behavior_summary(guidance)
        risk = compute_meltdown_risk(summary)
        # Parent stats
        self.total_attempts_var.set(str(metrics.total_attempts))
        self.overall_rate_var.set(f"{metrics.overall_rate * 100.0:.1f} %")
        self.meltdown_level_var.set(f"{risk.level} risk")
        self.meltdown_score_var.set(f"{risk.score} %")
        self.meltdown_message_var.set(risk.message)
        self.meltdown_bar["value"] = risk.score
        # Last attempt
        self.last_phrase_var.set(metrics.last_phrase or "—")
        self.last_raw_var.set(metrics.last_raw or "—")
        self.last_corrected_var.set(metrics.last_corrected or "—")
        if metrics.total_attempts == 0:
            self.last_status_var.set("Waiting for first attempt...")
        elif metrics.last_needs_correction:
            self.last_status_var.set("Correction suggested on the last attempt.")
        else:
            self.last_status_var.set("Last attempt sounded great.")
        # Child view
        if metrics.total_attempts == 0:
            self.child_status_var.set("Listening for your voice…")
            self.child_helper_var.set("Say it like this")
            self.child_phrase_var.set("Waiting for the first try…")
            self.child_encouragement_var.set(
                "When you talk, I listen. If it’s tricky, we’ll try together."
            )
        else:
            if metrics.last_needs_correction:
                self.child_status_var.set("Let's try it like this.")
                self.child_helper_var.set("Say it like this")
                phrase = (
                    metrics.last_corrected or metrics.last_phrase or "Let's try together."
                )
                self.child_phrase_var.set(phrase)
                self.child_encouragement_var.set(
                    "It’s okay if it’s hard. We’ll slow down and say it together."
                )
            else:
                self.child_status_var.set("That sounded great!")
                self.child_helper_var.set("You said it!")
                phrase = metrics.last_raw or metrics.last_phrase or "Nice job."
                self.child_phrase_var.set(phrase)
                self.child_encouragement_var.set(
                    "Nice work. Take a breath, smile, and get ready for the next word."
                )
        self.root.after(self._refresh_interval_ms, self._refresh_ui)

    def run(self) -> None:
        """Start the Tk main loop."""
        self.root.mainloop()


def run_gui(config: Optional[CompanionConfig] = None) -> None:
    gui = CompanionGUI(config or CONFIG)
    gui.run()

