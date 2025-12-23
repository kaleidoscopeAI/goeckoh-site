"""
Tkinter GUI for the KQBC agent.

This GUI builds on the offline companion interface by introducing an
embedded AGI status card.  Two tabs present information for the
caregiver and the child: the parent view shows a live summary of
speech attempts, an early warning indicator, and the cognitive state
extracted from the AGI substrate.  The child view remains a simple
and encouraging interface that reflects whether a correction was
suggested on the last attempt.

A background thread runs ``SpeechLoop`` which in turn updates the
underlying ``KQBCAgent``.  Periodically the GUI polls the metrics and
AGI status to refresh the display.  Everything is fully offline; no
network operations or browsers are involved.
"""

from __future__ import annotations

import asyncio
import csv
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import tkinter as tk
from tkinter import ttk

from .config import CompanionConfig, CONFIG
from .agent import KQBCAgent, AGIStatus
from .speech_loop import SpeechLoop


@dataclass
class MetricsSnapshot:
    total_attempts: int = 0
    total_corrections: int = 0
    overall_rate: float = 0.0
    last_phrase: str = ""
    last_raw: str = ""
    last_corrected: str = ""
    last_needs_correction: bool = False


def load_metrics(config: CompanionConfig) -> MetricsSnapshot:
    """Load the current metrics from disk."""
    path = config.paths.metrics_csv
    if not path.exists():
        return MetricsSnapshot()
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return MetricsSnapshot()
    total_attempts = len(rows)
    total_corrections = sum(1 for r in rows if r.get("needs_correction") in {"1", "true", "True"} )
    last = rows[-1]
    return MetricsSnapshot(
        total_attempts=total_attempts,
        total_corrections=total_corrections,
        overall_rate=(total_corrections / total_attempts) if total_attempts else 0.0,
        last_phrase=last.get("phrase_text", ""),
        last_raw=last.get("raw_text", ""),
        last_corrected=last.get("corrected_text", ""),
        last_needs_correction=last.get("needs_correction") in {"1", "true", "True"},
    )


class CompanionGUI:
    """Tk-based graphical interface for the KQBC agent."""

    def __init__(self, config: Optional[CompanionConfig] = None, agent: Optional[KQBCAgent] = None) -> None:
        self.config = config or CONFIG
        self.agent = agent or KQBCAgent(self.config)
        # Tk root
        self.root = tk.Tk()
        self.root.title(f"{self.config.child_name}'s Companion")
        self.root.geometry("960x680")
        self.root.configure(bg="#020617")

        # Styles
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

        # Notebook for parent and child views
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=12, pady=12)
        self.parent_frame = ttk.Frame(notebook, style="TFrame")
        self.child_frame = ttk.Frame(notebook, style="TFrame")
        notebook.add(self.parent_frame, text="Parent View")
        notebook.add(self.child_frame, text="Child View")

        # Build UI sections
        self._build_parent_view()
        self._build_child_view()

        # Speech loop runs in background thread and writes logs
        self._loop = SpeechLoop(self.config, agent=self.agent)
        self._loop_thread = threading.Thread(target=self._run_speech_loop, daemon=True)
        self._loop_thread.start()

        # Refresh UI periodically
        self.refresh_interval_ms = 2500
        self.root.after(self.refresh_interval_ms, self._refresh_ui)

    # Parent view construction
    def _build_parent_view(self) -> None:
        pf = self.parent_frame

        # Header
        caregiver = getattr(self.config, "caregiver_name", "Caregiver")
        title = ttk.Label(
            pf,
            text=f"{caregiver}'s dashboard for {self.config.child_name}",
            style="Title.TLabel",
        )
        title.pack(anchor="w", pady=(0, 4))
        subtitle = ttk.Label(
            pf,
            text="Live view of practice, corrections, early warnings and cognition.",
            style="Body.TLabel",
        )
        subtitle.pack(anchor="w", pady=(0, 8))

        # Stats row
        stats = ttk.Frame(pf, style="TFrame")
        stats.pack(fill="x", pady=(0, 8))
        self.total_attempts_var = tk.StringVar(value="0")
        self.overall_rate_var = tk.StringVar(value="0.0 %")
        self._make_stat(stats, "Total attempts", self.total_attempts_var).pack(side="left", padx=(0, 20))
        self._make_stat(stats, "Overall correction rate", self.overall_rate_var).pack(side="left")

        # Meltdown risk card (static for now)
        risk_card = ttk.Frame(pf, style="TFrame", relief="groove")
        risk_card.pack(fill="x", pady=(0, 8))
        ttk.Label(risk_card, text="Meltdown risk (last 50 events)", style="Risk.TLabel").pack(anchor="w", padx=8, pady=(6, 2))
        self.meltdown_level_var = tk.StringVar(value="No data yet")
        self.meltdown_score_var = tk.StringVar(value="0 %")
        self.meltdown_message_var = tk.StringVar(
            value="No behavioural events logged yet. This card will display early warnings."
        )
        risk_row = ttk.Frame(risk_card, style="TFrame")
        risk_row.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Label(risk_row, textvariable=self.meltdown_level_var, style="Risk.TLabel").pack(side="left", padx=(0, 8))
        ttk.Label(risk_row, textvariable=self.meltdown_score_var, style="Body.TLabel").pack(side="left")
        self.meltdown_bar = ttk.Progressbar(risk_card, orient="horizontal", mode="determinate", maximum=100)
        self.meltdown_bar.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Label(
            risk_card,
            textvariable=self.meltdown_message_var,
            style="Body.TLabel",
            wraplength=900,
            justify="left",
        ).pack(fill="x", padx=8, pady=(0, 8))

        # Cognitive state card
        cog_card = ttk.Frame(pf, style="TFrame", relief="groove")
        cog_card.pack(fill="x", pady=(0, 8))
        ttk.Label(cog_card, text="Cognitive state", style="Risk.TLabel").pack(anchor="w", padx=8, pady=(6, 2))
        # Vars for AGI state
        self.freq_var = tk.StringVar(value="0.0 GHz")
        self.temp_var = tk.StringVar(value="0.0 °C")
        self.DA_var = tk.StringVar(value="0.00")
        self.Ser_var = tk.StringVar(value="0.00")
        self.NE_var = tk.StringVar(value="0.00")
        self.coherence_var = tk.StringVar(value="0.000")
        self.awareness_var = tk.StringVar(value="0.000")
        self.phi_var = tk.StringVar(value="0.000")
        # Layout as grid
        grid = ttk.Frame(cog_card, style="TFrame")
        grid.pack(fill="x", padx=8, pady=(0, 8))
        def add_row(label: str, var: tk.StringVar, row: int, col: int) -> None:
            ttk.Label(grid, text=label, style="StatLabel.TLabel").grid(row=row, column=col*2, sticky="w", padx=(0, 4), pady=2)
            ttk.Label(grid, textvariable=var, style="StatValue.TLabel").grid(row=row, column=col*2+1, sticky="w", padx=(0, 10), pady=2)
        add_row("Freq:", self.freq_var, 0, 0)
        add_row("Temp:", self.temp_var, 0, 1)
        add_row("DA:", self.DA_var, 1, 0)
        add_row("Ser:", self.Ser_var, 1, 1)
        add_row("NE:", self.NE_var, 2, 0)
        add_row("Coherence:", self.coherence_var, 2, 1)
        add_row("Awareness:", self.awareness_var, 3, 0)
        add_row("Phi:", self.phi_var, 3, 1)

        # Live session snapshot
        last_card = ttk.Frame(pf, style="TFrame", relief="groove")
        last_card.pack(fill="both", expand=True, pady=(0, 0))
        ttk.Label(last_card, text="Live session snapshot", style="Risk.TLabel").pack(anchor="w", padx=8, pady=(6, 2))
        self.last_phrase_var = tk.StringVar(value="—")
        self.last_raw_var = tk.StringVar(value="—")
        self.last_corrected_var = tk.StringVar(value="—")
        self.last_status_var = tk.StringVar(value="Waiting for first attempt…")
        ttk.Label(last_card, text="Target phrase:", style="StatLabel.TLabel").pack(anchor="w", padx=8, pady=(4, 0))
        ttk.Label(last_card, textvariable=self.last_phrase_var, style="Body.TLabel").pack(anchor="w", padx=16, pady=(0, 4))
        ttk.Label(last_card, text="Child said:", style="StatLabel.TLabel").pack(anchor="w", padx=8, pady=(4, 0))
        ttk.Label(last_card, textvariable=self.last_raw_var, style="Body.TLabel", wraplength=900).pack(anchor="w", padx=16, pady=(0, 4))
        ttk.Label(last_card, text="Companion model:", style="StatLabel.TLabel").pack(anchor="w", padx=8, pady=(4, 0))
        ttk.Label(last_card, textvariable=self.last_corrected_var, style="Body.TLabel", wraplength=900).pack(anchor="w", padx=16, pady=(0, 4))
        ttk.Label(last_card, textvariable=self.last_status_var, style="Body.TLabel").pack(anchor="w", padx=8, pady=(6, 8))

    def _make_stat(self, parent: ttk.Frame, label: str, var: tk.StringVar) -> ttk.Frame:
        f = ttk.Frame(parent, style="TFrame")
        ttk.Label(f, text=label, style="StatLabel.TLabel").pack(anchor="w")
        ttk.Label(f, textvariable=var, style="StatValue.TLabel").pack(anchor="w")
        return f

    # Child view
    def _build_child_view(self) -> None:
        cf = self.child_frame
        greet = ttk.Label(cf, text=f"Hi {self.config.child_name} 👋", style="Title.TLabel")
        greet.pack(anchor="w", pady=(0, 4))
        self.child_status_var = tk.StringVar(value="Listening for your voice…")
        status = ttk.Label(cf, textvariable=self.child_status_var, style="Body.TLabel")
        status.pack(anchor="w", pady=(0, 8))
        phrase_card = ttk.Frame(cf, style="TFrame", relief="groove")
        phrase_card.pack(fill="both", expand=True, pady=(0, 0))
        self.child_helper_var = tk.StringVar(value="Say it like this")
        self.child_phrase_var = tk.StringVar(value="Waiting for the first try…")
        self.child_encouragement_var = tk.StringVar(
            value="When you talk, I listen. If it’s tricky, we’ll try together."
        )
        ttk.Label(phrase_card, textvariable=self.child_helper_var, style="StatLabel.TLabel").pack(anchor="w", padx=8, pady=(6, 0))
        ttk.Label(phrase_card, textvariable=self.child_phrase_var, style="StatValue.TLabel", wraplength=900).pack(anchor="w", padx=16, pady=(4, 8))
        ttk.Label(phrase_card, textvariable=self.child_encouragement_var, style="Body.TLabel", wraplength=900).pack(anchor="w", padx=16, pady=(0, 8))

    # Background speech loop
    def _run_speech_loop(self) -> None:
        async def runner():
            await self._loop.run()
        asyncio.run(runner())

    # Periodic refresh
    def _refresh_ui(self) -> None:
        # Update metrics
        metrics = load_metrics(self.config)
        self.total_attempts_var.set(str(metrics.total_attempts))
        self.overall_rate_var.set(f"{metrics.overall_rate * 100.0:.1f} %")
        self.last_phrase_var.set(metrics.last_phrase or "—")
        self.last_raw_var.set(metrics.last_raw or "—")
        self.last_corrected_var.set(metrics.last_corrected or "—")
        if metrics.total_attempts == 0:
            self.last_status_var.set("Waiting for first attempt…")
        elif metrics.last_needs_correction:
            self.last_status_var.set("Correction suggested on the last attempt.")
        else:
            self.last_status_var.set("Last attempt sounded great.")
        # Update AGI status
        status = self.agent.get_status()
        self.freq_var.set(f"{status.freq_GHz:.2f} GHz")
        self.temp_var.set(f"{status.temp_C:.2f} °C")
        self.DA_var.set(f"{status.DA:.2f}")
        self.Ser_var.set(f"{status.Ser:.2f}")
        self.NE_var.set(f"{status.NE:.2f}")
        self.coherence_var.set(f"{status.coherence:.3f}")
        self.awareness_var.set(f"{status.awareness:.3f}")
        self.phi_var.set(f"{status.phi_proxy:.3f}")
        # Update child view
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
                phrase = metrics.last_corrected or metrics.last_phrase or "Let's try together."
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
        # For now, the meltown card remains static.  In a full system this
        # could be driven by AGI state as well.
        self.root.after(self.refresh_interval_ms, self._refresh_ui)

    def run(self) -> None:
        self.root.mainloop()


def run_gui(config: Optional[CompanionConfig] = None) -> None:
    gui = CompanionGUI(config=config or CONFIG)
    gui.run()