from __future__ import annotations

import asyncio
import csv
import random
from datetime import datetime
from typing import List, Optional

from .config import CompanionConfig, CONFIG
from .agent import KQBCAgent


class SpeechLoop:
    """Simulates speech processing and cognitive updates for the KQBC agent."""

    def __init__(self, config: Optional[CompanionConfig] = None, agent: Optional[KQBCAgent] = None) -> None:
        self.config = config or CONFIG
        self.agent = agent or KQBCAgent(self.config)
        # A small vocabulary.  In practice this would come from a curriculum.
        self.vocabulary: List[str] = [
            "hello",
            "water",
            "thank you",
            "help",
            "good morning",
            "yes",
            "no",
        ]
        # Probabilities for generating behavioural events on each turn
        self.behaviour_events = [
            ("anxious", 0.05),
            ("perseveration", 0.03),
            ("high_energy", 0.04),
            ("meltdown", 0.01),
            ("encouragement", 0.06),
        ]

    async def run(self) -> None:
        """Run the speech loop indefinitely."""
        self._ensure_metrics_header()
        self._ensure_guidance_header()
        while True:
            await self._simulate_one_turn()
            await asyncio.sleep(random.uniform(1.5, 3.0))

    async def _simulate_one_turn(self) -> None:
        phrase = random.choice(self.vocabulary)
        # Simulate the child's utterance by occasionally removing or altering characters
        if random.random() < 0.3:
            # mispronounce by altering a vowel or dropping a letter
            if len(phrase) > 2:
                idx = random.randint(0, len(phrase) - 1)
                if phrase[idx] in "aeiou":
                    new_char = random.choice("aeiou")
                    raw = phrase[:idx] + new_char + phrase[idx + 1 :]
                else:
                    raw = phrase[:idx] + phrase[idx + 1 :]
            else:
                raw = phrase
        else:
            raw = phrase
        needs_correction = self.agent.evaluate_correction(phrase, raw)
        corrected = phrase if needs_correction else raw
        # Update the AGI with the child's utterance (always) to build context
        self.agent.update_state(user_input=raw)

        timestamp = datetime.utcnow().isoformat()
        metrics_path = self.config.paths.metrics_csv
        with metrics_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                phrase,
                raw,
                corrected,
                "1" if needs_correction else "0",
            ])
        # Behavioural events remain random but could be tied to AGI state in future
        for event, prob in self.behaviour_events:
            if random.random() < prob:
                with self.config.paths.guidance_csv.open("a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([timestamp, event, phrase])
                break

    def _ensure_metrics_header(self) -> None:
        path = self.config.paths.metrics_csv
        if not path.exists():
            path.touch()
        if path.stat().st_size == 0:
            with path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "phrase_text",
                    "raw_text",
                    "corrected_text",
                    "needs_correction",
                ])

    def _ensure_guidance_header(self) -> None:
        path = self.config.paths.guidance_csv
        if not path.exists():
            path.touch()
        if path.stat().st_size == 0:
            with path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "event",
                    "phrase_text",
                ])

