"""Async simulator that mirrors the legacy speech_loop scripts.

This loop does not access microphones. Instead it generates sample
phrases, optionally routes them through ``KQBCAgent`` to update the
cognitive substrate, and appends CSV logs identical to the real
companion. Behavioural events are sampled from simple probabilities to
keep downstream dashboards exercised during demos/tests.
"""

def __init__(
    self,
    config: Optional[CompanionConfig] = None,
    *,
    agent: Optional[KQBCAgent] = None,
    vocabulary: Optional[Sequence[str]] = None,
    behaviour_events: Optional[Sequence[Tuple[str, float]]] = None,
    use_agent: bool = True,
) -> None:
    self.config = config or CompanionConfig()
    self.agent = agent if agent is not None else (KQBCAgent(self.config) if use_agent else None)
    self.vocabulary: List[str] = list(
        vocabulary
        or (
            "hello",
            "water",
            "thank you",
            "help",
            "good morning",
            "yes",
            "no",
        )
    )
    self.behaviour_events: List[Tuple[str, float]] = list(
        behaviour_events
        or (
            ("anxious", 0.05),
            ("perseveration", 0.03),
            ("high_energy", 0.04),
            ("meltdown", 0.01),
            ("encouragement", 0.06),
        )
    )

async def run(self) -> None:
    """Run indefinitely, writing metrics + guidance rows."""

    self._ensure_metrics_header()
    self._ensure_guidance_header()
    while True:
        await self._simulate_one_turn()
        await asyncio.sleep(random.uniform(1.5, 3.0))

async def _simulate_one_turn(self) -> None:
    phrase = random.choice(self.vocabulary)
    raw = self._maybe_mutate_phrase(phrase)

    if self.agent:
        needs_correction = self.agent.evaluate_correction(phrase, raw)
        corrected = phrase if needs_correction else raw
        self.agent.update_state(user_input=raw)
    else:
        needs_correction = raw != phrase
        corrected = phrase if needs_correction else raw

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

    for event, prob in self.behaviour_events:
        if random.random() < prob:
            with self.config.paths.guidance_csv.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, event, phrase])
            break

def _maybe_mutate_phrase(self, phrase: str) -> str:
    if random.random() >= 0.3:
        return phrase
    if len(phrase) <= 2:
        return phrase
    idx = random.randint(0, len(phrase) - 1)
    if phrase[idx] in "aeiou":
        new_char = random.choice("aeiou")
        return phrase[:idx] + new_char + phrase[idx + 1 :]
    return phrase[:idx] + phrase[idx + 1 :]

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
