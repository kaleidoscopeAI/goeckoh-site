"""Voice guidance that turns the companion into a calming friend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .advanced_voice_mimic import VoiceCrystal
from .audio_io import AudioIO
from .data_store import DataStore
from .heart_core import enforce_first_person


@dataclass(frozen=True)
class GuidanceScript:
    event: str
    title: str
    message: str
    style: str = "neutral"  # Default voice style
    breathing_steps: Optional[List[str]] = None


GUIDANCE_SCRIPTS: Dict[str, GuidanceScript] = {
    "anxious": GuidanceScript(
        event="anxious",
        title="Calm Breathing Buddy",
        message="Let's pause together. We will take three slow breaths.",
        style="calm",
        breathing_steps=[
            "Breath one: in through your nose for four, out for six.",
            "Breath two: shoulders down, think of your favorite safe place.",
            "Breath three: whisper your favorite color as you exhale.",
        ],
    ),
    "perseveration": GuidanceScript(
        event="perseveration",
        title="New Adventure Prompt",
        message="I hear that phrase a lot. Want to try a silly switch up together?",
        style="neutral",
        breathing_steps=[
            "Let's count five dinosaurs or cars before we try again.",
        ],
    ),
    "high_energy": GuidanceScript(
        event="high_energy",
        title="Movement Reset",
        message="Sounds like lots of energy! Let's stretch arms up high, shake them out, and wiggle toes.",
        style="calm",
    ),
    "encouragement": GuidanceScript(
        event="encouragement",
        title="Cheer Squad",
        message="Nice work! I love how you said that. Ready for the next word when you are.",
        style="excited",
    ),
    "caregiver_reset": GuidanceScript(
        event="caregiver_reset",
        title="Caregiver Breath",
        message="This is your reminder to take a sip of water, drop your shoulders, and breathe. You’re doing great.",
        style="calm",
    ),
}


class GuidanceCoach:
    """Delivers friendly prompts via voice and logs the event."""

    def __init__(self, voice: VoiceCrystal, audio_io: AudioIO, data_store: DataStore) -> None:
        self.voice = voice
        self.audio_io = audio_io
        self.data_store = data_store

    def speak(self, event: str, override_text: str | None = None) -> None:
        script = GUIDANCE_SCRIPTS.get(event)
        if not script and not override_text:
            return

        text = enforce_first_person(override_text or self._build_message(script))
        style = script.style if script else "neutral"
        # Inner-style delivery reduces load during anxious/high-energy states.
        mode = "inner" if event in {"anxious", "high_energy", "perseveration"} else "outer"
        audio = self.voice.speak(text, style=style, mode=mode)
        if audio.size > 0:
            self.audio_io.play(audio)
        self.data_store.log_guidance_event(event, (script.title if script else "Custom"), text)


    def _build_message(self, script: GuidanceScript | None) -> str:
        if not script:
            return ""
        parts = [script.message]
        if script.breathing_steps:
            parts.extend(script.breathing_steps)
        return " ".join(parts)
