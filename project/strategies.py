from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True, slots=True)
class Strategy:
    """Represents a single ABA-inspired calming strategy."""

    category: str
    title: str
    description: str
    cues: List[str] = field(default_factory=list)


STRATEGIES: List[Strategy] = [
    Strategy(
        category="sensory_tools",
        title="Sensory Toolkit",
        description=(
            "Offer weighted blankets, noise-canceling headphones, or fidgets to dampen overload "
            "before and after stressful events."
        ),
        cues=["sensory overload", "loud spaces", "preparation"],
    ),
    Strategy(
        category="predictable_routines",
        title="Visual Schedules + Safe Spaces",
        description="Keep a visual routine and a designated calm corner so transitions feel safe.",
        cues=["transitions", "school prep", "bedtime"],
    ),
    Strategy(
        category="social_stories",
        title="Social Stories & Visual Scripts",
        description="Preview changes or social events with short narratives or picture cards.",
        cues=["appointments", "community outings"],
    ),
    Strategy(
        category="mindfulness",
        title="Mindfulness + Breath Work",
        description="Guide three slow breaths or grounding games to lower anxiety spikes.",
        cues=["early meltdown signs", "evenings"],
    ),
    Strategy(
        category="movement_breaks",
        title="Heavy Work / Movement Breaks",
        description="Schedule wall pushes, trampoline jumps, or resistance games to release energy.",
        cues=["hyperactivity", "pre-lesson"],
    ),
    Strategy(
        category="meltdown_first_aid",
        title="Meltdown First Aid Plan",
        description="Dim lights, remove extra stimuli, and offer preferred sensory tools immediately.",
        cues=["meltdown", "de-escalation"],
    ),
    Strategy(
        category="communication",
        title="Plain-Language Modeling & PECS",
        description="Use short, literal instructions and personalized PECS boards.",
        cues=["communication goals", "nonverbal support"],
    ),
    Strategy(
        category="personalization",
        title="Interest-Based Learning Moments",
        description="Fold the child's passions into routines (dinosaur bath stories, train schedules).",
        cues=["motivation", "daily hygiene"],
    ),
    Strategy(
        category="caregiver_self_care",
        title="Caregiver Self-Care & Respite",
        description="Schedule breaks/support groups so adults can stay regulated.",
        cues=["caregiver stress", "fatigue"],
    ),
]


EVENT_TO_CATEGORIES: Dict[str, List[str]] = {
    "anxious": ["mindfulness", "sensory_tools"],
    "perseveration": ["personalization", "communication"],
    "high_energy": ["movement_breaks", "sensory_tools"],
    "meltdown": ["meltdown_first_aid", "sensory_tools", "mindfulness"],
    "encouragement": ["personalization", "communication"],
}


class StrategyAdvisor:
    """Returns curated strategies for caregiver dashboards."""

    def suggest(self, event: str, limit: int = 3) -> List[Strategy]:
        cats = EVENT_TO_CATEGORIES.get(event)
        if not cats:
            return STRATEGIES[:limit]
        ordered: List[Strategy] = []
        for cat in cats:
            ordered.extend([s for s in STRATEGIES if s.category == cat])
        return (ordered or STRATEGIES)[:limit]