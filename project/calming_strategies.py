"""Evidence-based calming strategy catalog inspired by ABA best practices."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass(frozen=True, slots=True)
class Strategy:
    category: str
    title: str
    description: str
    cues: List[str] = field(default_factory=list)


STRATEGIES: List[Strategy] = [
    Strategy(
        category="sensory_tools",
        title="Sensory Toolkit",
        description=(
            "Offer weighted blankets, noise-canceling headphones, fidgets, or stress balls to reduce sensory overload "
            "before/after stressful events."
        ),
        cues=["sensory overload", "loud spaces", "preparation"],
    ),
    Strategy(
        category="predictable_routines",
        title="Visual Schedules + Safe Spaces",
        description=(
            "Keep a predictable daily rhythm with visual schedules and a designated calm corner so transitions feel safe."
        ),
        cues=["transitions", "school prep", "bedtime"],
    ),
    Strategy(
        category="social_stories",
        title="Social Stories & Visual Scripts",
        description=(
            "Use short narratives or picture cards to preview upcoming changes, medical visits, or social interactions."
        ),
        cues=["appointments", "community outings"],
    ),
    Strategy(
        category="collaboration",
        title="Parent-Teacher Sync",
        description=(
            "Share progress between caregivers, teachers, and therapists weekly so supports stay consistent across home/school."
        ),
        cues=["IEP planning", "new routines"],
    ),
    Strategy(
        category="mindfulness",
        title="Mindfulness + Breath Work",
        description=(
            "Guide the child through deep breathing, grounding games, or short mindfulness stories to lower anxiety spikes."
        ),
        cues=["early meltdown signs", "evenings"],
    ),
    Strategy(
        category="meltdown_first_aid",
        title="Meltdown First Aid Plan",
        description=(
            "Watch for early warning signs (withdrawal, pacing, louder stims), dim lights, remove extra stimuli, and offer "
            "preferred sensory tools."
        ),
        cues=["meltdown", "de-escalation"],
    ),
    Strategy(
        category="movement_breaks",
        title="Heavy Work / Movement Breaks",
        description=(
            "Schedule wall pushes, trampoline jumps, obstacle courses, or resistance band games to release energy before learning blocks."
        ),
        cues=["hyperactivity", "pre-lesson"],
    ),
    Strategy(
        category="therapeutic_programs",
        title="CBT / ACT / Mindfulness Programs",
        description=(
            "Coordinate with clinicians to implement CBT, ACT, or mindfulness curricula that build flexible thinking and acceptance."
        ),
        cues=["therapy goals", "general anxiety"],
    ),
    Strategy(
        category="personalization",
        title="Interest-Based Learning Moments",
        description=(
            "Observe the child's passions and fold them into routines (e.g., dinosaur bath stories, train-themed schedules) to keep engagement high."
        ),
        cues=["motivation", "daily hygiene"],
    ),
    Strategy(
        category="communication",
        title="Plain-Language Modeling & PECS",
        description=(
            "Use short, literal instructions, model speech during routines, and personalize PECS boards with household photos to support expressive language."
        ),
        cues=["communication goals", "nonverbal support"],
    ),
    Strategy(
        category="social_skills",
        title="Structured Turn-Taking & Social Groups",
        description=(
            "Turn board games, sibling play, and park visits into chances to practice sharing, waiting, and using polite requests; supplement with social skills groups."
        ),
        cues=["playdates", "community outings"],
    ),
    Strategy(
        category="education_advocacy",
        title="Prepared IEP Advocacy",
        description=(
            "Document wins and challenges, arrive at IEP meetings with specific goals, and communicate professionally to secure OT, speech, or classroom supports."
        ),
        cues=["school meetings", "progress reviews"],
    ),
    Strategy(
        category="aba_supports",
        title="ABA & Respite Exploration",
        description=(
            "Consult ABA providers for individualized programs targeting self-care, play, or feeding skills, and leverage respite services to recharge caregivers."
        ),
        cues=["new skill targets", "caregiver burnout"],
    ),
    Strategy(
        category="caregiver_self_care",
        title="Caregiver Self-Care & Respite",
        description=(
            "Schedule breaks, join support groups, and utilize respite offerings so you can remain a calm, effective advocate for your child."
        ),
        cues=["caregiver stress", "fatigue"],
    ),
]

EVENT_TO_CATEGORIES: Dict[str, List[str]] = {
    "meltdown": ["meltdown_first_aid", "sensory_tools", "mindfulness"],
    "transition": ["predictable_routines", "social_stories", "movement_breaks"],
    "anxious_speech": ["mindfulness", "sensory_tools", "therapeutic_programs"],
    "anxious": ["mindfulness", "sensory_tools", "therapeutic_programs"],
    "hyperactivity": ["movement_breaks", "sensory_tools"],
    "high_energy": ["movement_breaks", "sensory_tools"],
    "care_team_sync": ["collaboration", "education_advocacy"],
    "school_meeting": ["education_advocacy", "communication"],
    "communication_practice": ["communication", "social_skills"],
    "caregiver_reset": ["caregiver_self_care"],
    "interest_planning": ["personalization"],
    "perseveration": ["personalization", "communication"],
    "encouragement": ["personalization", "social_skills"],
}


def list_categories() -> List[str]:
    return sorted({s.category for s in STRATEGIES})


def by_category(category: str) -> List[Strategy]:
    return [s for s in STRATEGIES if s.category == category]


def suggest_for_event(event: str) -> List[Strategy]:
    cats = EVENT_TO_CATEGORIES.get(event)
    if not cats:
        return STRATEGIES[:3]
    ordered: List[Strategy] = []
    for cat in cats:
        ordered.extend(by_category(cat))
    return ordered or STRATEGIES[:3]


class StrategyAdvisor:
    """Tiny helper used by the realtime loop for context-aware nudges."""

    def __init__(self) -> None:
        self.events_seen: Dict[str, int] = {}

    def suggest(self, event: str) -> List[Strategy]:
        self.events_seen[event] = self.events_seen.get(event, 0) + 1
        return suggest_for_event(event)
