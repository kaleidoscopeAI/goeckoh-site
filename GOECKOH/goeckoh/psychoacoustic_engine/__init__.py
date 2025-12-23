"""
Psychoacoustic Engine primitives.

Modules here are self-contained and deterministic: no random number
generation is used so the same inputs yield the same visuals/audio.
"""

from .attempt_analysis import AttemptFeatures, analyze_attempt
from .bubble_foam import BubbleState, compute_bubble_state
from .voice_field import generate_voice_field, procedural_phase, generate_sh_modes
from .voice_profile import VoiceFingerprint, SpeakerProfile
from .voice_logger import log_voice_characteristics
from .bubble_synthesizer import MockVocoder, feed_text_through_bubble

__all__ = [
    "AttemptFeatures",
    "analyze_attempt",
    "BubbleState",
    "compute_bubble_state",
    "generate_voice_field",
    "procedural_phase",
    "generate_sh_modes",
    "MockVocoder",
    "VoiceFingerprint",
    "SpeakerProfile",
    "log_voice_characteristics",
    "feed_text_through_bubble",
]
