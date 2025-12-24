"""
Package shim to expose psychoacoustic core modules under ``cloning_bubble.core``.
"""

from .attempt_analysis import AttemptFeatures, analyze_chunk
from .bubble_foam import (
    BubbleState,
    compute_bubble_state,
    compute_bubble_state_vertices,
)
from .bubble_synthesizer import (
    MockVocoder,
    controls_to_attempt_features,
    feed_text_through_bubble,
)
from .voice_profile import SpeakerProfile, VoiceFingerprint

__all__ = [
    "AttemptFeatures",
    "analyze_chunk",
    "BubbleState",
    "compute_bubble_state",
    "compute_bubble_state_vertices",
    "MockVocoder",
    "controls_to_attempt_features",
    "feed_text_through_bubble",
    "SpeakerProfile",
    "VoiceFingerprint",
]
