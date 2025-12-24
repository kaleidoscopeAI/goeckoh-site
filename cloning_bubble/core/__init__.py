"""
Package shim to expose psychoacoustic core modules under ``cloning_bubble.core``.

This mirrors the top-level modules so imports like
``from cloning_bubble.core.attempt_analysis import analyze_chunk`` work in tests
without requiring a full package layout.
"""

import sys

import attempt_analysis as _attempt_analysis
import bubble_foam as _bubble_foam
import bubble_synthesizer as _bubble_synthesizer
import voice_profile as _voice_profile

# Register submodules for direct imports (e.g., cloning_bubble.core.voice_profile)
sys.modules[__name__ + ".attempt_analysis"] = _attempt_analysis
sys.modules[__name__ + ".voice_profile"] = _voice_profile
sys.modules[__name__ + ".bubble_foam"] = _bubble_foam
sys.modules[__name__ + ".bubble_synthesizer"] = _bubble_synthesizer

from attempt_analysis import AttemptFeatures, analyze_chunk  # noqa: E402,F401
from bubble_foam import (  # noqa: E402,F401
    BubbleState,
    compute_bubble_state,
    compute_bubble_state_vertices,
)
from bubble_synthesizer import (  # noqa: E402,F401
    MockVocoder,
    controls_to_attempt_features,
    feed_text_through_bubble,
)
from voice_profile import SpeakerProfile, VoiceFingerprint  # noqa: E402,F401

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
