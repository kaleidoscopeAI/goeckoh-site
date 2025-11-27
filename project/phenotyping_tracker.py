# ECHO_V4_UNIFIED/phenotyping_tracker.py
from __future__ import annotations
import time
from enum import Enum
from typing import Optional, List
from dataclasses import dataclass, field
import numpy as np

class FragmentClass(Enum):
    """Enumeration for the classification of a vocal fragment."""
    CLEAR_SPEECH = "clear_speech"
    AMBIGUOUS = "ambiguous"
    NONVERBAL = "nonverbal"

@dataclass
class FragmentMetadata:
    """Dataclass to hold all metadata for a single classified vocal fragment."""
    timestamp: float
    # audio: np.ndarray  # Storing the raw audio in a list can consume a lot of memory.
                         # It's better to store a path to the saved snippet.
    snippet_id: int
    vad_score: float
    asr_confidence: float
    triggered_by: str # e.g., "ASR", "PPP", "Guardian"
    classification: FragmentClass
    annotation: Optional[str] = None # e.g., user-verified label

class PhenotypingTracker:
    """
    Tracks and classifies vocal fragments over time to build a long-term
    picture of the user's expressive development.
    """
    def __init__(self):
        self.fragments: List[FragmentMetadata] = []

    def classify_fragment(self, vad_score: float, asr_confidence: float) -> FragmentClass:
        """
        Classifies a fragment based on VAD and ASR scores.
        """
        if asr_confidence > 0.85 and vad_score > 0.7:
            return FragmentClass.CLEAR_SPEECH
        elif vad_score > 0.3:
            return FragmentClass.AMBIGUOUS
        else:
            return FragmentClass.NONVERBAL

    def log_fragment(
        self,
        snippet_id: int,
        vad_score: float,
        asr_confidence: float,
        trigger: str
    ) -> FragmentMetadata:
        """
        Classifies a new fragment and logs its metadata.
        """
        classification = self.classify_fragment(vad_score, asr_confidence)
        
        metadata = FragmentMetadata(
            timestamp=time.time(),
            snippet_id=snippet_id,
            vad_score=vad_score,
            asr_confidence=asr_confidence,
            triggered_by=trigger,
            classification=classification,
        )
        self.fragments.append(metadata)
        # In a real implementation, this would also be persisted to the DB.
        
        return metadata
