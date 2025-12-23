# phoneme_utils.py
from enum import Enum, auto
from typing import List

import numpy as np


class PhonemeClass(Enum):
    VOWEL = auto()
    PLOSIVE = auto()
    FRICATIVE = auto()
    NASAL = auto()
    APPROXIMANT = auto()
    OTHER = auto()


# crude grapheme-to-phoneme mapping for English
_VOWELS = set("aeiou")
_PLOSIVES = set("pbtdkgcq")
_FRICATIVES = set("fszvxh")
_NASALS = set("mn")
_APPROXIMANTS = set("rlwyj")


def _class_for_char(ch: str) -> PhonemeClass:
    ch = ch.lower()
    if ch in _VOWELS:
        return PhonemeClass.VOWEL
    if ch in _PLOSIVES:
        return PhonemeClass.PLOSIVE
    if ch in _FRICATIVES:
        return PhonemeClass.FRICATIVE
    if ch in _NASALS:
        return PhonemeClass.NASAL
    if ch in _APPROXIMANTS:
        return PhonemeClass.APPROXIMANT
    return PhonemeClass.OTHER


def text_to_phoneme_classes(text: str) -> List[PhonemeClass]:
    """Very simple grapheme-based phoneme proxy."""
    out: List[PhonemeClass] = []
    for ch in text:
        if ch.isalpha():
            out.append(_class_for_char(ch))
    if not out:
        out.append(PhonemeClass.OTHER)
    return out


def compute_spikiness_envelope(
    text: str,
    length: int,
    base_sharpness: float,
    plosive_boost: float = 0.4,
    fricative_boost: float = 0.25,
) -> np.ndarray:
    """
    Map text -> time-varying spikiness envelope in [0,1].

    Rough idea:
      - Vowels  -> rounder (Bouba).
      - Plosives/fricatives -> spikier (Kiki).
      - We create a discrete sequence over characters, then upsample to length.
    """
    pcs = text_to_phoneme_classes(text)
    n = len(pcs)

    char_spike = np.zeros(n, dtype=np.float32)
    for i, cls in enumerate(pcs):
        val = base_sharpness
        if cls is PhonemeClass.PLOSIVE:
            val += plosive_boost
        elif cls is PhonemeClass.FRICATIVE:
            val += fricative_boost
        # limit to [0,1]
        char_spike[i] = np.clip(val, 0.0, 1.0)

    # Upsample char-level array to control-length array
    if n == 1:
        env = np.full(length, char_spike[0], dtype=np.float32)
    else:
        src_x = np.linspace(0.0, 1.0, n)
        dst_x = np.linspace(0.0, 1.0, length)
        env = np.interp(dst_x, src_x, char_spike)

    return env.astype(np.float32)
