import re


FIRST_PERSON_MAP = {
    r"\b(my son wants|he wants|he want)\b": "I want",
    r"\b(I want the|I want a)\s*$": "I want",
    r"\b(want ball|want the ball)\b": "I want the ball",
    r"\b(can I have)\b": "I want",
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def enforce_first_person(text: str) -> str:
    t = text.lower()
    for pattern, repl in FIRST_PERSON_MAP.items():
        t = re.sub(pattern, repl.lower(), t)
    # Transform leading "you" to "I"
    t = re.sub(r"^you\b", "I", t)
    # simple you->I, your->my when clearly self-directed
    t = re.sub(r"\byour\b", "my", t)
    t = re.sub(r"\byou\b", "I", t)
    return t


def clean_asr_text(raw_text: str) -> str:
    """Turn raw Whisper output into a short, first-person sentence."""
    t = normalize_whitespace(raw_text)
    t = enforce_first_person(t)
    # Capitalize first letter, ensure period at end
    if t:
        t = t[0].upper() + t[1:]
        if not re.search(r"[.!?]$", t):
            t += "."
    return t
