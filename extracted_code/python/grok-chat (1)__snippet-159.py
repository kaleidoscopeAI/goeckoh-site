def __init__(self):
    # Pure Python spell dict (small sample; expand)
    self.spell_dict = {"imrpove": "improve", "clonning": "cloning", "dependancy": "dependency"}
    # Grammar rules: Simple patterns
    self.rules = [
        (r"\bi\s+is\b", "I am"),  # Subject-verb
        (r"\byou\s+is\b", "you are"),
    ]

def correct(self, text: str) -> str:
    # Spell correction
    words = text.split()
    corrected_words = [self.spell_dict.get(word.lower(), word) for word in words]
    corrected = " ".join(corrected_words)

    # Grammar rules
    for pattern, replacement in self.rules:
        corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)

    return corrected
