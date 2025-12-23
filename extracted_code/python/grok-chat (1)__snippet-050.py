class PragmaticAnalysis:
    type: PragmaticType
    confidence: float
    adjusted_text: str  # Rephrased for clarity if needed
    social_intent: str  # e.g., "request help", "express frustration"

class PragmaticEngine:
    def __init__(self):
        # Patterns for pragmatics (expand with ML later)
        self.idiom_patterns = [r"kick the bucket", r"piece of cake", r"break a leg"]
        self.sarcasm_markers = ["yeah right", "oh sure", "as if"]
        self.social_cues = {
            "help": ["can you", "please", "need"],
            "frustration": ["ugh", "why", "can't"],
        }

    def analyze(self, text: str, semantic_embed: np.ndarray) -> PragmaticAnalysis:
        text_lower = text.lower()

        # Detect idioms
        for pattern in self.idiom_patterns:
            if re.search(pattern, text_lower):
                return PragmaticAnalysis("idiom", 0.9, text, "figurative expression")

        # Detect sarcasm (simple heuristic)
        if any(marker in text_lower for marker in self.sarcasm_markers):
            return PragmaticAnalysis("sarcasm", 0.8, text, "opposite intent")

        # Social cues
        for intent, cues in self.social_cues.items():
            if any(cue in text_lower for cue in cues):
                return PragmaticAnalysis("social_cue", 0.85, text, intent)

        # Default literal
        return PragmaticAnalysis("literal", 1.0, text, "direct statement")

