"""Analyze emotional content of text"""

POSITIVE_WORDS = {'good', 'great', 'excellent', 'amazing', 'wonderful', 
                  'fantastic', 'love', 'happy', 'joy', 'success'}
NEGATIVE_WORDS = {'bad', 'terrible', 'awful', 'hate', 'sad', 'fail', 
                  'problem', 'error', 'issue', 'difficult'}
HIGH_AROUSAL_WORDS = {'excited', 'urgent', 'critical', 'breakthrough', 
                      'revolutionary', 'shocking', 'amazing'}

def analyze(self, text: str) -> EmotionalSignature:
    """Extract emotional signature from text"""
    words = text.lower().split()
    total_words = len(words)

    if total_words == 0:
        return EmotionalSignature()

    # Valence calculation
    positive_count = sum(1 for w in words if w in self.POSITIVE_WORDS)
    negative_count = sum(1 for w in words if w in self.NEGATIVE_WORDS)
    valence = (positive_count - negative_count) / (total_words + 1)
    valence = np.tanh(valence * 10)  # Normalize to [-1, 1]

    # Arousal calculation
    arousal_count = sum(1 for w in words if w in self.HIGH_AROUSAL_WORDS)
    arousal = min(arousal_count / (total_words + 1) * 10, 1.0)

    # Coherence (based on sentence structure)
    sentences = text.split('.')
    avg_sentence_length = len(words) / (len(sentences) + 1)
    coherence = 1.0 / (1.0 + np.exp(-(avg_sentence_length - 15) / 5))

    return EmotionalSignature(
        valence=float(valence),
        arousal=float(arousal),
        coherence=float(coherence),
        semantic_temperature=1.0 + 0.5 * arousal
    )

