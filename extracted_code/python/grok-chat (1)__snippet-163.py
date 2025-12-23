# Global Coherence Level: Mean coherence from emotions
return sum(math.tanh(e) for e in emotions) / len(emotions) if emotions else 0.0

