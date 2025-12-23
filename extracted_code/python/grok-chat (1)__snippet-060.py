def simulate_audio_input():
    # Simulate user speech input
    return input("Speak (type phrase): ").strip()

def correct_speech(text):
    # Simple heuristic correction (expand rules)
    text = text.lower().capitalize()
    text = text.replace("i is", "I am")
    return text

def compute_gcl(emotions):
    # Global Coherence Level: Mean coherence from emotions
    return sum(math.tanh(e) for e in emotions) / len(emotions) if emotions else 0.0

