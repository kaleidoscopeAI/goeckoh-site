def simulate_audio_input():
    return input("Speak (type phrase): ").strip()

def correct_to_first_person(text):
    text = re.sub(r"\byou\b", "I", text, flags=re.IGNORECASE)
    text = re.sub(r"\byour\b", "my", text, flags=re.IGNORECASE)
    return text.capitalize()

