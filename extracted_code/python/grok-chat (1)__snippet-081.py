def simulate_audio_input():
    return input("Speak (type phrase): ").strip()

def correct_to_first_person(text):
    # Pure string: Replace "you" with "I", etc.
    text = text.replace(" you ", " I ").replace(" your ", " my ")
    return text[0].upper() + text[1:]

